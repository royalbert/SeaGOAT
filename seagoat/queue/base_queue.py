import logging
import threading
from dataclasses import dataclass, field
from queue import Empty, PriorityQueue
from typing import Any, Dict, Tuple
from uuid import uuid4

HIGH_PRIORITY = 0.0
# Exit statuses a server uses when its worker can no longer do its job (see seagoat.server.ExitCode).
WORKER_CRASHED_EXIT_CODE = 7
REPOSITORY_GONE_EXIT_CODE = 8
MEDIUM_PRIORITY = 0.5
LOW_PRIORITY = 1.0


@dataclass(order=True)
class Task:
    priority: float
    name: str
    args: Tuple[Any, ...] = field(default_factory=tuple, compare=False)
    kwargs: Dict[str, Any] = field(default_factory=dict, compare=False)
    task_id: str = field(default_factory=lambda: uuid4().hex, compare=True)


class BaseQueue:
    def __init__(self, on_fatal=None, **kwargs):
        """on_fatal(exit_code) is called when the worker can no longer do its job. The server
        passes a function that ends the process; without one only the worker thread stops, which
        is what a queue embedded in another process -- a test run, a library user -- needs."""
        self.kwargs = kwargs
        self._on_fatal = on_fatal
        self._task_queue = PriorityQueue()
        self._worker_thread = threading.Thread(target=self._worker_function)
        self._worker_thread.start()

    def _get_context(self) -> Dict[str, Any]:
        return {}

    def enqueue(
        self,
        task_name,
        *args,
        priority=HIGH_PRIORITY,
        wait_for_result=True,
        **kwargs,
    ):
        result_queue = PriorityQueue()
        task = Task(
            priority=priority,
            name=task_name,
            args=args,
            kwargs={**kwargs, "__result_queue": result_queue},
        )
        self._task_queue.put(task)
        if wait_for_result:
            return result_queue.get()
        return None

    def handle_maintenance(self, context):
        pass

    def shutdown(self):
        self._task_queue.put(
            Task(priority=HIGH_PRIORITY, name="shutdown", args=(), kwargs={})
        )
        self._worker_thread.join()

    def _handle_task(self, context, task: Task):
        logging.info("Handling task: %s", task.name)
        handler_name = f"handle_{task.name}"
        handler = getattr(self, handler_name, None)
        if handler:
            kwargs = dict(task.kwargs or {})
            result_queue = kwargs.pop("__result_queue", None)
            result = handler(context, *task.args, **kwargs)
            if result_queue is not None:
                result_queue.put(result)

    def _fatal(self, exit_code):
        if self._on_fatal is not None:
            self._on_fatal(exit_code)

    def _worker_function(self):
        logging.info("Starting worker thread...")
        try:
            self._worker_loop()
        except Exception:
            # The worker is the only thread that analyzes chunks and answers queries; once it
            # has died nothing will again. Report why, then let the owner end the process.
            try:
                logging.exception(
                    "The SeaGOAT worker thread crashed and can no longer analyze or answer "
                    "queries. Fix the cause reported above and start the server again."
                )
            except Exception:
                pass  # a failing log handler must not prevent the report below from acting
            self._fatal(WORKER_CRASHED_EXIT_CODE)

    def _worker_loop(self):
        context = self._get_context()

        while True:
            try:
                task = self._task_queue.get(timeout=0.1)
                if task.name == "shutdown":
                    break
                self._handle_task(context, task)
            except Empty:
                self.handle_maintenance(context)
