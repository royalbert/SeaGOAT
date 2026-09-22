import logging
from time import sleep
from unittest.mock import Mock

import pytest

from seagoat.queue.task_queue import TaskQueue
from seagoat.repository import Repository


@pytest.fixture(name="create_task_queue")
def create_task_queue(repo):
    def _create_task_queue():
        return TaskQueue(repo_path=repo.working_dir, minimum_chunks_to_analyze=0)

    return _create_task_queue


@pytest.mark.parametrize(
    "chunks_analyzed, unanalyzed, expected_accuracy",
    [
        (0, 0, 100),
        (1, 999999, 1),
        (1000, 0, 100),
        (0, 20, 0),
        (5, 150, 2),
        (50, 450, 11),
        (5, 15, 45),
        (10, 10, 91),
        (100, 100, 91),
        (100_000, 100_001, 91),
        (15, 5, 99),
        (150, 5, 99),
        (150_000, 5, 99),
    ],
)
def test_handle_get_stats(
    create_task_queue, chunks_analyzed, unanalyzed, expected_accuracy
):
    task_queue = create_task_queue()
    context = {
        "seagoat_engine": Mock(),
    }

    context["seagoat_engine"].cache.data = {
        "chunks_already_analyzed": set(range(chunks_analyzed)),
        "chunks_not_yet_analyzed": set(range(unanalyzed)),
    }

    stats = task_queue.handle_get_stats(context)
    task_queue.shutdown()

    assert stats["accuracy"]["percentage"] == expected_accuracy


def test_important_files_are_analyzed_first(create_task_queue, mocker, repo):
    enqueue = mocker.patch("seagoat.queue.task_queue.TaskQueue.enqueue")
    create_task_queue()
    sleep(2.0)
    repository = Repository(repo.working_dir)
    repository.analyze_files()
    order_of_files_analyzed = []
    for call in enqueue.mock_calls:
        path = call.args[1].path
        if not order_of_files_analyzed or order_of_files_analyzed[-1] != path:
            order_of_files_analyzed.append(path)

    # due to sorting by file priority, chunks of the same file should
    # be grouped together
    assert len(set(order_of_files_analyzed)) == len(order_of_files_analyzed)

    # the exact order of files should also match the priority list
    assert [file.path for file, _ in repository.top_files()] == order_of_files_analyzed


def test_worker_crash_is_logged_and_stops_the_process(mocker, repo, caplog):
    """An exception escaping the worker used to kill only the worker thread and leave
    a server that answers nothing. It must be logged and terminate the process."""
    from seagoat.queue.base_queue import WORKER_CRASHED_EXIT_CODE, BaseQueue

    exit_ = mocker.patch("seagoat.queue.base_queue.os._exit")

    class Broken(BaseQueue):
        def _get_context(self):
            raise RuntimeError("boom during context setup")

    queue = Broken()
    queue._worker_thread.join(timeout=5)

    exit_.assert_called_once_with(WORKER_CRASHED_EXIT_CODE)
    assert "worker thread crashed" in caplog.text
    assert "boom during context setup" in caplog.text


def test_worker_crash_report_is_flushed_before_the_process_dies(mocker):
    """os._exit skips the interpreter's cleanup, and that includes flushing buffered streams.
    Without an explicit flush the crash report logged a line earlier is discarded and the process
    dies with no message -- the silent death this handler exists to prevent."""
    from seagoat.queue.base_queue import WORKER_CRASHED_EXIT_CODE, BaseQueue

    order = []

    class _Recording(logging.Handler):
        def emit(self, record):
            pass

        def flush(self):
            order.append("flushed")

    handler = _Recording()
    logging.getLogger().addHandler(handler)
    mocker.patch(
        "seagoat.queue.base_queue.os._exit",
        side_effect=lambda code: order.append(("exited", code)),
    )

    class Broken(BaseQueue):
        def _get_context(self):
            raise RuntimeError("boom during context setup")

    try:
        Broken()._worker_thread.join(timeout=5)
    finally:
        logging.getLogger().removeHandler(handler)

    # flushed at least once, and every flush before the exit
    assert "flushed" in order
    assert order[-1] == ("exited", WORKER_CRASHED_EXIT_CODE)
