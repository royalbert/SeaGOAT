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


class _BrokenQueue:
    """A queue whose worker dies on its first step."""

    @staticmethod
    def make(**kwargs):
        from seagoat.queue.base_queue import BaseQueue

        class Broken(BaseQueue):
            def _get_context(self):
                raise RuntimeError("boom during context setup")

        return Broken(**kwargs)


def test_worker_crash_is_reported_and_handed_to_on_fatal(caplog):
    """A dead worker used to leave a server that answers nothing, with nothing in the logs.
    The crash must be reported and handed to the owner, which decides whether to exit."""
    from seagoat.queue.base_queue import WORKER_CRASHED_EXIT_CODE

    on_fatal = Mock()
    _BrokenQueue.make(on_fatal=on_fatal)._worker_thread.join(timeout=5)

    on_fatal.assert_called_once_with(WORKER_CRASHED_EXIT_CODE)
    assert "worker thread crashed" in caplog.text
    assert "boom during context setup" in caplog.text


def test_a_queue_without_on_fatal_never_ends_its_host_process(mocker):
    """Only the server may end the process. A queue embedded anywhere else -- a test run, a
    library user -- must lose only its worker thread."""
    exit_ = mocker.patch("os._exit")

    queue = _BrokenQueue.make()
    queue._worker_thread.join(timeout=5)

    assert not queue._worker_thread.is_alive()
    exit_.assert_not_called()


def test_worker_crash_reaches_on_fatal_even_if_a_log_handler_raises():
    """A log handler that raises while the crash is reported must not stop the report from
    reaching the owner; otherwise the worker dies silently again."""
    from seagoat.queue.base_queue import WORKER_CRASHED_EXIT_CODE

    class _Raising(logging.Handler):
        def emit(self, record):
            raise OSError("log destination is gone")

        def handleError(self, record):
            raise OSError("log destination is gone")

    handler = _Raising()
    logging.getLogger().addHandler(handler)
    on_fatal = Mock()
    try:
        _BrokenQueue.make(on_fatal=on_fatal)._worker_thread.join(timeout=5)
    finally:
        logging.getLogger().removeHandler(handler)

    on_fatal.assert_called_once_with(WORKER_CRASHED_EXIT_CODE)


def test_maintenance_stops_when_the_repository_is_gone():
    """A vanished repository must stop the worker and be handed to the owner, not fail on every
    maintenance pass or go on to analyze."""
    from seagoat.queue.base_queue import REPOSITORY_GONE_EXIT_CODE
    from seagoat.queue.task_queue import TaskQueue
    from seagoat.repository import RepositoryGone

    queue = Mock()
    engine = Mock()
    engine.repository.get_status_hash.side_effect = RepositoryGone("/tmp/gone")
    context = {
        "seagoat_engine": engine,
        "last_maintenance": None,
        "last_repo_state_hash": None,
    }

    TaskQueue.handle_maintenance(queue, context)

    queue._fatal.assert_called_once_with(REPOSITORY_GONE_EXIT_CODE)
    assert queue._task_queue.put.call_args.args[0].name == "shutdown"
    engine.analyze_codebase.assert_not_called()
