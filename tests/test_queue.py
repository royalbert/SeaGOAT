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


def test_cache_snapshots_are_bounded_by_time_not_by_batches(mocker, repo):
    """The cache is one pickle of the whole repository's data, so each snapshot costs time
    proportional to the repository. Snapshotting after every chunk makes indexing quadratic in
    repository size. A snapshot is taken only once a batch has reached storage, at most once per
    interval, and always on the final flush."""
    from seagoat.engine import CACHE_PERSIST_INTERVAL_SECONDS, Engine

    repo.add_file_change_commit(
        file_name="many_lines.py",
        contents="".join(f"value_{i} = {i}\n" for i in range(200)),
        author=repo.actors["John Doe"],
        commit_message="Add a file that spans several batches",
    )
    engine = Engine(repo.working_dir)
    clock = mocker.patch("seagoat.engine.time")
    clock.monotonic.return_value = 1000.0
    persist = mocker.patch.object(engine.cache, "persist")
    engine.repository.analyze_files()
    chunks = [
        chunk
        for file, _ in engine.repository.top_files()
        for chunk in file.get_chunks()
        if chunk.chunk_id not in engine.cache.data["chunks_already_analyzed"]
    ]
    batch_size = engine.config["server"]["chroma"]["batchSize"]
    assert len(chunks) > 2 * batch_size, "fixture repo must span several batches"

    for chunk in chunks[: batch_size - 1]:
        engine.process_chunk(chunk)
    # nothing has reached storage yet, so nothing may be recorded as analyzed on disk
    assert persist.call_count == 0

    half = len(chunks) // 2
    for chunk in chunks[batch_size - 1 : half]:
        engine.process_chunk(chunk)
    # the first batch to reach storage snapshots; later ones in the interval do not
    assert persist.call_count == 1

    clock.monotonic.return_value = 1000.0 + CACHE_PERSIST_INTERVAL_SECONDS
    for chunk in chunks[half:]:
        engine.process_chunk(chunk)
    # the interval has passed, so the next batch to reach storage snapshots again
    assert persist.call_count == 2

    engine.flush()
    # the final flush always snapshots, whatever the clock says
    assert persist.call_count == 3
    # far fewer snapshots than batches, let alone chunks
    assert persist.call_count < len(chunks) // batch_size


def test_the_final_partial_batch_reaches_the_vector_store(repo):
    """Chunks are written in batches. When the queue runs out of work the last, partial batch must
    be written too: the cache already records those chunks as analyzed, so a batch left in memory
    is never retried and those lines are missing from semantic search for good."""
    from pathlib import Path

    import chromadb
    from chromadb.config import Settings

    from seagoat.cache import Cache
    from seagoat.engine import Engine
    from seagoat.queue.task_queue import TaskQueue

    repo.add_file_change_commit(
        file_name="lines.py",
        contents="".join(f"value_{i} = {i}\n" for i in range(54)),
        author=repo.actors["John Doe"],
        commit_message="Add a file whose chunks do not fill the last batch",
    )
    engine = Engine(repo.working_dir)
    engine.repository.analyze_files()
    chunks = [
        chunk
        for file, _ in engine.repository.top_files()
        for chunk in file.get_chunks()
    ]
    batch_size = engine.config["server"]["chroma"]["batchSize"]
    assert len(chunks) % batch_size, "fixture must leave a partial final batch"

    context = {"seagoat_engine": engine}
    busy, drained = Mock(), Mock()
    busy._task_queue.qsize.return_value = 1
    drained._task_queue.qsize.return_value = 0
    for chunk in chunks[:-1]:
        TaskQueue.handle_analyze_chunk(busy, context, chunk)
    TaskQueue.handle_analyze_chunk(drained, context, chunks[-1])

    store = Cache("chroma", Path(repo.working_dir), {}).get_cache_folder()
    client = chromadb.PersistentClient(
        path=str(store), settings=Settings(anonymized_telemetry=False)
    )
    stored = client.get_collection("code_data").count()
    assert stored == len(chunks)


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
