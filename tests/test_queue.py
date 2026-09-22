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
    proportional to the repository, not to the progress since the last one. A snapshot per batch
    is still quadratic in repository size; the number of snapshots has to be bounded by time.
    Within one interval only the first batch snapshots, a batch after the interval snapshots
    again, and the final flush always does."""
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

    half = len(chunks) // 2
    for chunk in chunks[:half]:
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
    # ten batches reached storage; the old per-batch scheme snapshotted eleven times
    assert persist.call_count < len(chunks) // batch_size
