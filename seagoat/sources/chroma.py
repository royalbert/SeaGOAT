import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from seagoat.cache import Cache
from seagoat.repository import Repository
from seagoat.result import Result
from seagoat.utils.config import get_config_values

MAXIMUM_VECTOR_DISTANCE = 1.5


def get_metadata_and_distance_from_chromadb_result(chromadb_results):
    return (
        list(
            zip(
                chromadb_results["metadatas"][0],
                chromadb_results["distances"][0],
            )
        )
        if chromadb_results["metadatas"] and chromadb_results["distances"]
        else None
    ) or []


def format_results(query_text: str, repository, chromadb_results):
    files = {}

    for metadata, distance in get_metadata_and_distance_from_chromadb_result(
        chromadb_results
    ):
        if distance > MAXIMUM_VECTOR_DISTANCE:
            break
        path = str(metadata["path"])
        line = int(metadata["line"])
        git_object_id = str(metadata["git_object_id"])
        full_path = Path(repository.path) / path

        if not full_path.exists():
            continue

        if not repository.is_up_to_date_git_object(path, git_object_id):
            continue

        gitfile = repository.get_file(path)

        if path not in files:
            files[path] = Result(query_text, gitfile)
        files[path].add_line(line, distance)

    return files.values()


# The largest batch the indexer will embed in one call regardless of configuration.
# Every record in a batch is held in memory with its embedding until the call
# returns, and measured throughput is flat from 256 upward (the embedding model,
# not the batch, is the limit past that point), so larger values only cost memory.
MAX_BATCH_SIZE = 256


def _clamp_batch_size(requested, maximum: int) -> int:
    """Keep the configured batch size inside a safe range.

    A value below 1 would never flush; a value above MAX_BATCH_SIZE or above the
    Chroma client's per-call limit is capped. Every clamp is logged rather than
    allowed to take the indexer down.
    """
    try:
        value = int(requested)
    except (TypeError, ValueError):
        logging.warning("chroma.batchSize %r is not an integer; using 1", requested)
        return 1
    if value < 1:
        logging.warning("chroma.batchSize %d is below 1; using 1", value)
        return 1
    ceiling = min(MAX_BATCH_SIZE, maximum) if maximum else MAX_BATCH_SIZE
    if value > ceiling:
        logging.warning(
            "chroma.batchSize %d exceeds the maximum batch size %d; using %d",
            value,
            ceiling,
            ceiling,
        )
        return ceiling
    return value


def initialize(repository: Repository):
    cache = Cache("chroma", Path(repository.path), {})
    config = get_config_values(Path(repository.path))

    chroma_client = chromadb.PersistentClient(
        path=str(cache.get_cache_folder()),
        settings=Settings(
            anonymized_telemetry=False,
        ),
    )
    embedding_function_name = config["server"]["chroma"]["embeddingFunction"]["name"]
    embedding_function_kwargs = config["server"]["chroma"]["embeddingFunction"][
        "arguments"
    ]
    embedding_function = getattr(embedding_functions, embedding_function_name)(
        **embedding_function_kwargs
    )
    chroma_collection = chroma_client.get_or_create_collection(
        name="code_data", embedding_function=embedding_function
    )

    batch_size = _clamp_batch_size(
        config["server"]["chroma"]["batchSize"], chroma_client.get_max_batch_size()
    )
    batch_buffer = {"ids": [], "documents": [], "metadatas": []}

    def _flush_batch():
        """Write the buffered chunks. Returns True if anything was written."""
        if not batch_buffer["ids"]:
            return False
        chroma_collection.upsert(
            ids=batch_buffer["ids"],
            documents=batch_buffer["documents"],
            metadatas=batch_buffer["metadatas"],
        )
        batch_buffer["ids"].clear()
        batch_buffer["documents"].clear()
        batch_buffer["metadatas"].clear()
        return True

    def fetch(query_text: str, limit: int):
        # Slightly overfetch results as it will sorted using a different score later
        maximum_chunks_to_fetch = 100  # this should be plenty, especially because many times context could be included
        n_results = min((limit + 1) * 2, maximum_chunks_to_fetch)

        chromadb_results = chroma_collection.query(
            query_texts=[query_text],
            n_results=n_results,
        )

        return format_results(query_text, repository, chromadb_results)

    def cache_chunk(chunk):
        batch_buffer["ids"].append(chunk.chunk_id)
        batch_buffer["documents"].append(chunk.chunk)
        batch_buffer["metadatas"].append(
            {
                "path": chunk.path,
                "line": chunk.codeline,
                "git_object_id": chunk.object_id,
            }
        )
        if len(batch_buffer["ids"]) >= batch_size:
            return _flush_batch()
        return False

    def cache_repo():
        # chromadb does not need any repo cache action
        pass

    return {
        "fetch": fetch,
        "cache_chunk": cache_chunk,
        "cache_repo": cache_repo,
        "flush_batch": _flush_batch,
    }
