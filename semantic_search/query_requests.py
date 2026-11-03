
import json
import os
from pathlib import Path

from semantic_search.query_embedding import (
    cache_key,
    load_cached_embedding,
    validate_query,
)


class QueryAlreadyReserved(RuntimeError):
    """A query has already been reserved for an AWS request."""


def reserve_query(query, journal_dir, cache_dir):
    """Reserve one query at most once, without invoking AWS.

    A reservation is deliberately retained even if a later
    request fails. It must never be automatically retried.
    """
    query = validate_query(query)
    key = cache_key(query)

    cached = load_cached_embedding(query, cache_dir)
    if cached is not None:
        return {"status": "cached", "cache_key": key}

    journal_dir = Path(journal_dir)
    journal_dir.mkdir(parents=True, exist_ok=True)

    path = journal_dir / f"{key}.json"

    record = {
        "cache_key": key,
        "query": query,
        "status": "reserved",
        "model_id": "amazon.titan-embed-text-v2:0",
        "region": "us-east-1",
        "dimensions": 512,
        "normalize": True,
    }

    payload = json.dumps(
        record, indent=2, allow_nan=False
    ) + "\n"

    try:
        fd = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        # A previous request may be completed or uncertain.
        # Do not silently launch another.
        raise QueryAlreadyReserved(
            "Query already reserved; inspect its outcome "
            "before any further paid request."
        ) from exc

    with os.fdopen(fd, "w", encoding="utf-8") as file:
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())

    dir_fd = os.open(journal_dir, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)

    # Recheck cache in case another process published a result
    # after the first cache check. Never invoke AWS on a hit.
    cached = load_cached_embedding(query, cache_dir)
    if cached is not None:
        return {"status": "cached", "cache_key": key}

    return {
        "status": "reserved",
        "cache_key": key,
        "reservation_path": str(path),
    }
