
import fcntl
import json
from pathlib import Path

from semantic_search.query_embedding import (
    cache_key,
    load_cached_embedding,
    validate_query,
)
from semantic_search.query_requests import reserve_query


class QueryQuotaExceeded(RuntimeError):
    """The configured request reservation limit was reached."""


def reserve_with_quota(query, journal_dir, cache_dir, budget_path):
    """Atomically check quota and reserve one new query.

    This function NEVER invokes AWS. It does not authorize
    paid execution. Reservations are conservatively counted
    even when their outcomes are uncertain.
    """
    query = validate_query(query)
    journal_dir = Path(journal_dir)
    cache_dir = Path(cache_dir)
    budget_path = Path(budget_path)

    config = json.loads(
        budget_path.read_text(encoding="utf-8")
    )

    if config.get("region") != "us-east-1":
        raise ValueError("Unexpected AWS region")

    if config.get("model_id") != "amazon.titan-embed-text-v2:0":
        raise ValueError("Unexpected embedding model")

    if config.get("dimensions") != 512:
        raise ValueError("Unexpected embedding dimensions")

    if config.get("normalize") is not True:
        raise ValueError("Embedding normalization must be enabled")

    limit = config.get("maximum_new_query_requests")

    if type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError("Invalid request budget")

    # Authorization is enforced by the query runtime.
    # This module only enforces the reservation quota.
    if type(config.get("paid_query_execution_authorized")) is not bool:
        raise ValueError("Invalid authorization configuration")

    journal_dir.mkdir(parents=True, exist_ok=True)

    # flock serializes quota accounting across local processes.
    # The lock file may persist safely between executions.
    lock_path = journal_dir / ".quota.lock"

    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        try:
            key = cache_key(query)

            cached = load_cached_embedding(
                query, cache_dir
            )

            if cached is not None:
                return {
                    "status": "cached",
                    "cache_key": key,
                }

            # Every existing reservation consumes a slot,
            # regardless of its execution outcome.
            reservations = list(journal_dir.glob("*.json"))

            if len(reservations) >= limit:
                raise QueryQuotaExceeded(
                    f"Query reservation quota reached: {limit}"
                )

            # reserve_query uses O_EXCL to prevent duplicates.
            return reserve_query(
                query, journal_dir, cache_dir
            )
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
