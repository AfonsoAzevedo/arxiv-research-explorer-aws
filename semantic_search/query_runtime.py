
import json
from pathlib import Path

from semantic_search.query_embedding import (
    load_cached_embedding,
    save_cached_embedding,
    validate_embedding_response,
    validate_query,
    REGION,
    MODEL_ID,
    DIMENSIONS,
    NORMALIZE,
)
from semantic_search.query_authorization import verify_query_authorization
from semantic_search.query_quota import reserve_with_quota
from semantic_search.query_service import OfflineQueryService


class QueryExecutionNotAuthorized(RuntimeError):
    """Paid query execution is disabled."""


def search_query(
    query,
    cache_dir,
    journal_dir,
    budget_path,
    client_factory,
    top_k=5,
    authorization_path=None,
    index_size=312,
):
    """Run cached or authorized query retrieval.

    The caller supplies a client factory. This module never
    constructs an AWS client independently.
    """
    query = validate_query(query)

    if type(top_k) is not int or not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20")

    cache_dir = Path(cache_dir)
    journal_dir = Path(journal_dir)
    budget_path = Path(budget_path)

    service = OfflineQueryService(
        cache_dir,
        index_size=index_size,
    )

    cached = load_cached_embedding(query, cache_dir)
    if cached is not None:
        return service.search(query, top_k=top_k)

    config = json.loads(
        budget_path.read_text(encoding="utf-8")
    )

    if config.get("paid_query_execution_authorized") is not True:
        raise QueryExecutionNotAuthorized(
            "Paid query execution is disabled. "
            "No AWS request was made."
        )

    # Both authorization gates are required for uncached queries.
    # The authorization file must match this exact query.
    verify_query_authorization(query, authorization_path)

    # Reserve before constructing the client or invoking AWS.
    reservation = reserve_with_quota(
        query, journal_dir, cache_dir, budget_path
    )

    # Another process might have cached the query while
    # the reservation was being acquired.
    if reservation["status"] == "cached":
        return service.search(query, top_k=top_k)

    assert reservation["status"] == "reserved"

    # Any exception after reservation leaves the journal
    # entry intact. Never automatically retry it.
    client = client_factory()

    response = client.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "inputText": query,
            "dimensions": DIMENSIONS,
            "normalize": NORMALIZE,
        }),
    )

    payload = json.loads(response["body"].read())
    validated = validate_embedding_response(payload)

    save_cached_embedding(
        query, validated, cache_dir
    )

    return service.search(query, top_k=top_k)
