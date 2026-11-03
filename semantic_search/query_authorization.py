
"""Single-query authorization validation for DS483 semantic search."""

import json
from pathlib import Path

from semantic_search.query_embedding import cache_key, validate_query


class QueryNotAuthorized(PermissionError):
    """The query is not covered by the authorization record."""


def verify_query_authorization(query, authorization_path):
    """Validate an exact query-scoped authorization record.

    This function does not contact AWS or consume authorization.
    A separate reservation journal must prevent repeated execution.
    """
    query = validate_query(query)

    if authorization_path is None:
        raise QueryNotAuthorized(
            "Single-query authorization path is required"
        )

    if not isinstance(authorization_path, (str, Path)):
        raise QueryNotAuthorized(
            "Invalid single-query authorization path"
        )

    path = Path(authorization_path)

    if not path.is_file():
        raise QueryNotAuthorized(
            "Single-query authorization file does not exist"
        )

    record = json.loads(path.read_text(encoding="utf-8"))

    if record.get("schema_version") != 1:
        raise QueryNotAuthorized("Invalid authorization schema")

    if record.get("authorized") is not True:
        raise QueryNotAuthorized("Query execution not authorized")

    if record.get("maximum_requests") != 1:
        raise QueryNotAuthorized(
            "Authorization must permit exactly one request"
        )

    if record.get("query_cache_key") != cache_key(query):
        raise QueryNotAuthorized(
            "Authorization does not match this query"
        )

    return True
