
"""Application entry point for DS483 semantic search."""

from pathlib import Path

from semantic_search.bedrock_client import create_bedrock_client
from semantic_search.query_runtime import search_query

BASE = Path(__file__).resolve().parent

BUDGET_PATH = BASE / "config/query_budget.json"
AUTHORIZATION_PATH = BASE / "config/single_query_authorization.json"
CACHE_DIR = BASE / "data/application_query_cache"
JOURNAL_DIR = BASE / "data/application_query_journal"


def search(query, top_k=5, index_size=368):
    """Search papers using the project's configured backend.

    Cached searches do not invoke Bedrock. New queries require
    explicit authorization and an available request quota.
    """
    return search_query(
        query=query,
        cache_dir=CACHE_DIR,
        journal_dir=JOURNAL_DIR,
        budget_path=BUDGET_PATH,
        client_factory=create_bedrock_client,
        top_k=top_k,
        authorization_path=AUTHORIZATION_PATH,
        index_size=index_size,
    )
