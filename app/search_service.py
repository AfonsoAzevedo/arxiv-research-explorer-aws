"""
Provide cached-only semantic paper search for the Streamlit app.

The search uses previously generated Titan V2 embeddings and
the existing 368-paper FAISS index without making AWS requests.
"""

from pathlib import Path

from semantic_search.query_service import (
    OfflineQueryService,
    QueryNotCached,
)


# Locate the existing cached embeddings in the project.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = (
    PROJECT_ROOT
    / "semantic_search/data/application_query_cache"
)

INDEX_SIZE = 368


def load_search_service():
    """Load the existing FAISS index for offline semantic search."""

    # Check that the query cache exists before starting the service.
    if not CACHE_DIR.is_dir():
        raise FileNotFoundError(
            f"Semantic search cache not found: {CACHE_DIR}"
        )

    return OfflineQueryService(
        cache_dir=CACHE_DIR,
        index_size=INDEX_SIZE,
    )


def find_related_papers(service, query, top_k=5):
    """Find related papers using an already cached query embedding."""

    # Reject empty queries before searching the index.
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Please enter a research topic.")

    # The offline service rejects uncached queries without calling Bedrock.
    return service.search(query, top_k=top_k)
