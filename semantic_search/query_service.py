
from pathlib import Path

from semantic_search.query_embedding import (
    load_cached_embedding,
    validate_query,
)
from semantic_search.retrieval import SemanticRetriever


class QueryNotCached(RuntimeError):
    """Query needs an embedding before offline search."""


class OfflineQueryService:
    """Perform semantic retrieval using existing cached embeddings."""

    def __init__(self, cache_dir, index_size=312):
        self.cache_dir = Path(cache_dir)
        self.retriever = SemanticRetriever(index_size=index_size)

    def search(self, query, top_k=5):
        query = validate_query(query)

        cached = load_cached_embedding(
            query, self.cache_dir
        )

        if cached is None:
            raise QueryNotCached(
                "No embedding cached for this query. "
                "No AWS request was made."
            )

        return self.retriever.search(
            cached["embedding"],
            top_k=top_k,
        )
