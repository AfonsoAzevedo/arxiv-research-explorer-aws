# Handle filesystems path
from pathlib import Path

from semantic_search.query_embedding import (
    load_cached_embedding,  # Retrieves an existing query embedding
    validate_query,  # Validates the research query
)

# Perform vector-based paper retrieval
from semantic_search.retrieval import SemanticRetriever

# Create a specific error (will be used on main.py)
class QueryNotCached(RuntimeError):
    """Query needs an embedding before offline search"""


class OfflineQueryService:
    """Perform semantic retrieval using existing cached embeddings"""

    def __init__(self, cache_dir, index_size=312, corpus_path=None):
        self.cache_dir = Path(cache_dir)

        # Use a custom metadata file when one is provided
        self.retriever = SemanticRetriever(
            index_size=index_size,
            corpus_path=corpus_path,
        )

    def search(self, query, top_k=5):
        query = validate_query(query)

        # Search for an existing embedding
        cached = load_cached_embedding(query, self.cache_dir)

        # Reject uncached queries
        if cached is None:
            raise QueryNotCached(
                "No embedding cached for this query. No AWS request was made."
            )

        # Perform retrieval
        return self.retriever.search(
            cached["embedding"],
            top_k=top_k,
        )
