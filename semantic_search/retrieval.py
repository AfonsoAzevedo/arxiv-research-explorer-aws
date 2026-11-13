# Reads paper identifiers and metadata
import json

# Locates saved files
from pathlib import Path

# Loads and searches the vector index
import faiss

# Processes numerical embedding vectors
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SUPPORTED_INDEX_SIZES = (312, 368)
CORPUS_PATH = DATA / "retrieval_25k.jsonl"
DIMENSIONS = 512

class SemanticRetriever:
    def __init__(self, index_size=312, corpus_path=None):
        # Validate the requested index
        if type(index_size) is not int or index_size not in SUPPORTED_INDEX_SIZES:
            raise ValueError("Supported index sizes: 312 or 368")

        # Locate the FAISS files
        index_directory = DATA / f"faiss_{index_size}"
        index_path = (
            index_directory / "index.faiss"
        )  # index.faiss Stores the FAISS index containing numerical vectors and the structures used to search them
        mapping_path = (
            index_directory / "paper_ids.json"
        )  # paper_ids.json Stores the paper identifiers associated with positions in that index

        # Load the index and paper identifiers
        self.index = faiss.read_index(str(index_path))
        # Read the paper-ID mapping
        self.paper_ids = json.loads(mapping_path.read_text(encoding="utf-8"))

        # Validate the FAISS index
        assert self.index.d == DIMENSIONS  # FAISS uses 512-dimensional vectors
        assert (
            self.index.metric_type == faiss.METRIC_INNER_PRODUCT
        )  # FAISS uses inner-product similarity
        assert (
            self.index.ntotal == len(self.paper_ids) == index_size
        )  # The index and ID mapping contain the expected number of entries
        assert (
            len(set(self.paper_ids)) == index_size
        )  # Every paper identifier is unique

        # Load the paper metadata
        wanted = set(
            self.paper_ids
        )  # Contains the paper indetifiers stored in the FAISS index
        self.papers = {}

        # Allow a smaller metadata file while keeping the original default
        selected_corpus = Path(corpus_path) if corpus_path is not None else CORPUS_PATH

        # Reading the metadata
        # Parse its JSON record -> Extract its paper ID -> Check whether that ID belongs to the FAISS index -> Store the paper's metadata if it matches
        with selected_corpus.open(encoding="utf-8") as file:
            for line in file:
                paper = json.loads(line)
                pid = paper["paper_id"]

                if pid in wanted:
                    assert pid not in self.papers
                    self.papers[pid] = paper

        assert (
            set(self.papers) == wanted
        )  # Checks that metadata was found for every paper in the index

    def search(self, embedding, top_k=5, exclude_id=None):
        # Validate how many papers I want
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")

        # Convert the embedding into a numpy array (cached embeddings are stored as list of numbers in JSON files)
        vector = np.asarray(embedding, dtype=np.float32)

        # Input validation
        if vector.shape != (DIMENSIONS,):
            raise ValueError("Expected a 512-dimensional vector")

        if not np.isfinite(vector).all():
            raise ValueError("Embedding contains nonfinite values")

        # Calculate the norm
        norm = float(np.linalg.norm(vector))

        # Verifie the vector is approximately normalized
        if not 0.99 <= norm <= 1.01:
            raise ValueError("Embedding must be normalized")

        # Normalize and reshape the vector
        # vector / norm and new vector: (1, 512)
        # Make the array contiguous
        vector = np.ascontiguousarray((vector / norm).reshape(1, -1))

        # Determine how many neighbors to search
        search_k = min(self.index.ntotal, top_k + (1 if exclude_id is not None else 0))

        # Perform the actual FAISS search
        scores, positions = self.index.search(vector, search_k)
        results = []

        for position, score in zip(positions[0], scores[0]):
            if position < 0:  # Skip invalid positions
                continue

            paper_id = self.paper_ids[int(position)]

            # If the retrieved paper matches the excluded ID, the loop skips it
            if paper_id == exclude_id:
                continue

            # Retrieve its metadata
            paper = self.papers[paper_id]

            # Build the result
            results.append(
                {
                    "paper_id": paper_id,
                    "title": paper["title"],
                    "abstract": paper["abstract"],
                    "category": paper["sampling_group"],
                    "score": float(score),
                    "arxiv_url": "https://arxiv.org/abs/" + paper_id,
                }
            )

            # Stop after enough results
            if len(results) == top_k:
                break

        return results
