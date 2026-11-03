
import json
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SUPPORTED_INDEX_SIZES = (312, 368)
CORPUS_PATH = DATA / "retrieval_25k.jsonl"
DIMENSIONS = 512


class SemanticRetriever:
    def __init__(self, index_size=312):
        if type(index_size) is not int or index_size not in SUPPORTED_INDEX_SIZES:
            raise ValueError("Supported index sizes: 312 or 368")

        index_directory = DATA / f"faiss_{index_size}"
        index_path = index_directory / "index.faiss"
        mapping_path = index_directory / "paper_ids.json"

        self.index = faiss.read_index(str(index_path))
        self.paper_ids = json.loads(
            mapping_path.read_text(encoding="utf-8")
        )

        assert self.index.d == DIMENSIONS
        assert self.index.metric_type == faiss.METRIC_INNER_PRODUCT
        assert self.index.ntotal == len(self.paper_ids) == index_size
        assert len(set(self.paper_ids)) == index_size

        wanted = set(self.paper_ids)
        self.papers = {}

        with CORPUS_PATH.open(encoding="utf-8") as file:
            for line in file:
                paper = json.loads(line)
                pid = paper["paper_id"]

                if pid in wanted:
                    assert pid not in self.papers
                    self.papers[pid] = paper

        assert set(self.papers) == wanted

    def search(self, embedding, top_k=5, exclude_id=None):
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")

        vector = np.asarray(embedding, dtype=np.float32)

        if vector.shape != (DIMENSIONS,):
            raise ValueError("Expected a 512-dimensional vector")

        if not np.isfinite(vector).all():
            raise ValueError("Embedding contains nonfinite values")

        norm = float(np.linalg.norm(vector))

        if not 0.99 <= norm <= 1.01:
            raise ValueError("Embedding must be normalized")

        vector = np.ascontiguousarray(
            (vector / norm).reshape(1, -1)
        )

        search_k = min(
            self.index.ntotal,
            top_k + (1 if exclude_id is not None else 0)
        )

        scores, positions = self.index.search(vector, search_k)
        results = []

        for position, score in zip(positions[0], scores[0]):
            if position < 0:
                continue

            paper_id = self.paper_ids[int(position)]

            if paper_id == exclude_id:
                continue

            paper = self.papers[paper_id]

            results.append({
                "paper_id": paper_id,
                "title": paper["title"],
                "abstract": paper["abstract"],
                "category": paper["sampling_group"],
                "score": float(score),
                "arxiv_url": "https://arxiv.org/abs/" + paper_id,
            })

            if len(results) == top_k:
                break

        return results
