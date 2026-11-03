import hashlib
import json
from pathlib import Path

import faiss
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path("semantic_search/data")

QUERY = (
    "Machine learning methods for improving "
    "classification robustness and prediction accuracy"
)

CONFIG = {
    "query": QUERY,
    "model_id": "amazon.titan-embed-text-v2:0",
    "region": "us-east-1",
    "dimensions": 512,
    "normalize": True,
}

canonical = json.dumps(
    CONFIG,
    sort_keys=True,
    ensure_ascii=False,
)

key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

papers = [
    json.loads(line)
    for line in (BASE / "pilot_10.jsonl").read_text(encoding="utf-8").splitlines()
]

paper_by_id = {paper["paper_id"]: paper for paper in papers}

mapping = json.loads(
    (BASE / "faiss/paper_mapping.json").read_text()
)

cache = json.loads(
    (BASE / "queries/query_cache.json").read_text()
)

tfidf = joblib.load(BASE / "tfidf/pilot_tfidf.joblib")
index = faiss.read_index(str(BASE / "faiss/pilot.index"))

ids = mapping["paper_ids"]

source_hash = hashlib.sha256(
    (BASE / "pilot_10.jsonl").read_bytes()
).hexdigest()

assert source_hash == mapping["source_sha256"]
assert source_hash == tfidf["source_sha256"]
assert ids == tfidf["paper_ids"]
assert ids == [paper["paper_id"] for paper in papers]
assert index.ntotal == len(ids) == 10
assert index.d == 512

entry = cache["entries"][key]

assert entry["model_id"] == CONFIG["model_id"]
assert entry["region"] == CONFIG["region"]
assert entry["dimensions"] == CONFIG["dimensions"]
assert entry["normalize"] == CONFIG["normalize"]

query_embedding = np.asarray(
    [entry["embedding"]],
    dtype=np.float32,
)

assert query_embedding.shape == (1, 512)
assert np.all(np.isfinite(query_embedding))

faiss.normalize_L2(query_embedding)

# Semantic retrieval
semantic_scores, semantic_indices = index.search(query_embedding, 5)

# Lexical retrieval
query_vector = tfidf["vectorizer"].transform([QUERY])

lexical_scores = cosine_similarity(
    query_vector,
    tfidf["matrix"],
)[0]

lexical_indices = np.argsort(-lexical_scores, kind="stable")[:5]


def display_results(method, positions, scores):
    print(f"\n{method}")
    print("=" * 65)

    ranked_ids = []

    for rank, (position, score) in enumerate(zip(positions, scores), 1):
        paper_id = ids[int(position)]
        paper = paper_by_id[paper_id]

        ranked_ids.append(paper_id)

        print(f"\n{rank}. {paper['title']}")
        print(f"   arXiv ID: {paper_id}")
        print(f"   Similarity: {float(score):.4f}")
        print(f"   https://arxiv.org/abs/{paper_id}")

    return ranked_ids


print("QUERY:", QUERY)
print("CORPUS SIZE:", len(ids))
print("TF-IDF query nonzero features:", query_vector.nnz)

semantic_ids = display_results(
    "TITAN V2 + FAISS",
    semantic_indices[0],
    semantic_scores[0],
)

lexical_ids = display_results(
    "TF-IDF + COSINE",
    lexical_indices,
    lexical_scores[lexical_indices],
)

overlap = set(semantic_ids) & set(lexical_ids)

print("\nCOMPARISON SUMMARY")
print("Top-5 overlap:", len(overlap), "papers")
print("Titan rank 1:", semantic_ids[0])
print("TF-IDF rank 1:", lexical_ids[0])
print("Identical rankings:", semantic_ids == lexical_ids)
print("AWS requests made: 0")
print("INDEPENDENT QUERY COMPARISON COMPLETED")
