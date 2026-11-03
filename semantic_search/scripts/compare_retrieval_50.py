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

source = BASE / "pilot_50.jsonl"
source_sha = hashlib.sha256(source.read_bytes()).hexdigest()

papers = [json.loads(line) for line in source.read_text().splitlines()]
paper_by_id = {p["paper_id"]: p for p in papers}

mapping = json.loads(
    (BASE / "faiss_50/paper_mapping_50.json").read_text()
)

tfidf = joblib.load(
    BASE / "tfidf_50/pilot_50_tfidf.joblib"
)

cache = json.loads(
    (BASE / "queries/query_cache.json").read_text()
)

index = faiss.read_index(
    str(BASE / "faiss_50/pilot_50.index")
)

ids = mapping["paper_ids"]

assert source_sha == mapping["source_sha256"]
assert source_sha == tfidf["source_sha256"]
assert ids == tfidf["paper_ids"]
assert ids == [p["paper_id"] for p in papers]
assert index.ntotal == len(ids) == 50
assert index.d == 512

entry = cache["entries"][key]

for field in ("model_id", "region", "dimensions", "normalize"):
    assert entry[field] == CONFIG[field]

query_embedding = np.asarray(
    [entry["embedding"]],
    dtype=np.float32,
)

assert query_embedding.shape == (1, 512)
assert np.all(np.isfinite(query_embedding))

faiss.normalize_L2(query_embedding)

semantic_scores, semantic_positions = index.search(
    query_embedding, 10
)

query_vector = tfidf["vectorizer"].transform([QUERY])
lexical_scores = cosine_similarity(
    query_vector,
    tfidf["matrix"],
)[0]

lexical_positions = np.argsort(
    -lexical_scores, kind="stable"
)[:10]


def show_results(name, positions, scores):
    print(f"\n{name}")
    print("=" * 70)

    ranked_ids = []

    for rank, (position, score) in enumerate(zip(positions, scores), 1):
        paper_id = ids[int(position)]
        paper = paper_by_id[paper_id]
        ranked_ids.append(paper_id)

        print(f"\n{rank}. {paper['title']}")
        print(f"   ID: {paper_id}")
        print(f"   Group: {paper['sampling_group']}")
        print(f"   Similarity: {float(score):.4f}")
        print(f"   https://arxiv.org/abs/{paper_id}")

    return ranked_ids


print("QUERY:", QUERY)
print("CORPUS:", len(ids), "real arXiv papers")
print("TF-IDF matched features:", query_vector.nnz)

semantic_ids = show_results(
    "TITAN V2 + FAISS",
    semantic_positions[0],
    semantic_scores[0],
)

lexical_ids = show_results(
    "TF-IDF + COSINE",
    lexical_positions,
    lexical_scores[lexical_positions],
)

print("\nCOMPARISON SUMMARY")
print("Top-5 overlap:", len(set(semantic_ids[:5]) & set(lexical_ids[:5])))
print("Top-10 overlap:", len(set(semantic_ids) & set(lexical_ids)))
print("Titan rank 1:", semantic_ids[0])
print("TF-IDF rank 1:", lexical_ids[0])
print("Identical rankings:", semantic_ids == lexical_ids)
print("AWS requests made: 0")
print("50-PAPER INDEPENDENT QUERY COMPARISON COMPLETED")
