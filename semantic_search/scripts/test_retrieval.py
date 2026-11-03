import json
from pathlib import Path

import faiss
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path("semantic_search/data")

papers = [
    json.loads(line)
    for line in (BASE / "pilot_10.jsonl").read_text().splitlines()
]

paper_by_id = {paper["paper_id"]: paper for paper in papers}

mapping = json.loads(
    (BASE / "faiss/paper_mapping.json").read_text()
)

checkpoint = json.loads(
    (BASE / "embeddings/checkpoint.json").read_text()
)

tfidf = joblib.load(BASE / "tfidf/pilot_tfidf.joblib")
index = faiss.read_index(str(BASE / "faiss/pilot.index"))

ids = mapping["paper_ids"]

assert ids == tfidf["paper_ids"]
assert set(ids) == set(checkpoint["results"])
assert index.ntotal == len(ids) == 10
assert (
    mapping["source_sha256"]
    == checkpoint["source_sha256"]
    == tfidf["source_sha256"]
)

# Use an already embedded paper as the test query.
query_id = "1206.4677"
query_text = paper_by_id[query_id]["input_text"]

print("QUERY PAPER:", query_id)
print("TITLE:", paper_by_id[query_id]["title"])

# Semantic retrieval using the existing Titan vector.
query_embedding = np.asarray(
    [checkpoint["results"][query_id]["embedding"]],
    dtype=np.float32,
)

faiss.normalize_L2(query_embedding)

semantic_scores, semantic_positions = index.search(query_embedding, 5)

# Lexical retrieval using the same document text.
query_vector = tfidf["vectorizer"].transform([query_text])
lexical_scores = cosine_similarity(
    query_vector,
    tfidf["matrix"],
)[0]

lexical_positions = np.argsort(-lexical_scores, kind="stable")[:5]

def show_results(name, positions, scores):
    print(f"\n{name}")
    print("-" * 65)

    for rank, (position, score) in enumerate(zip(positions, scores), 1):
        paper_id = ids[int(position)]
        paper = paper_by_id[paper_id]

        print(f"{rank}. [{paper_id}] {paper['title']}")
        print(f"   Similarity: {float(score):.4f}")
        print(f"   https://arxiv.org/abs/{paper_id}")

show_results(
    "TITAN + FAISS",
    semantic_positions[0],
    semantic_scores[0],
)

show_results(
    "TF-IDF",
    lexical_positions,
    lexical_scores[lexical_positions],
)

assert ids[int(semantic_positions[0][0])] == query_id
assert ids[int(lexical_positions[0])] == query_id

print("\nUNIFIED RETRIEVAL TEST PASSED")
print("Both methods retrieved the query paper at rank 1.")
print("AWS requests made: 0")
