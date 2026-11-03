import csv
import hashlib
import json
import random
from pathlib import Path

import faiss
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path("semantic_search/data")
OUTPUT = BASE / "evaluation_50"
OUTPUT.mkdir(parents=True, exist_ok=True)

QUERY = (
    "Machine learning methods for improving "
    "classification robustness and prediction accuracy"
)

config = {
    "query": QUERY,
    "model_id": "amazon.titan-embed-text-v2:0",
    "region": "us-east-1",
    "dimensions": 512,
    "normalize": True,
}

key = hashlib.sha256(
    json.dumps(
        config,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
).hexdigest()

source = BASE / "pilot_50.jsonl"
source_sha = hashlib.sha256(source.read_bytes()).hexdigest()

papers = [
    json.loads(line)
    for line in source.read_text(encoding="utf-8").splitlines()
]
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
assert index.ntotal == len(ids) == 50

entry = cache["entries"][key]
for field in ("model_id", "region", "dimensions", "normalize"):
    assert entry[field] == config[field]

query_embedding = np.asarray(
    [entry["embedding"]],
    dtype=np.float32,
)
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

semantic_ids = [
    ids[int(pos)] for pos in semantic_positions[0]
]
lexical_ids = [
    ids[int(pos)] for pos in lexical_positions
]

# Pool results without exposing method or rank to the judge.
pooled_ids = sorted(set(semantic_ids) | set(lexical_ids))

rng = random.Random(483)
rng.shuffle(pooled_ids)

judgments_path = OUTPUT / "blinded_judgments.csv"
rankings_path = OUTPUT / "rankings_private.json"

with judgments_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "query_id",
        "query",
        "paper_id",
        "title",
        "abstract",
        "relevance",
    ])

    for paper_id in pooled_ids:
        paper = paper_by_id[paper_id]
        writer.writerow([
            "Q1",
            QUERY,
            paper_id,
            paper["title"],
            paper["abstract"],
            "",
        ])

rankings = {
    "query_id": "Q1",
    "query": QUERY,
    "source_sha256": source_sha,
    "semantic_top10": semantic_ids,
    "tfidf_top10": lexical_ids,
}

rankings_path.write_text(
    json.dumps(rankings, indent=2) + "\n",
    encoding="utf-8",
)

print("BLINDED EVALUATION FILE CREATED")
print("Query ID: Q1")
print("Semantic results:", len(semantic_ids))
print("TF-IDF results:", len(lexical_ids))
print("Unique pooled papers:", len(pooled_ids))
print("Blinded CSV:", judgments_path)
print("Private rankings:", rankings_path)
print("Relevance labels:", "0 = irrelevant, 1 = partial, 2 = direct")
print("AWS requests made: 0")
