import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path("semantic_search/data")
SOURCE = BASE / "pilot_10.jsonl"
FAISS_MAPPING = BASE / "faiss/paper_mapping.json"
OUTPUT_DIR = BASE / "tfidf"

papers = [
    json.loads(line)
    for line in SOURCE.read_text(encoding="utf-8").splitlines()
]

mapping = json.loads(
    FAISS_MAPPING.read_text(encoding="utf-8")
)

paper_ids = [paper["paper_id"] for paper in papers]
texts = [paper["input_text"] for paper in papers]

source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()

assert source_hash == mapping["source_sha256"]
assert paper_ids == mapping["paper_ids"]
assert len(papers) == 10

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    norm="l2",
    dtype=np.float32,
)

matrix = vectorizer.fit_transform(texts)

assert matrix.shape[0] == 10
assert matrix.shape[1] > 0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "paper_ids": paper_ids,
        "source_sha256": source_hash,
    },
    OUTPUT_DIR / "pilot_tfidf.joblib",
)

# Reload and verify.
artifact = joblib.load(OUTPUT_DIR / "pilot_tfidf.joblib")

assert artifact["paper_ids"] == mapping["paper_ids"]
assert artifact["source_sha256"] == source_hash
assert artifact["matrix"].shape == matrix.shape

# Each document should retrieve itself as the nearest match.
queries = artifact["vectorizer"].transform(texts)
similarities = cosine_similarity(queries, artifact["matrix"])

assert np.array_equal(
    np.argmax(similarities, axis=1),
    np.arange(10),
)

print("TF-IDF BASELINE VALIDATION PASSED")
print("Documents:", matrix.shape[0])
print("Vocabulary features:", matrix.shape[1])
print("Nonzero values:", matrix.nnz)
print("Paper-ID mapping:", "Matches FAISS exactly")
print("Self-retrieval:", "10/10")
print("Artifact:", OUTPUT_DIR / "pilot_tfidf.joblib")
print("AWS requests made: 0")
