import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path("semantic_search/data")
SOURCE = BASE / "pilot_50.jsonl"
FAISS_MAPPING = BASE / "faiss_50/paper_mapping_50.json"

OUTPUT_DIR = BASE / "tfidf_50"
OUTPUT_PATH = OUTPUT_DIR / "pilot_50_tfidf.joblib"

raw = SOURCE.read_bytes()
source_sha = hashlib.sha256(raw).hexdigest()

papers = [json.loads(line) for line in raw.splitlines()]
mapping = json.loads(
    FAISS_MAPPING.read_text(encoding="utf-8")
)

paper_ids = [paper["paper_id"] for paper in papers]
texts = [paper["input_text"] for paper in papers]

assert len(paper_ids) == len(set(paper_ids)) == 50
assert source_sha == mapping["source_sha256"]
assert paper_ids == mapping["paper_ids"]

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    norm="l2",
    dtype=np.float32,
)

matrix = vectorizer.fit_transform(texts)

assert matrix.shape[0] == 50
assert matrix.shape[1] > 0
assert np.all(np.isfinite(matrix.data))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

artifact = {
    "vectorizer": vectorizer,
    "matrix": matrix,
    "paper_ids": paper_ids,
    "source_sha256": source_sha,
    "configuration": {
        "lowercase": True,
        "stop_words": "english",
        "ngram_range": [1, 2],
        "norm": "l2",
    },
}

joblib.dump(artifact, OUTPUT_PATH)

# Independently reload the saved artifact.
loaded = joblib.load(OUTPUT_PATH)

assert loaded["paper_ids"] == mapping["paper_ids"]
assert loaded["source_sha256"] == source_sha
assert loaded["matrix"].shape == matrix.shape

# Verify self-retrieval for all 50 documents.
query_vectors = loaded["vectorizer"].transform(texts)

similarities = cosine_similarity(
    query_vectors,
    loaded["matrix"],
)

assert np.array_equal(
    np.argmax(similarities, axis=1),
    np.arange(50),
)

print("50-PAPER TF-IDF VALIDATION PASSED")
print("Documents:", matrix.shape[0])
print("Vocabulary features:", matrix.shape[1])
print("Nonzero matrix values:", matrix.nnz)
print("Matrix shape:", matrix.shape)
print("Paper-ID mapping:", "Matches FAISS exactly")
print("Self-retrieval:", "50/50")
print("Artifact:", OUTPUT_PATH)
print("Artifact bytes:", OUTPUT_PATH.stat().st_size)
print("Source SHA-256:", source_sha)
print("AWS requests made: 0")
