import hashlib
import json
from pathlib import Path

import faiss
import numpy as np

BASE = Path("semantic_search/data")
SOURCE = BASE / "pilot_50.jsonl"
MANIFEST = BASE / "pilot_50_embedding_manifest.json"
STATE = BASE / "embeddings_50/state.json"

OUTPUT = BASE / "faiss_50"
INDEX_PATH = OUTPUT / "pilot_50.index"
MAPPING_PATH = OUTPUT / "paper_mapping_50.json"

raw = SOURCE.read_bytes()
source_sha = hashlib.sha256(raw).hexdigest()

manifest = json.loads(MANIFEST.read_text())
state = json.loads(STATE.read_text())

papers = [
    json.loads(line)
    for line in raw.splitlines()
]

paper_ids = [p["paper_id"] for p in papers]
results = state["results"]

# Verify dataset and embedding provenance.
assert source_sha == manifest["source_sha256"]
assert source_sha == state["source_sha256"]
assert paper_ids == manifest["paper_ids"]

assert len(paper_ids) == 50
assert len(set(paper_ids)) == 50
assert set(results) == set(paper_ids)

assert state["model_id"] == "amazon.titan-embed-text-v2:0"
assert state["region"] == "us-east-1"
assert state["dimensions"] == 512
assert state["normalize"] is True

# Preserve exactly the original dataset order.
vectors = np.asarray(
    [results[paper_id]["embedding"] for paper_id in paper_ids],
    dtype=np.float32,
)

assert vectors.shape == (50, 512)
assert np.all(np.isfinite(vectors))

original_norms = np.linalg.norm(vectors, axis=1)
assert np.allclose(original_norms, 1.0, atol=0.01)

faiss.normalize_L2(vectors)

# Exact cosine-equivalent nearest-neighbor search.
index = faiss.IndexFlatIP(512)
index.add(vectors)

assert index.ntotal == 50

mapping = {
    "source_sha256": source_sha,
    "model_id": state["model_id"],
    "region": state["region"],
    "dimensions": 512,
    "normalize": True,
    "index_type": "IndexFlatIP",
    "paper_ids": paper_ids,
}

OUTPUT.mkdir(parents=True, exist_ok=True)

faiss.write_index(index, str(INDEX_PATH))

MAPPING_PATH.write_text(
    json.dumps(mapping, indent=2) + "\n",
    encoding="utf-8",
)

# Independently reload artifacts.
loaded_index = faiss.read_index(str(INDEX_PATH))

loaded_mapping = json.loads(
    MAPPING_PATH.read_text(encoding="utf-8")
)

assert loaded_index.ntotal == 50
assert loaded_index.d == 512
assert loaded_mapping["paper_ids"] == paper_ids
assert loaded_mapping["source_sha256"] == source_sha

# Verify exact position-to-vector mapping.
for position in range(50):
    reconstructed = loaded_index.reconstruct(position)
    assert np.allclose(
        reconstructed,
        vectors[position],
        atol=1e-6,
    )

# Verify each paper retrieves itself at rank 1.
scores, indices = loaded_index.search(vectors, 1)

assert np.array_equal(
    indices[:, 0],
    np.arange(50),
)

assert np.allclose(
    scores[:, 0],
    1.0,
    atol=1e-4,
)

print("50-PAPER FAISS VALIDATION PASSED")
print("Index type:", loaded_mapping["index_type"])
print("Dimensions:", loaded_index.d)
print("Indexed papers:", loaded_index.ntotal)
print("Unique paper IDs:", len(set(paper_ids)))
print("Mapping entries:", len(loaded_mapping["paper_ids"]))
print("Self-retrieval:", "50/50")
print("Index file:", INDEX_PATH)
print("Mapping file:", MAPPING_PATH)
print("Index bytes:", INDEX_PATH.stat().st_size)
print("Source SHA-256:", source_sha)
print("AWS requests made: 0")
