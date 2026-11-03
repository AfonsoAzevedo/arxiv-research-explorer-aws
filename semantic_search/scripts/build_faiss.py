import json
from pathlib import Path

import faiss
import numpy as np

BASE = Path("semantic_search/data")
PAPERS_PATH = BASE / "pilot_10.jsonl"
CHECKPOINT_PATH = BASE / "embeddings/checkpoint.json"
OUTPUT_DIR = BASE / "faiss"

INDEX_PATH = OUTPUT_DIR / "pilot.index"
MAPPING_PATH = OUTPUT_DIR / "paper_mapping.json"

papers = [
    json.loads(line)
    for line in PAPERS_PATH.read_text(encoding="utf-8").splitlines()
]

checkpoint = json.loads(
    CHECKPOINT_PATH.read_text(encoding="utf-8")
)

results = checkpoint["results"]

paper_ids = [paper["paper_id"] for paper in papers]

assert len(paper_ids) == len(set(paper_ids)) == 10
assert set(paper_ids) == set(results)

vectors = np.asarray(
    [results[paper_id]["embedding"] for paper_id in paper_ids],
    dtype=np.float32,
)

assert vectors.shape == (10, 512)
assert np.all(np.isfinite(vectors))

norms = np.linalg.norm(vectors, axis=1)
assert np.allclose(norms, 1.0, atol=0.01)

# Normalize float32 vectors for cosine-equivalent search.
faiss.normalize_L2(vectors)

index = faiss.IndexFlatIP(512)
index.add(vectors)

assert index.ntotal == 10

mapping = {
    "source_sha256": checkpoint["source_sha256"],
    "model_id": checkpoint["model_id"],
    "dimensions": 512,
    "index_type": "IndexFlatIP",
    "paper_ids": paper_ids,
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

faiss.write_index(index, str(INDEX_PATH))
MAPPING_PATH.write_text(
    json.dumps(mapping, indent=2) + "\n",
    encoding="utf-8",
)

# Independently reload saved artifacts.
loaded_index = faiss.read_index(str(INDEX_PATH))
loaded_mapping = json.loads(MAPPING_PATH.read_text())

assert loaded_index.ntotal == len(loaded_mapping["paper_ids"]) == 10
assert loaded_index.d == 512
assert loaded_mapping["paper_ids"] == paper_ids

for position, paper_id in enumerate(paper_ids):
    reconstructed = loaded_index.reconstruct(position)
    assert np.allclose(reconstructed, vectors[position], atol=1e-6)

# Self-search should return each paper as its closest match.
scores, indices = loaded_index.search(vectors, 1)

assert np.array_equal(indices[:, 0], np.arange(10))
assert np.allclose(scores[:, 0], 1.0, atol=1e-4)

print("FAISS INDEX VALIDATION PASSED")
print("Index type:", mapping["index_type"])
print("Dimensions:", loaded_index.d)
print("Indexed papers:", loaded_index.ntotal)
print("Mapping entries:", len(loaded_mapping["paper_ids"]))
print("Self-retrieval accuracy:", "10/10")
print("Index file:", INDEX_PATH)
print("Mapping file:", MAPPING_PATH)
print("AWS requests made: 0")
