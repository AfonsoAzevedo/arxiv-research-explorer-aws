"""
Rebuild the recorded 368-paper FAISS index from local saved artifacts.

The baseline index holds 312 papers, and the checkpoint contains
56 additional Titan V2 embeddings. No AWS requests are made.

By default, rebuild in memory and compare against the existing index.
Use --output-dir to save a separate copy without overwriting files.
"""

import argparse
import hashlib
import json
from pathlib import Path

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "semantic_search/data"

# Locate the saved baseline, expansion files, and final provenance
BASELINE = DATA / "faiss_312"
EXPANSION = DATA / "embeddings_25k"
FINAL = DATA / "faiss_368"

BASELINE_INDEX = BASELINE / "index.faiss"
BASELINE_IDS = BASELINE / "paper_ids.json"
EXPANSION_MANIFEST = EXPANSION / "cybersecurity_56_manifest.json"
EXPANSION_STATE = EXPANSION / "cybersecurity_56_state.json"
PROVENANCE = FINAL / "provenance.json"

DIMENSIONS = 512
BASELINE_SIZE = 312
EXPANSION_SIZE = 56
INDEX_SIZE = BASELINE_SIZE + EXPANSION_SIZE


def read_json(path):
    """Load a saved JSON artifact"""
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path):
    """Identify an artifact using its SHA-256 fingerprint"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources():
    """Check the original files against the recorded provenance"""

    provenance = read_json(PROVENANCE)

    # Confirm the intended index configuration and number of papers
    expected = {
        "index_type": "IndexFlatIP",
        "dimensions": DIMENSIONS,
        "normalized_embeddings": True,
        "total_papers": INDEX_SIZE,
        "baseline_papers": BASELINE_SIZE,
        "added_papers": EXPANSION_SIZE,
    }

    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"Unexpected provenance field: {key}")

    # Reject changed input files before rebuilding the index
    sources = {
        "baseline_index_sha256": BASELINE_INDEX,
        "baseline_mapping_sha256": BASELINE_IDS,
        "expansion_manifest_sha256": EXPANSION_MANIFEST,
        "expansion_checkpoint_sha256": EXPANSION_STATE,
    }

    for key, path in sources.items():
        if file_sha256(path) != provenance[key]:
            raise ValueError(f"Source fingerprint mismatch: {path}")

    return provenance


def rebuild_index():
    """Combine 312 saved vectors with the 56 recorded new embeddings"""

    provenance = verify_sources()

    original_ids = read_json(BASELINE_IDS)
    manifest = read_json(EXPANSION_MANIFEST)
    state = read_json(EXPANSION_STATE)

    # Check that the expansion checkpoint belongs to this exact manifest
    if state.get("manifest_sha256") != provenance["expansion_manifest_sha256"]:
        raise ValueError("Expansion checkpoint and manifest do not match")

    if manifest.get("model_id") != "amazon.titan-embed-text-v2:0":
        raise ValueError("Unexpected embedding model")

    if (
        manifest.get("region") != "us-east-1"
        or manifest.get("dimensions") != DIMENSIONS
        or manifest.get("normalize") is not True
    ):
        raise ValueError("Unexpected embedding configuration")

    additional_ids = [paper["paper_id"] for paper in manifest["papers"]]
    results = state["results"]

    # Verify paper counts, uniqueness, and completed embedding requests
    if len(original_ids) != BASELINE_SIZE or len(additional_ids) != EXPANSION_SIZE:
        raise ValueError("Unexpected number of papers")

    paper_ids = original_ids + additional_ids

    if len(set(paper_ids)) != INDEX_SIZE:
        raise ValueError("Paper IDs must be unique")

    if set(results) != set(additional_ids):
        raise ValueError("Missing or unexpected expansion embeddings")

    if any(attempt["status"] != "completed" for attempt in state["attempts"]):
        raise ValueError("Unresolved expansion requests")

    # Load the original normalized inner-product index without changing it
    index = faiss.read_index(str(BASELINE_INDEX))

    if (
        index.d != DIMENSIONS
        or index.ntotal != BASELINE_SIZE
        or index.metric_type != faiss.METRIC_INNER_PRODUCT
    ):
        raise ValueError("Unexpected baseline FAISS index")

    # Collect the saved expansion vectors in the exact manifest order
    vectors = np.asarray(
        [results[paper_id]["embedding"] for paper_id in additional_ids],
        dtype=np.float32,
    )

    if vectors.shape != (EXPANSION_SIZE, DIMENSIONS):
        raise ValueError("Unexpected expansion vector dimensions")

    if not np.isfinite(vectors).all():
        raise ValueError("Expansion contains invalid vector values")

    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=0.01):
        raise ValueError("Expansion vectors must be normalized")

    # Normalize before inner-product search so scores equal cosine similarity
    faiss.normalize_L2(vectors)
    index.add(vectors)

    if index.ntotal != INDEX_SIZE:
        raise ValueError("Reconstructed index has the wrong paper count")

    # Check that all indexed vectors retrieve themselves as the closest match
    all_vectors = np.stack(
        [index.reconstruct(position) for position in range(INDEX_SIZE)]
    )

    if not np.allclose(np.linalg.norm(all_vectors, axis=1), 1.0, atol=0.01):
        raise ValueError("Reconstructed vectors are not normalized")

    scores, positions = index.search(np.ascontiguousarray(all_vectors), 1)

    if not np.array_equal(positions[:, 0], np.arange(INDEX_SIZE)):
        raise ValueError("Self-retrieval did not return all 368 papers")

    if not np.allclose(scores[:, 0], 1.0, atol=1e-4):
        raise ValueError("Unexpected self-retrieval similarity scores")

    return index, paper_ids, provenance, all_vectors


def compare_saved_index(index, paper_ids, reconstructed):
    """Compare the rebuilt vectors with the working local 368-paper index"""

    saved_index_path = FINAL / "index.faiss"
    saved_ids_path = FINAL / "paper_ids.json"

    if not saved_index_path.is_file() or not saved_ids_path.is_file():
        print("Saved final index not present; comparison skipped")
        return

    if read_json(saved_ids_path) != paper_ids:
        raise ValueError("Reconstructed paper-ID order differs from saved index")

    saved = faiss.read_index(str(saved_index_path))
    if (
        saved.d != DIMENSIONS
        or saved.ntotal != INDEX_SIZE
        or saved.metric_type != faiss.METRIC_INNER_PRODUCT
    ):
        raise ValueError("Unexpected saved final FAISS index")

    saved_vectors = np.stack(
        [saved.reconstruct(position) for position in range(INDEX_SIZE)]
    )

    if not np.allclose(reconstructed, saved_vectors, rtol=1e-6, atol=1e-6):
        raise ValueError("Reconstructed vectors differ from saved FAISS index")

    print("Saved 368-paper index comparison: PASSED")


def save_copy(output_dir, index, paper_ids, provenance):
    """Write a separate reconstruction without replacing existing files"""

    # Refuse an existing directory to protect the original index
    output_dir.mkdir(parents=True, exist_ok=False)

    index_path = output_dir / "index.faiss"
    ids_path = output_dir / "paper_ids.json"
    provenance_path = output_dir / "provenance.json"

    faiss.write_index(index, str(index_path))
    ids_path.write_text(json.dumps(paper_ids, indent=2) + "\n", encoding="utf-8")
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )

    # Load the written copy once more before reporting success
    loaded = faiss.read_index(str(index_path))
    if loaded.ntotal != INDEX_SIZE or read_json(ids_path) != paper_ids:
        raise ValueError("Saved reconstruction did not validate")

    print("Rebuilt artifacts saved to:", output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional new directory for a separate copy; existing paths are refused",
    )
    args = parser.parse_args()

    index, paper_ids, provenance, vectors = rebuild_index()
    print("Source SHA-256 fingerprints: VERIFIED")
    print("Reconstructed papers:", index.ntotal)
    print("Dimensions:", index.d)
    print("Self-retrieval:", f"{INDEX_SIZE}/{INDEX_SIZE}")

    compare_saved_index(index, paper_ids, vectors)

    if args.output_dir is not None:
        save_copy(args.output_dir, index, paper_ids, provenance)
    else:
        print("Read-only check complete. No files written.")

    print("AWS requests made: 0")


if __name__ == "__main__":
    main()
