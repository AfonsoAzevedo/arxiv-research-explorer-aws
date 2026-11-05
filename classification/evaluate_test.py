"""Run the single held-out classification evaluation with a frozen model."""

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "full_40000_seed42"
TEST_PATH = ROOT / "data" / "test.jsonl"
MANIFEST_PATH = ROOT / "data" / "manifest.json"
FREEZE_PATH = ROOT / "results" / "model_freeze.json"
OUTPUT_PATH = ROOT / "results" / "final_test_2025_2026.json"
ATTEMPT_PATH = ROOT / "results" / "final_test_2025_2026.attempt.json"

EXPECTED_MODEL_SHA256 = (
    "60f84cbc1d99a9cff44bf089cec25e0ba4343575cab51ca3a3218777f83533e1"
)
EXPECTED_TEST_SHA256 = (
    "409f4cc11bcc9c46eb55eb20ad116173133aa9c8e5e4038c7591a9c52ec7f3c3"
)


def sha256_file(path):
    """Calculate a file checksum without modifying its contents."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_model():
    """Require the approved model and unchanged training configuration."""
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    metadata = json.loads(
        (MODEL_DIR / "reproducibility.json").read_text(encoding="utf-8")
    )
    completion = json.loads(
        (MODEL_DIR / "COMPLETE.json").read_text(encoding="utf-8")
    )

    assert freeze["status"] == "frozen_before_final_test"
    assert freeze["model_sha256"] == EXPECTED_MODEL_SHA256
    assert freeze["pipeline_parameters"] == metadata["pipeline_parameters"]
    assert freeze["training_sample_sha256"] == metadata["training_sample_sha256"]
    assert freeze["final_test_expected_records"] == 207257
    assert metadata["training_records"] == 40000
    assert metadata["random_seed"] == 42
    assert completion["status"] == "complete"
    assert completion["artifacts"]["classifier.joblib"]["sha256"] == (
        EXPECTED_MODEL_SHA256
    )

    model_path = MODEL_DIR / "classifier.joblib"
    assert sha256_file(model_path) == EXPECTED_MODEL_SHA256

    return freeze, model_path


def load_verified_test():
    """Validate the held-out test set without fitting a model."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert sha256_file(TEST_PATH) == EXPECTED_TEST_SHA256

    categories = sorted(manifest["categories"])
    texts, labels = [], []
    counts = Counter()
    seen_ids = set()
    seen_hashes = set()

    with TEST_PATH.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            record = json.loads(line)
            paper_id = record["id"]
            label = record["label"]
            abstract = record["abstract"]
            stored_hash = record["abstract_sha256"]
            submitted = datetime.fromisoformat(record["submitted"])

            assert isinstance(paper_id, str) and paper_id
            assert label in categories
            assert record["categories"][0] == label
            assert isinstance(abstract, str) and len(abstract.split()) >= 30
            assert submitted.tzinfo is not None
            assert 2025 <= submitted.astimezone(timezone.utc).year <= 2026

            # Recalculate abstract hashes rather than trusting stored values.
            normalized = " ".join(abstract.split()).lower()
            actual_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            assert stored_hash == actual_hash, (
                f"Abstract mismatch at line {line_number}"
            )
            assert paper_id not in seen_ids
            assert stored_hash not in seen_hashes

            seen_ids.add(paper_id)
            seen_hashes.add(stored_hash)
            texts.append(abstract)
            labels.append(label)
            counts[label] += 1

    assert len(texts) == 207257
    assert len(texts) == manifest["split_counts"]["test"]
    assert dict(counts) == manifest["category_counts"]["test"]
    return texts, labels, categories, dict(sorted(counts.items()))


def evaluate_once():
    """Predict once and save all final metrics without tuning."""
    freeze, model_path = verify_frozen_model()
    texts, labels, categories, distribution = load_verified_test()

    if OUTPUT_PATH.exists() or ATTEMPT_PATH.exists():
        raise FileExistsError("Final test evaluation already attempted or saved.")

    # Reserve the evaluation before loading the model or making predictions.
    attempt = {
        "status": "started",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_sha256": EXPECTED_MODEL_SHA256,
        "test_sha256": EXPECTED_TEST_SHA256,
    }
    with ATTEMPT_PATH.open("x", encoding="utf-8") as stream:
        json.dump(attempt, stream, indent=2)
        stream.write("\n")

    print("Frozen model and held-out dataset verified.", flush=True)
    print("Single final evaluation authorized.", flush=True)

    # Load only the locally verified artifact produced by our own training.
    model = joblib.load(model_path)
    predictions = model.predict(texts)

    majority_count = max(distribution.values())
    results = {
        "evaluation": "single_final_held_out_test",
        "status": "complete",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_sha256": freeze["model_sha256"],
        "training_sample_sha256": freeze["training_sample_sha256"],
        "test_sha256": EXPECTED_TEST_SHA256,
        "test_period": "2025-2026",
        "records": len(labels),
        "category_order": categories,
        "class_distribution": distribution,
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, labels=categories, average="macro")
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "majority_class": max(distribution, key=distribution.get),
        "majority_class_baseline": majority_count / len(labels),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=categories,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=categories
        ).tolist(),
        "limitations": [
            "Labels represent the first listed arXiv category.",
            "Exact duplicates were excluded; near duplicates were not audited.",
            "Training was class-balanced; test frequencies are naturally imbalanced.",
            "Chronological distribution shift may affect performance.",
            "No hyperparameters or model weights were changed after validation.",
        ],
    }

    # Create the final evidence file exclusively to prevent overwrites.
    with OUTPUT_PATH.open("x", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())

    print(f"Accuracy: {results['accuracy']:.6f}")
    print(f"Macro-F1: {results['macro_f1']:.6f}")
    print(f"Balanced accuracy: {results['balanced_accuracy']:.6f}")
    print(f"Majority-class baseline: {results['majority_class_baseline']:.6f}")
    print(f"Results saved: {OUTPUT_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen classifier on the held-out test set."
    )
    parser.add_argument(
        "--run-final-test",
        action="store_true",
        help="Explicit authorization for the single final evaluation.",
    )
    args = parser.parse_args()

    if not args.run_final_test:
        parser.error("Final evaluation requires --run-final-test.")

    evaluate_once()


if __name__ == "__main__":
    main()
