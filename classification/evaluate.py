"""Evaluate a fitted arXiv classifier on the 2024 validation set."""

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from classification.train import EXPECTED_CATEGORIES

CLASSIFICATION_DIR = Path(__file__).resolve().parent
VALIDATION_PATH = CLASSIFICATION_DIR / "data" / "validation.jsonl"


def load_validation_data(path=VALIDATION_PATH):
    """Load 2024 validation records without modifying them."""

    if Path(path).resolve() != VALIDATION_PATH.resolve():
        raise ValueError("Only the designated 2024 validation file is allowed.")

    texts, labels, ids = [], [], []
    seen_ids = set()
    seen_hashes = set()

    with VALIDATION_PATH.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            paper = json.loads(line)

            paper_id = paper["id"]
            label = paper["label"]
            abstract = paper["abstract"]
            digest = paper["abstract_sha256"]
            submitted_text = paper["submitted"]

            if not isinstance(paper_id, str) or not paper_id.strip():
                raise ValueError(f"Invalid paper ID at line {line_number}")

            if not isinstance(label, str):
                raise ValueError(f"Invalid label type at line {line_number}")

            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"Invalid hash at line {line_number}")

            if not isinstance(submitted_text, str):
                raise ValueError(
                    f"Invalid submission date at line {line_number}"
                )

            try:
                submitted = datetime.fromisoformat(submitted_text)
            except ValueError as exc:
                raise ValueError(
                    f"Malformed submission date at line {line_number}"
                ) from exc

            if label not in EXPECTED_CATEGORIES:
                raise ValueError(f"Invalid category at line {line_number}")

            if submitted.tzinfo is None:
                raise ValueError(f"Missing timezone at line {line_number}")

            if submitted.astimezone(timezone.utc).year != 2024:
                raise ValueError(f"Invalid validation year at line {line_number}")

            if not isinstance(abstract, str) or len(abstract.split()) < 30:
                raise ValueError(f"Invalid abstract at line {line_number}")

            normalized = " ".join(abstract.split()).lower()
            computed_hash = hashlib.sha256(
                normalized.encode("utf-8")
            ).hexdigest()

            if digest != computed_hash:
                raise ValueError(
                    f"Abstract hash mismatch at line {line_number}"
                )

            if paper_id in seen_ids or digest in seen_hashes:
                raise ValueError(
                    f"Duplicate validation record at line {line_number}"
                )

            seen_ids.add(paper_id)
            seen_hashes.add(digest)

            texts.append(abstract)
            labels.append(label)
            ids.append(paper_id)

    if not texts:
        raise ValueError("Validation dataset is empty.")

    return texts, labels, ids


def evaluate_classifier(model, texts, labels):
    """Evaluate an already fitted model without fitting or tuning it."""

    predictions = model.predict(texts)
    categories = sorted(EXPECTED_CATEGORIES)

    return {
        "records": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(
            labels, predictions, labels=categories, average="macro"
        )),
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
        "category_order": categories,
        "class_distribution": dict(sorted(Counter(labels).items())),
    }


def save_evaluation_results(results, output_path):
    """Save evaluation JSON without exposing a partially written final file."""
    import os
    import tempfile

    output_path = Path(output_path)

    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing results: {output_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(results, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

        os.link(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return output_path


def main():
    """Evaluate a trusted saved classifier on the 2024 validation set."""
    import argparse
    import hashlib
    import joblib

    parser = argparse.ArgumentParser(
        description="Evaluate an existing classifier on 2024 validation data."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing classifier.joblib and training_metrics.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New JSON file for validation metrics",
    )
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    model_path = model_dir / "classifier.joblib"
    metrics_path = model_dir / "training_metrics.json"
    metadata_path = model_dir / "reproducibility.json"
    output_path = args.output.resolve()

    for path in (model_path, metrics_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required artifact missing: {path}")

    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing results: {output_path}"
        )

    training_metrics = json.loads(
        metrics_path.read_text(encoding="utf-8")
    )
    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    expected_hash = training_metrics["model_sha256"]
    digest = hashlib.sha256()

    with model_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    if digest.hexdigest() != expected_hash:
        raise ValueError("Model checksum verification failed.")

    print("Model checksum: VERIFIED")
    print("Loading trusted local model:", model_path)

    model = joblib.load(model_path)

    print("Loading 2024 validation dataset...")
    texts, labels, _ = load_validation_data()

    print("Evaluating", len(labels), "validation records...")
    results = evaluate_classifier(model, texts, labels)

    results["dataset"] = "2024_validation"
    results["model_sha256"] = expected_hash
    results["training_sample_sha256"] = metadata[
        "training_sample_sha256"
    ]
    results["training_sample_size"] = metadata[
        "training_sample_size"
    ]

    save_evaluation_results(results, output_path)

    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Macro-F1: {results['macro_f1']:.4f}")
    print("Results saved:", output_path)


if __name__ == "__main__":
    main()
