"""
Train the DS483 arXiv research category classifier.

Model:
    TF-IDF + Logistic Regression

Training data:
    classification/data/train.jsonl

Important:
    The vectorizer must be fitted only on training records.
    Validation and test records are never used during fitting.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from classification.diagnostics import (
    reproducibility_metadata,
    save_training_artifacts,
    train_with_diagnostics,
)

CLASSIFICATION_DIR = Path(__file__).resolve().parent
TRAIN_PATH = CLASSIFICATION_DIR / "data" / "train.jsonl"

RANDOM_SEED = 42


EXPECTED_CATEGORIES = frozenset({
    "cs.AI", "cs.CL", "cs.CR", "cs.CV",
    "cs.DB", "cs.LG", "cs.NI", "cs.RO",
})


def load_training_data(path: Path):
    """Load and validate labeled training abstracts."""

    if not path.is_file():
        raise FileNotFoundError(path)

    texts = []
    labels = []
    paper_ids = []
    seen_ids = set()
    seen_hashes = set()

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            paper = json.loads(line)

            paper_id = paper["id"]
            abstract = paper["abstract"]
            label = paper["label"]
            digest = paper["abstract_sha256"]
            submitted = paper["submitted"]

            if not isinstance(paper_id, str) or not paper_id.strip():
                raise ValueError(f"Invalid ID at line {line_number}")

            if not isinstance(abstract, str) or len(abstract.split()) < 30:
                raise ValueError(f"Invalid abstract at line {line_number}")

            if label not in EXPECTED_CATEGORIES:
                raise ValueError(f"Unexpected label at line {line_number}")

            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"Invalid hash at line {line_number}")

            normalized = " ".join(abstract.split()).lower()
            computed_hash = hashlib.sha256(
                normalized.encode("utf-8")
            ).hexdigest()

            if digest != computed_hash:
                raise ValueError(
                    f"Abstract hash mismatch at line {line_number}"
                )

            if not isinstance(submitted, str):
                raise ValueError(
                    f"Invalid submission date at line {line_number}"
                )

            try:
                created = datetime.fromisoformat(submitted)
            except ValueError as exc:
                raise ValueError(
                    f"Malformed submission date at line {line_number}"
                ) from exc

            if created.tzinfo is None:
                raise ValueError(
                    f"Timezone missing at line {line_number}"
                )

            created = created.astimezone(timezone.utc)

            if created.year >= 2024:
                raise ValueError(
                    f"Non-training date at line {line_number}"
                )

            if paper_id in seen_ids:
                raise ValueError(f"Duplicate ID: {paper_id}")

            if digest in seen_hashes:
                raise ValueError(f"Duplicate abstract hash: {digest}")

            seen_ids.add(paper_id)
            seen_hashes.add(digest)

            texts.append(abstract)
            labels.append(label)
            paper_ids.append(paper_id)

    if not texts:
        raise ValueError("Training dataset is empty.")

    return texts, labels, paper_ids


def select_balanced_pilot(texts, labels, paper_ids, max_examples):
    """Select a deterministic, balanced subset for a bounded pilot."""

    category_count = len(EXPECTED_CATEGORIES)

    if not (len(texts) == len(labels) == len(paper_ids)):
        raise ValueError(
            "Pilot input lengths do not match."
        )

    if len(set(paper_ids)) != len(paper_ids):
        raise ValueError(
            "Duplicate paper IDs in pilot input."
        )

    if max_examples < category_count:
        raise ValueError(
            "Pilot size must include at least one example per category."
        )

    if max_examples > 800:
        raise ValueError("Pilot cannot exceed 800 examples.")

    if max_examples % category_count != 0:
        raise ValueError(
            "Pilot size must be divisible by the number of categories."
        )

    per_category = max_examples // category_count
    counts = Counter()

    selected_texts = []
    selected_labels = []
    selected_ids = []

    for text, label, paper_id in zip(texts, labels, paper_ids):
        if counts[label] >= per_category:
            continue

        selected_texts.append(text)
        selected_labels.append(label)
        selected_ids.append(paper_id)
        counts[label] += 1

    if set(counts) != EXPECTED_CATEGORIES:
        raise ValueError("Pilot is missing required categories.")

    if any(count != per_category for count in counts.values()):
        raise ValueError("Insufficient examples for balanced pilot.")

    if len(selected_texts) != max_examples:
        raise ValueError("Pilot record count mismatch.")

    return selected_texts, selected_labels, selected_ids


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.95,
            max_features=100000,
            sublinear_tf=True,
            dtype=np.float32,
        )),
        ("classifier", LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=RANDOM_SEED,
        )),
    ])


def verify_production_training_inputs(texts, labels, paper_ids):
    """Verify production inputs without fitting a model."""

    expected_records = 40000
    expected_per_category = 5000
    expected_sha256 = (
        "d1aac2d0b524edda20cd8acf971a71d771c863523396d4b8c8db59ac6a985d29"
    )

    if not (
        len(texts) == len(labels) == len(paper_ids) == expected_records
    ):
        raise ValueError("Production training requires exactly 40,000 records.")

    counts = Counter(labels)

    if set(counts) != EXPECTED_CATEGORIES:
        raise ValueError("Production training categories do not match.")

    if any(count != expected_per_category for count in counts.values()):
        raise ValueError("Production training category balance is incorrect.")

    if len(set(paper_ids)) != expected_records:
        raise ValueError("Duplicate paper IDs in production training inputs.")

    actual_sha256 = hashlib.sha256()

    with TRAIN_PATH.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            actual_sha256.update(chunk)

    if actual_sha256.hexdigest() != expected_sha256:
        raise ValueError("Production training file checksum mismatch.")

    # Confirm the actual inputs match the verified source file
    # exactly, including record order and abstract contents.
    verified_texts, verified_labels, verified_ids = load_training_data(
        TRAIN_PATH
    )

    if (
        texts != verified_texts
        or labels != verified_labels
        or paper_ids != verified_ids
    ):
        raise ValueError(
            "Production training inputs differ from verified source data."
        )

    return {
        "records": expected_records,
        "categories": dict(sorted(counts.items())),
        "training_file_sha256": actual_sha256.hexdigest(),
    }


def verify_production_authorization(full_train, approval_token):
    """Validate explicit production-training authorization."""

    expected_token = "DS483_APPROVE_40000_TRAINING"

    if not full_train:
        raise PermissionError(
            "Production training requires --full-train."
        )

    if approval_token != expected_token:
        raise PermissionError(
            "Production training approval token missing or incorrect."
        )

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect training data without fitting a model.",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Request a bounded local training pilot.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Maximum training examples for pilot mode.",
    )
    parser.add_argument(
        "--full-train",
        action="store_true",
        help="Reserved for an explicitly authorized EC2 run.",
    )
    args = parser.parse_args()

    if args.full_train:
        raise RuntimeError(
            "FULL TRAINING DISABLED: EC2 authorization pending."
        )

    if args.pilot and not (1 <= args.max_examples <= 800):
        raise ValueError(
            "Pilot mode requires --max-examples between 1 and 800."
        )

    if args.pilot and args.dry_run:
        raise ValueError(
            "Choose either --pilot or --dry-run, not both."
        )

    if args.max_examples and not args.pilot:
        raise ValueError(
            "--max-examples requires --pilot."
        )

    print("DS483 Classification Training")
    print("-----------------------------")
    print("Training file:", TRAIN_PATH)

    texts, labels, paper_ids = load_training_data(TRAIN_PATH)

    print("Training records:", len(texts))
    print("Unique paper IDs:", len(set(paper_ids)))
    print("Categories:", dict(sorted(Counter(labels).items())))

    pipeline = build_pipeline()

    print("\nPipeline configuration:")
    print(pipeline)

    if args.dry_run:
        print("\nDRY RUN PASSED: No model was fitted or saved.")
        return

    if args.pilot:
        X, y, selected_ids = select_balanced_pilot(
            texts, labels, paper_ids, args.max_examples
        )

        print("\nCONTROLLED PILOT")
        print("Selected examples:", len(X))
        print("Selected categories:", dict(sorted(Counter(y).items())))

        output_dir = (
            CLASSIFICATION_DIR
            / "models"
            / f"pilot_{args.max_examples}_seed{RANDOM_SEED}"
        )

        if output_dir.exists():
            raise FileExistsError(
                f"Pilot output already exists: {output_dir}"
            )

        pipeline, diagnostics = train_with_diagnostics(
            pipeline, X, y
        )

        metadata = reproducibility_metadata(
            TRAIN_PATH,
            RANDOM_SEED,
            pipeline,
            selected_ids,
            y,
            X,
        )

        metadata["run_type"] = "bounded_local_pilot"
        metadata["training_records"] = len(X)

        paths = save_training_artifacts(
            pipeline,
            diagnostics,
            metadata,
            output_dir,
        )

        print("\nTRAINING DIAGNOSTICS")
        print(json.dumps(diagnostics, indent=2))
        print("\nSaved artifacts:")
        for name, location in paths.items():
            print(f"{name}: {location}")

        print("\nCONTROLLED PILOT COMPLETED")
        return

    raise RuntimeError(
        "Training requires explicit --pilot authorization. "
        "Full training remains disabled."
    )


if __name__ == "__main__":
    main()
