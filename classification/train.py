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
import json
from collections import Counter
from pathlib import Path

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

CLASSIFICATION_DIR = Path(__file__).resolve().parent
TRAIN_PATH = CLASSIFICATION_DIR / "data" / "train.jsonl"

RANDOM_SEED = 42


def load_training_data(path: Path):
    texts = []
    labels = []
    paper_ids = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            paper = json.loads(line)

            texts.append(paper["abstract"])
            labels.append(paper["label"])
            paper_ids.append(paper["id"])

    if not texts:
        raise ValueError("Training dataset is empty.")

    if len(paper_ids) != len(set(paper_ids)):
        raise ValueError("Duplicate paper IDs in training data.")

    return texts, labels, paper_ids


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect training data without fitting a model.",
    )
    args = parser.parse_args()

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

    raise RuntimeError(
        "Training is intentionally disabled until "
        "the dry run and resource checks pass."
    )


if __name__ == "__main__":
    main()
