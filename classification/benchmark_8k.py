"""Local 8,000-paper classification resource benchmark."""

import json
import resource
import tempfile
import time
import warnings
from collections import Counter
from pathlib import Path

import joblib

from classification.train import build_pipeline, load_training_data

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "classification/data/train.jsonl"

SAMPLES_PER_CATEGORY = 1000
EXPECTED_CATEGORIES = 8


def main():
    texts, labels, ids = load_training_data(TRAIN_PATH)

    counts = Counter()
    selected = []

    for index, label in enumerate(labels):
        if counts[label] < SAMPLES_PER_CATEGORY:
            selected.append(index)
            counts[label] += 1

    assert len(selected) == 8000
    assert len(counts) == EXPECTED_CATEGORIES
    assert all(n == SAMPLES_PER_CATEGORY for n in counts.values())

    X = [texts[i] for i in selected]
    y = [labels[i] for i in selected]

    print("=== 8,000-Paper Benchmark ===", flush=True)
    print("Training records:", len(X), flush=True)
    print("Category counts:", dict(sorted(counts.items())), flush=True)

    pipeline = build_pipeline()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        start = time.perf_counter()
        pipeline.fit(X, y)
        elapsed = time.perf_counter() - start

    vocabulary = len(
        pipeline.named_steps["tfidf"].vocabulary_
    )

    convergence_warnings = [
        str(w.message)
        for w in caught
        if "converg" in str(w.message).lower()
    ]

    print("\n=== Fit Results ===", flush=True)
    print(f"Training runtime: {elapsed:.3f} seconds", flush=True)
    print("TF-IDF vocabulary:", vocabulary, flush=True)
    print(
        "Convergence warnings:",
        len(convergence_warnings),
        flush=True,
    )

    for warning in convergence_warnings:
        print("WARNING:", warning, flush=True)

    samples = [
        "Deep learning for computer vision and object detection",
        "SQL database indexing and query optimization",
        "Autonomous robot navigation and motion planning",
    ]

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "benchmark.joblib"

        joblib.dump(pipeline, path)
        model_size = path.stat().st_size
        restored = joblib.load(path)

        before = pipeline.predict(samples)
        after = restored.predict(samples)

        assert list(before) == list(after)
        assert len(restored.classes_) == EXPECTED_CATEGORIES

        print("\n=== Model Artifact ===", flush=True)
        print("Save/reload: PASSED", flush=True)
        print("Prediction consistency: PASSED", flush=True)
        print(
            f"Serialized size: {model_size / 1024**2:.2f} MiB",
            flush=True,
        )

    peak_rss = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss

    print("\n=== Resource Measurements ===", flush=True)
    print(
        f"Peak process RSS: {peak_rss / 1024**2:.2f} MiB",
        flush=True,
    )
    print(f"Training runtime: {elapsed:.3f} seconds", flush=True)

    print("\nBENCHMARK PASSED", flush=True)


if __name__ == "__main__":
    main()
