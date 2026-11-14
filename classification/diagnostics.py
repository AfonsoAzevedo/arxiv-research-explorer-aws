"""Training diagnostics and reproducibility utilities."""

import hashlib
import json
import platform
import resource
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning


def sha256_file(path: Path) -> str:
    """Compute a SHA-256 checksum without loading the entire file."""
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def train_with_diagnostics(pipeline, texts, labels):
    """Fit an explicitly supplied pipeline and measure its execution."""
    # Measure how long model fitting takes
    start = time.perf_counter()

    # Record warnings so I can check whether training converged
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.fit(texts, labels)

    elapsed = time.perf_counter() - start
    estimator = pipeline.named_steps["classifier"]
    vectorizer = pipeline.named_steps["tfidf"]

    convergence = [
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]

    # Collect training time, memory usage, features, and warnings
    diagnostics = {
        "training_records": len(texts),
        "training_seconds": elapsed,
        "peak_process_rss_mib": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024 ** 2 if sys.platform == "darwin" else 1024)
        ),
        "tfidf_features": len(vectorizer.vocabulary_),
        "optimizer_iterations": estimator.n_iter_.tolist(),
        "max_iterations": estimator.max_iter,
        "convergence_warnings": convergence,
        "other_warnings": [
            str(item.message)
            for item in caught
            if not issubclass(item.category, ConvergenceWarning)
        ],
    }

    return pipeline, diagnostics


def reproducibility_metadata(
    training_path: Path,
    seed: int,
    pipeline,
    paper_ids,
    labels,
    texts,
) -> dict:
    """Collect input checksum, versions, and model configuration."""
    # Save the environment and exact training configuration for reproducibility
    return {
        "training_file_sha256": sha256_file(training_path),
        "training_sample_sha256": training_sample_fingerprint(
            paper_ids, labels, texts
        ),
        "training_sample_size": len(texts),
        "random_seed": seed,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": numpy.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "pipeline_parameters": {
            key: repr(value)
            for key, value in pipeline.get_params().items()
            if "__" in key
        },
    }


def save_training_artifacts(
    pipeline,
    diagnostics: dict,
    metadata: dict,
    output_dir: Path,
) -> dict:
    """Save complete training artifacts without overwriting an existing run."""
    import os
    import shutil
    import tempfile

    output_dir = Path(output_dir)
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)

    if os.path.lexists(output_dir):
        raise FileExistsError(
            f"Artifact directory already exists: {output_dir}"
        )

    # Build model artifacts in a temporary directory first
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            dir=parent,
        )
    )

    published = False

    try:
        model_path = temporary_dir / "classifier.joblib"
        metrics_path = temporary_dir / "training_metrics.json"
        metadata_path = temporary_dir / "reproducibility.json"

        # Save the fitted TF-IDF and Logistic Regression pipeline
        joblib.dump(pipeline, model_path)

        # Include the model checksum with the measured training results
        saved_metrics = {
            **diagnostics,
            "serialized_model_mib": (
                model_path.stat().st_size / (1024 ** 2)
            ),
            "model_sha256": sha256_file(model_path),
        }

        metrics_path.write_text(
            json.dumps(saved_metrics, indent=2) + "\n",
            encoding="utf-8",
        )

        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )

        # Make sure every required artifact was created
        for artifact in (model_path, metrics_path, metadata_path):
            if not artifact.is_file() or artifact.stat().st_size == 0:
                raise RuntimeError(f"Incomplete artifact: {artifact}")

        # Publish the completed run atomically without replacing
        # an existing directory or symbolic link.
        # Publish the completed run without replacing an existing one
        os.symlink(
            temporary_dir.name,
            output_dir,
            target_is_directory=True,
        )
        published = True

    finally:
        if not published and temporary_dir.exists():
            shutil.rmtree(temporary_dir)

    return {
        "model": str(output_dir / "classifier.joblib"),
        "metrics": str(output_dir / "training_metrics.json"),
        "metadata": str(output_dir / "reproducibility.json"),
    }


def training_sample_fingerprint(
    paper_ids,
    labels,
    texts,
) -> str:
    """SHA-256 fingerprint of an ordered training sample."""

    if not (
        len(paper_ids) == len(labels) == len(texts)
    ):
        raise ValueError("Training sample lengths do not match.")

    if not paper_ids:
        raise ValueError("Training sample is empty.")

    if len(set(paper_ids)) != len(paper_ids):
        raise ValueError("Duplicate paper IDs in training sample.")

    digest = hashlib.sha256()

    # Fingerprint each paper ID, category, and abstract in order
    for paper_id, label, abstract in zip(
        paper_ids, labels, texts
    ):
        record = {
            "id": paper_id,
            "label": label,
            "abstract_sha256": hashlib.sha256(
                abstract.encode("utf-8")
            ).hexdigest(),
        }

        encoded = json.dumps(
            record,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)

    return digest.hexdigest()
