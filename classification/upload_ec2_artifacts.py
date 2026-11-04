"""Upload and independently verify classification model artifacts in S3."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import boto3

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "ec2-s3-config.json"

REQUIRED_FILES = {
    "classifier.joblib",
    "training_metrics.json",
    "reproducibility.json",
    "validation_2024.json",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_completion_marker(run_dir: Path, run_id: str) -> dict:
    """Describe the verified files required for a complete model run."""

    run_dir = Path(run_dir).resolve()

    if not run_id or not all(
        character.isalnum() or character in "-_"
        for character in run_id
    ):
        raise ValueError("Invalid run identifier.")

    checksums = {}

    for filename in sorted(REQUIRED_FILES):
        path = run_dir / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"Required artifact missing: {path}"
            )

        checksums[filename] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    return {
        "schema_version": 1,
        "run_id": run_id,
        "status": "complete",
        "artifacts": checksums,
    }


def upload_artifacts(run_dir: Path, run_id: str) -> None:
    """Upload verified local artifacts without replacing existing S3 objects."""

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    if config["region"] != "us-east-1":
        raise ValueError("Unexpected AWS region.")

    if not run_id or not all(
        character.isalnum() or character in "-_"
        for character in run_id
    ):
        raise ValueError("Invalid run identifier.")

    run_dir = Path(run_dir).resolve()

    for filename in REQUIRED_FILES:
        path = run_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing required artifact: {path}")

    # Verify that all artifacts belong to the same model run.
    training_metrics = json.loads(
        (run_dir / "training_metrics.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (run_dir / "reproducibility.json").read_text(encoding="utf-8")
    )
    validation = json.loads(
        (run_dir / "validation_2024.json").read_text(encoding="utf-8")
    )

    model_hash = sha256_file(run_dir / "classifier.joblib")

    if training_metrics["model_sha256"] != model_hash:
        raise ValueError("Local model checksum mismatch.")

    if validation["model_sha256"] != model_hash:
        raise ValueError("Validation model checksum mismatch.")

    if (
        metadata["training_sample_sha256"]
        != validation["training_sample_sha256"]
    ):
        raise ValueError("Training sample fingerprint mismatch.")

    if (
        metadata["training_sample_size"]
        != validation["training_sample_size"]
        or metadata["training_sample_size"]
        != training_metrics["training_records"]
    ):
        raise ValueError("Training record count mismatch.")

    if validation["dataset"] != "2024_validation":
        raise ValueError("Unexpected evaluation dataset.")

    if validation["records"] != 77187:
        raise ValueError("Unexpected validation record count.")

    print("Local artifact consistency: VERIFIED", flush=True)

    client = boto3.client("s3", region_name=config["region"])

    bucket = config["bucket"]
    prefix = config["model_output_prefix"] + run_id + "/"

    # Refuse to publish into a previously used run prefix.
    existing = client.list_objects_v2(
        Bucket=bucket,
        Prefix=prefix,
        MaxKeys=1,
    )

    if existing.get("KeyCount", 0):
        raise FileExistsError(f"S3 run prefix already exists: {prefix}")

    for filename in sorted(REQUIRED_FILES):
        local_path = run_dir / filename
        key = prefix + filename
        expected_hash = sha256_file(local_path)

        print(f"Uploading: s3://{bucket}/{key}", flush=True)

        with local_path.open("rb") as stream:
            client.put_object(
                Bucket=bucket,
                Key=key,
                Body=stream,
                IfNoneMatch="*",
            )

        with tempfile.TemporaryDirectory() as temporary:
            verification_path = Path(temporary) / filename

            client.download_file(
                bucket,
                key,
                str(verification_path),
            )

            if sha256_file(verification_path) != expected_hash:
                raise ValueError(
                    f"S3 verification failed for {filename}"
                )

        print(f"Verified SHA-256: {filename}", flush=True)

    # Publish the completion marker only after every artifact
    # has been uploaded and independently verified.
    marker = build_completion_marker(run_dir, run_id)
    marker_bytes = (
        json.dumps(marker, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    marker_key = prefix + "COMPLETE.json"

    client.put_object(
        Bucket=bucket,
        Key=marker_key,
        Body=marker_bytes,
        ContentType="application/json",
        IfNoneMatch="*",
    )

    with tempfile.TemporaryDirectory() as temporary:
        verification_path = Path(temporary) / "COMPLETE.json"

        client.download_file(
            bucket,
            marker_key,
            str(verification_path),
        )

        if verification_path.read_bytes() != marker_bytes:
            raise ValueError("S3 completion marker verification failed.")

    print("Completion marker: UPLOADED AND VERIFIED", flush=True)
    print("\nALL S3 ARTIFACT UPLOADS VERIFIED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    upload_artifacts(args.run_dir, args.run_id)


if __name__ == "__main__":
    main()
