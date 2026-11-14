"""Download and verify the two authorized classification datasets from S3."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import boto3

from classification.verify_ec2_data import verify_datasets

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "ec2-s3-config.json"
MANIFEST_PATH = BASE_DIR / "ec2-data-manifest.json"

EXPECTED_FILES = {"train.jsonl", "validation.jsonl"}


def download_datasets(data_dir: Path) -> None:
    # Load the approved S3 location and dataset checksums
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    if config.get("schema_version") != 1:
        raise ValueError("Unsupported S3 configuration version.")

    if config.get("region") != "us-east-1":
        raise ValueError("Unexpected AWS region.")

    objects = config["input_objects"]

    # Allow only the training and validation datasets
    if set(objects) != EXPECTED_FILES:
        raise ValueError("Unexpected S3 input objects.")

    if set(manifest["datasets"]) != EXPECTED_FILES:
        raise ValueError("Unexpected dataset manifest contents.")

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Refuse to overwrite files that already exist locally
    for filename in EXPECTED_FILES:
        if os.path.lexists(data_dir / filename):
            raise FileExistsError(
                f"Refusing to overwrite: {data_dir / filename}"
            )

    # Connect to S3 in the project's AWS region
    client = boto3.client("s3", region_name=config["region"])
    published_files = []

    try:
        for filename in sorted(EXPECTED_FILES):
            destination = data_dir / filename
            temporary_path = None

            try:
                # Download into a temporary file before publishing it
                with tempfile.NamedTemporaryFile(
                    dir=data_dir,
                    prefix=f".{filename}.",
                    suffix=".part",
                    delete=False,
                ) as stream:
                    temporary_path = Path(stream.name)

                print(
                    f"Downloading s3://{config['bucket']}/{objects[filename]}",
                    flush=True,
                )

                # Retrieve the approved dataset from S3
                client.download_file(
                    config["bucket"],
                    objects[filename],
                    str(temporary_path),
                )

                # Compare the download size against the saved manifest
                expected_size = manifest["datasets"][filename]["size_bytes"]

                if temporary_path.stat().st_size != expected_size:
                    raise ValueError(
                        f"Downloaded file size mismatch: {filename}"
                    )

                # Publish the completed download without overwriting a file
                os.link(temporary_path, destination)
                published_files.append(destination)

            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

        # Verify hashes and record counts for both downloaded files
        verify_datasets(data_dir)

    # Remove files from this run if any download or validation fails
    except Exception:
        for path in published_files:
            path.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=BASE_DIR / "data",
    )
    args = parser.parse_args()

    download_datasets(args.data_dir)


if __name__ == "__main__":
    main()
