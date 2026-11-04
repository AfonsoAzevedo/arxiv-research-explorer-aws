"""Verify EC2 training and validation downloads against the committed manifest."""

import argparse
import hashlib
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = BASE_DIR / "ec2-data-manifest.json"

EXPECTED_FILES = {"train.jsonl", "validation.jsonl"}


def verify_datasets(data_dir: Path) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported dataset manifest version.")

    if manifest.get("aws_region") != "us-east-1":
        raise ValueError("Unexpected AWS region in manifest.")

    datasets = manifest["datasets"]

    if set(datasets) != EXPECTED_FILES:
        raise ValueError("Unexpected dataset filenames in manifest.")

    for filename in sorted(EXPECTED_FILES):
        path = data_dir / filename
        expected = datasets[filename]

        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(
                f"Missing or invalid regular dataset file: {path}"
            )

        size = path.stat().st_size
        if size != expected["size_bytes"]:
            raise ValueError(f"File size mismatch: {filename}")

        digest = hashlib.sha256()
        records = 0

        with path.open("rb") as stream:
            for line in stream:
                digest.update(line)
                records += 1

        if digest.hexdigest() != expected["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {filename}")

        if records != expected["records"]:
            raise ValueError(f"Record count mismatch: {filename}")

        print(
            f"PASS: {filename} | "
            f"{records:,} records | "
            f"{size:,} bytes | "
            f"SHA-256 {digest.hexdigest()}",
            flush=True,
        )

    print("\nALL EC2 DATASET INTEGRITY CHECKS PASSED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=BASE_DIR / "data",
        help="Directory containing the downloaded JSONL datasets.",
    )
    args = parser.parse_args()

    verify_datasets(args.data_dir)


if __name__ == "__main__":
    main()
