#!/usr/bin/env python3
"""Reusable Titan V2 embedding runner. Dry-run foundation."""

import argparse
import hashlib
import json
import os
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BATCH_DIRECTORY = ROOT / "semantic_search/data/embeddings_25k"

MODEL_ID = "amazon.titan-embed-text-v2:0"
REGION = "us-east-1"
DIMENSIONS = 512


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_manifest(path, expected_sha):
    assert path.is_file(), "Manifest does not exist"
    assert sha256(path) == expected_sha, "Manifest checksum mismatch"

    manifest = load_json(path)

    batch_id = manifest["batch_id"]

    assert isinstance(batch_id, str)
    assert batch_id
    assert all(
        c.isascii() and (c.isalnum() or c == "_")
        for c in batch_id
    )

    assert path.name == f"{batch_id}_manifest.json"
    assert manifest["region"] == REGION
    assert manifest["model_id"] == MODEL_ID
    assert manifest["dimensions"] == DIMENSIONS
    assert manifest["normalize"] is True
    assert manifest["execution_authorized"] is False

    papers = manifest["papers"]
    limit = manifest["maximum_new_requests"]

    assert isinstance(papers, list)
    assert len(papers) > 0
    assert type(limit) is int
    assert limit == len(papers)

    ids = [p["paper_id"] for p in papers]
    assert len(set(ids)) == len(ids)

    dataset = ROOT / manifest["source_dataset"]
    assert dataset.is_file()
    assert sha256(dataset) == manifest["source_dataset_sha256"]

    expected = {p["paper_id"]: p for p in papers}
    inputs = {}

    with dataset.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            paper = json.loads(line)
            paper_id = paper["paper_id"]

            if paper_id in expected:
                assert paper_id not in inputs
                inputs[paper_id] = paper["input_text"]

    assert set(inputs) == set(expected)

    for paper_id, item in expected.items():
        text = inputs[paper_id]

        assert bool(text.strip())
        assert len(text) == item["input_characters"]

        actual_hash = hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

        assert actual_hash == item["input_text_sha256"]

    return manifest, inputs





def atomic_json(path, value):
    """Atomically replace a checkpoint, cleaning up failed writes."""

    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")

    assert path.parent.is_dir(), "Checkpoint directory missing"

    # Serialize fully before creating a temporary file.
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"

    created = False

    try:
        with temporary.open("x", encoding="utf-8") as f:
            created = True
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())

        os.replace(temporary, path)
        created = False

        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    except BaseException:
        if created:
            temporary.unlink(missing_ok=True)
        raise



def verify_authorization(manifest, manifest_sha):
    """Verify explicit authorization for exactly one batch."""

    batch_id = manifest["batch_id"]

    authorization_path = (
        BATCH_DIRECTORY / f"{batch_id}_authorization.json"
    )

    assert authorization_path.is_file(), (
        "Missing separate execution authorization. "
        "No AWS calls permitted."
    )

    authorization = load_json(authorization_path)

    expected = {
        "batch_id": batch_id,
        "manifest_sha256": manifest_sha,
        "allow_paid_requests": True,
        "authorized_request_limit": manifest["maximum_new_requests"],
        "region": REGION,
    }

    assert authorization == expected, (
        "Execution authorization mismatch. "
        "No AWS calls permitted."
    )

    return authorization




def acquire_execution_lock(batch_id):
    """Create an exclusive lock with an ownership token."""

    lock_path = BATCH_DIRECTORY / f"{batch_id}.lock"
    token = secrets.token_hex(32)

    fd = os.open(
        lock_path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o600,
    )

    try:
        message = json.dumps({
            "batch_id": batch_id,
            "purpose": "embedding_execution",
            "owner_token": token,
        }) + "\n"

        os.write(fd, message.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)

    return lock_path, token


def release_execution_lock(lock_path, token):
    """Release only a lock belonging to this execution."""

    lock_path = Path(lock_path)

    assert lock_path.parent.resolve() == BATCH_DIRECTORY.resolve()
    assert lock_path.is_file()

    saved = load_json(lock_path)

    assert saved["owner_token"] == token, (
        "Execution lock ownership mismatch. "
        "Refusing to release."
    )

    lock_path.unlink()


def inspect_checkpoint(manifest, manifest_sha):
    """Inspect an existing request journal without modifying it."""

    batch_id = manifest["batch_id"]
    state_path = BATCH_DIRECTORY / f"{batch_id}_state.json"
    lock_path = BATCH_DIRECTORY / f"{batch_id}.lock"

    if not state_path.exists():
        assert not lock_path.exists(), (
            "Execution lock exists without a checkpoint. "
            "Manual review required."
        )
        return 0, 0, 0

    state = load_json(state_path)

    assert state["batch_id"] == batch_id
    assert state["manifest_sha256"] == manifest_sha

    attempts = state["attempts"]
    results = state["results"]

    assert isinstance(attempts, list)
    assert isinstance(results, dict)

    expected_ids = [
        paper["paper_id"] for paper in manifest["papers"]
    ]
    attempt_ids = [
        attempt["paper_id"] for attempt in attempts
    ]

    assert len(attempts) <= manifest["maximum_new_requests"]
    assert attempt_ids == expected_ids[:len(attempt_ids)]

    completed = set()
    unresolved = set()

    paper_lookup = {
        paper["paper_id"]: paper
        for paper in manifest["papers"]
    }

    for attempt in attempts:
        paper_id = attempt["paper_id"]

        assert attempt["input_text_sha256"] == (
            paper_lookup[paper_id]["input_text_sha256"]
        )

        assert attempt["status"] in ("reserved", "completed")

        if attempt["status"] == "completed":
            completed.add(paper_id)
        else:
            unresolved.add(paper_id)

    assert set(results) == completed
    assert not completed.intersection(unresolved)

    import math

    for paper_id, result in results.items():
        assert result["input_text_sha256"] == (
            paper_lookup[paper_id]["input_text_sha256"]
        )

        vector = result["embedding"]

        assert isinstance(vector, list)
        assert len(vector) == DIMENSIONS
        assert all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in vector
        )

        norm = math.sqrt(sum(value * value for value in vector))
        assert abs(norm - 1.0) < 0.01

        tokens = result["input_tokens"]
        assert type(tokens) is int and tokens > 0

    if unresolved:
        assert lock_path.exists(), (
            "Unresolved request without execution lock"
        )

    # A lock with no unresolved requests still requires review.
    assert not (lock_path.exists() and not unresolved), (
        "Unexpected execution lock. Manual review required."
    )

    return len(completed), len(attempts), len(unresolved)




def create_bedrock_client():
    """Create a region-specific Bedrock Runtime client.

    This only constructs the client. It does not invoke a model.
    """
    import boto3
    from botocore.config import Config

    return boto3.Session().client(
        "bedrock-runtime",
        region_name=REGION,
        config=Config(
            retries={"total_max_attempts": 1},
            connect_timeout=10,
            read_timeout=60,
        ),
    )


def execute_batch(manifest, manifest_sha, inputs, client, sleep_fn):
    """Execute one authorized batch with durable per-request journaling.

    The caller supplies the Bedrock client and sleep function,
    allowing isolated testing without AWS.
    """
    import math

    verify_authorization(manifest, manifest_sha)

    batch_id = manifest["batch_id"]
    state_path = BATCH_DIRECTORY / f"{batch_id}_state.json"

    completed, attempts, unresolved = inspect_checkpoint(
        manifest, manifest_sha
    )

    assert unresolved == 0, (
        "Unresolved request. Manual review required."
    )

    lock_path, owner_token = acquire_execution_lock(batch_id)

    try:
        # Revalidate after acquiring the exclusive lock.
        completed, attempts, unresolved = inspect_locked_state(
            manifest, manifest_sha
        )

        assert unresolved == 0

        if state_path.exists():
            state = load_json(state_path)
        else:
            state = {
                "batch_id": batch_id,
                "manifest_sha256": manifest_sha,
                "attempts": [],
                "results": {},
            }

        for paper in manifest["papers"]:
            paper_id = paper["paper_id"]

            if paper_id in state["results"]:
                continue

            assert len(state["attempts"]) < (
                manifest["maximum_new_requests"]
            )

            # Pacing occurs before reservation.
            sleep_fn(5)

            state["attempts"].append({
                "paper_id": paper_id,
                "status": "reserved",
                "input_text_sha256": paper["input_text_sha256"],
            })

            atomic_json(state_path, state)

            # Never automatically retry a reserved attempt.
            response = client.invoke_model(
                modelId=MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "inputText": inputs[paper_id],
                    "dimensions": DIMENSIONS,
                    "normalize": True,
                }),
            )

            payload = json.loads(response["body"].read())
            vector = payload["embedding"]
            tokens = payload["inputTextTokenCount"]

            assert isinstance(vector, list)
            assert len(vector) == DIMENSIONS
            assert all(
                isinstance(v, (int, float))
                and not isinstance(v, bool)
                and math.isfinite(v)
                for v in vector
            )

            norm = math.sqrt(sum(v * v for v in vector))
            assert abs(norm - 1.0) < 0.01
            assert type(tokens) is int and tokens > 0

            state["results"][paper_id] = {
                "embedding": vector,
                "input_tokens": tokens,
                "vector_norm": norm,
                "input_text_sha256": paper["input_text_sha256"],
            }

            state["attempts"][-1]["status"] = "completed"
            atomic_json(state_path, state)

            print("SAVED:", paper_id, "tokens:", tokens)

    finally:
        # Release only when there are no unresolved reservations.
        # The lock belongs to this execution and must not be
        # removed if its checkpoint cannot be inspected safely.
        if not state_path.exists():
            release_execution_lock(lock_path, owner_token)
        else:
            latest = load_json(state_path)
            assert latest["batch_id"] == batch_id
            assert latest["manifest_sha256"] == manifest_sha

            if all(
                a["status"] == "completed"
                for a in latest["attempts"]
            ):
                release_execution_lock(lock_path, owner_token)

    print("BATCH EXECUTION COMPLETED")



def inspect_locked_state(manifest, manifest_sha):
    """Fully validate checkpoint while holding our execution lock."""
    import math

    batch_id = manifest["batch_id"]
    state_path = BATCH_DIRECTORY / f"{batch_id}_state.json"
    lock_path = BATCH_DIRECTORY / f"{batch_id}.lock"

    assert lock_path.is_file()

    if not state_path.exists():
        return 0, 0, 0

    state = load_json(state_path)

    assert state["batch_id"] == batch_id
    assert state["manifest_sha256"] == manifest_sha

    attempts = state["attempts"]
    results = state["results"]

    assert isinstance(attempts, list)
    assert isinstance(results, dict)

    papers = manifest["papers"]
    expected_ids = [p["paper_id"] for p in papers]
    attempted_ids = [a["paper_id"] for a in attempts]

    assert attempted_ids == expected_ids[:len(attempted_ids)]
    assert len(attempts) <= manifest["maximum_new_requests"]

    lookup = {p["paper_id"]: p for p in papers}
    completed = set()
    unresolved = set()

    for attempt in attempts:
        paper_id = attempt["paper_id"]

        assert attempt["input_text_sha256"] == (
            lookup[paper_id]["input_text_sha256"]
        )

        assert attempt["status"] in ("reserved", "completed")

        if attempt["status"] == "completed":
            completed.add(paper_id)
        else:
            unresolved.add(paper_id)

    assert set(results) == completed
    assert not completed.intersection(unresolved)

    for paper_id, result in results.items():
        assert result["input_text_sha256"] == (
            lookup[paper_id]["input_text_sha256"]
        )

        vector = result["embedding"]

        assert isinstance(vector, list)
        assert len(vector) == DIMENSIONS
        assert all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in vector
        )

        norm = math.sqrt(sum(v * v for v in vector))
        assert abs(norm - 1.0) < 0.01
        assert math.isclose(
            result["vector_norm"],
            norm,
            rel_tol=0,
            abs_tol=1e-10,
        )

        tokens = result["input_tokens"]
        assert type(tokens) is int and tokens > 0

    return len(completed), len(attempts), len(unresolved)


def main():
    parser = argparse.ArgumentParser(
        description="Reusable Titan V2 embedding runner"
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--manifest-sha256",
        required=True,
    )

    mode = parser.add_mutually_exclusive_group(required=True)

    mode.add_argument(
        "--dry-run",
        action="store_true",
    )

    mode.add_argument(
        "--execute",
        action="store_true",
    )

    args = parser.parse_args()

    path = args.manifest.resolve()

    assert path.parent == BATCH_DIRECTORY.resolve(), (
        "Manifest must be in the embedding batch directory"
    )

    manifest, inputs = verify_manifest(
        path,
        args.manifest_sha256,
    )

    print("REUSABLE RUNNER INPUT INTEGRITY VERIFIED")
    print("=" * 60)
    print("Batch:", manifest["batch_id"])
    print("Region:", REGION)
    print("Model:", MODEL_ID)
    print("Dimensions:", DIMENSIONS)
    completed, attempts, unresolved = inspect_checkpoint(
        manifest,
        args.manifest_sha256,
    )

    print("Papers:", len(inputs))
    print("Maximum requests:", manifest["maximum_new_requests"])
    print("Completed:", completed)
    print("Recorded attempts:", attempts)
    print("Unresolved attempts:", unresolved)
    print("Remaining:", len(inputs) - completed)
    print("Manifest SHA256:", sha256(path))
    if args.dry_run:
        print()
        print("DRY RUN SUCCESSFUL")
        print("AWS requests made: 0")
        print("Files modified by runner: 0")
        return

    # These checks MUST precede client construction.
    assert unresolved == 0, (
        "Unresolved request. Manual review required."
    )

    verify_authorization(
        manifest,
        args.manifest_sha256,
    )

    import time

    client = create_bedrock_client()

    execute_batch(
        manifest,
        args.manifest_sha256,
        inputs,
        client,
        time.sleep,
    )


if __name__ == "__main__":
    main()
