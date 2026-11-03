#!/usr/bin/env python3
"""Controlled, resumable Titan V2 embedding runner for batch_001."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "semantic_search/data"
BATCH = DATA / "embeddings_25k"

MANIFEST = BATCH / "batch_001_manifest.json"
AUTHORIZATION = BATCH / "batch_001_authorization.json"
STATE = BATCH / "batch_001_state.json"
LOCK = BATCH / "batch_001.lock"

EXPECTED_MANIFEST_SHA = (
    "ce0b90c91c8947b43af03eb715e8da3742ae73afe5fce591909aad16df384bff"
)
EXPECTED_DATASET_SHA = (
    "3e2679c47a04166467ea3b848748787fe4fea62993568fc485eb891e0c639a7c"
)

MODEL_ID = "amazon.titan-embed-text-v2:0"
REGION = "us-east-1"
DIMENSIONS = 512


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path, value):
    """Write a new snapshot durably before replacing the previous one."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def verify_inputs():
    assert MANIFEST.is_file()
    assert sha256(MANIFEST.read_bytes()) == EXPECTED_MANIFEST_SHA

    manifest = load_json(MANIFEST)
    assert manifest["batch_id"] == "batch_001"
    assert manifest["source_dataset_sha256"] == EXPECTED_DATASET_SHA
    assert manifest["region"] == REGION
    assert manifest["model_id"] == MODEL_ID
    assert manifest["dimensions"] == DIMENSIONS
    assert manifest["normalize"] is True
    assert manifest["maximum_new_requests"] == 10
    assert manifest["execution_authorized"] is False

    dataset = ROOT / manifest["source_dataset"]
    assert dataset.is_file()
    assert sha256(dataset.read_bytes()) == EXPECTED_DATASET_SHA

    expected = {p["paper_id"]: p for p in manifest["papers"]}
    assert len(expected) == 10

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
        assert sha256(text.encode("utf-8")) == item["input_text_sha256"]
        assert len(text) == item["input_characters"]
        assert bool(text.strip())

    return manifest, inputs


def load_state(manifest):
    if not STATE.exists():
        return {
            "batch_id": manifest["batch_id"],
            "manifest_sha256": EXPECTED_MANIFEST_SHA,
            "attempts": [],
            "results": {},
        }

    state = load_json(STATE)
    assert state["batch_id"] == manifest["batch_id"]
    assert state["manifest_sha256"] == EXPECTED_MANIFEST_SHA
    assert isinstance(state["attempts"], list)
    assert isinstance(state["results"], dict)

    attempt_ids = [a["paper_id"] for a in state["attempts"]]
    assert len(attempt_ids) == len(set(attempt_ids))
    assert set(state["results"]) <= set(attempt_ids)
    assert set(attempt_ids) <= {p["paper_id"] for p in manifest["papers"]}
    assert len(attempt_ids) <= 10

    for attempt in state["attempts"]:
        paper_id = attempt["paper_id"]
        assert attempt["status"] in ("reserved", "completed")
        if attempt["status"] == "completed":
            assert paper_id in state["results"]
        else:
            assert paper_id not in state["results"]

    return state


def execute(manifest, inputs, state):
    import boto3
    from botocore.config import Config

    assert AUTHORIZATION.is_file(), (
        "Missing separate execution authorization. No AWS calls made."
    )

    authorization = load_json(AUTHORIZATION)
    assert authorization == {
        "batch_id": "batch_001",
        "manifest_sha256": EXPECTED_MANIFEST_SHA,
        "allow_paid_requests": True,
        "authorized_request_limit": 10,
        "region": REGION,
    }, "Authorization mismatch. No AWS calls made."

    # Refuse uncertain prior attempts and possible concurrent execution.
    assert all(
        a["status"] == "completed" for a in state["attempts"]
    ), "Unresolved request found. Manual review required."

    lock_fd = os.open(
        LOCK,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o600,
    )

    try:
        os.write(lock_fd, b"batch_001 execution lock\n")
        os.fsync(lock_fd)

        # A fresh state read protects against changes since preflight.
        state = load_state(manifest)
        assert all(
            a["status"] == "completed" for a in state["attempts"]
        )

        client = boto3.Session().client(
            "bedrock-runtime",
            region_name=REGION,
            config=Config(
                retries={"total_max_attempts": 1},
                connect_timeout=10,
                read_timeout=60,
            ),
        )

        for paper in manifest["papers"]:
            paper_id = paper["paper_id"]

            if paper_id in state["results"]:
                print("ALREADY SAVED:", paper_id)
                continue

            # The authorization caps distinct request attempts.
            assert len(state["attempts"]) < 10

            # Persist the request reservation BEFORE contacting AWS.
            state["attempts"].append({
                "paper_id": paper_id,
                "status": "reserved",
                "input_text_sha256": paper["input_text_sha256"],
            })
            atomic_json(STATE, state)

            # If anything below fails, the reservation remains.
            # The script will not automatically retry this paper.
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

            tokens = payload["inputTextTokenCount"]
            assert isinstance(tokens, int) and tokens > 0

            state["results"][paper_id] = {
                "embedding": vector,
                "input_tokens": tokens,
                "vector_norm": norm,
                "input_text_sha256": paper["input_text_sha256"],
            }
            state["attempts"][-1]["status"] = "completed"
            atomic_json(STATE, state)

            print("SAVED:", paper_id, "tokens:", tokens)

    finally:
        os.close(lock_fd)
        # Keep the lock if execution ended with unresolved attempts.
        latest = load_state(manifest)
        if all(a["status"] == "completed" for a in latest["attempts"]):
            LOCK.unlink()

    print("BATCH EXECUTION COMPLETED")


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    manifest, inputs = verify_inputs()
    state = load_state(manifest)

    print("BATCH 001 INPUT INTEGRITY VERIFIED")
    print("=" * 55)
    print("Region:", REGION)
    print("Model:", MODEL_ID)
    print("Papers:", len(inputs))
    print("Completed:", len(state["results"]))
    print("Recorded attempts:", len(state["attempts"]))
    print("Remaining:", len(inputs) - len(state["results"]))

    if not args.execute:
        print("\nDRY RUN SUCCESSFUL")
        print("Authorization required for execution: YES")
        print("AWS requests made: 0")
        print("Files modified by this run: 0")
        return

    execute(manifest, inputs, state)


if __name__ == "__main__":
    main()
