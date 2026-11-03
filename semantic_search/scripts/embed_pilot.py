import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

SOURCE = Path("semantic_search/data/pilot_10.jsonl")
MANIFEST = Path("semantic_search/data/pilot_manifest.json")
OUTPUT_DIR = Path("semantic_search/data/embeddings")
JOURNAL = OUTPUT_DIR / "request_journal.json"
CHECKPOINT = OUTPUT_DIR / "checkpoint.json"

EXPECTED_MODEL = "amazon.titan-embed-text-v2:0"
EXPECTED_REGION = "us-east-1"
EXPECTED_DIMENSIONS = 512


def save_json_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")

    with temp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp, path)


def load_json(path, default):
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def validate_embedding(response):
    vector = response["embedding"]
    tokens = response["inputTextTokenCount"]

    if not isinstance(vector, list) or len(vector) != 512:
        raise ValueError("Invalid embedding dimensions")

    if any(
        not isinstance(x, (int, float)) or not math.isfinite(x)
        for x in vector
    ):
        raise ValueError("Non-finite embedding values")

    norm = math.sqrt(sum(x * x for x in vector))
    if not 0.99 <= norm <= 1.01:
        raise ValueError(f"Unexpected vector norm: {norm}")

    if type(tokens) is not int or not 0 < tokens <= 8192:
        raise ValueError(f"Invalid token count: {tokens}")

    return vector, tokens, norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-new", type=int, default=2)
    args = parser.parse_args()

    if not 1 <= args.max_new <= 10:
        raise ValueError("--max-new must be between 1 and 10")

    manifest = load_json(MANIFEST, None)
    if manifest is None:
        raise RuntimeError("Missing preflight manifest")

    actual_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if actual_hash != manifest["source_sha256"]:
        raise RuntimeError("Dataset hash mismatch")

    if (
        manifest["model_id"] != EXPECTED_MODEL
        or manifest["aws_region"] != EXPECTED_REGION
        or manifest["dimensions"] != EXPECTED_DIMENSIONS
        or manifest["normalize"] is not True
        or manifest["max_requests"] != 10
        or manifest["max_input_characters"] != 6000
        or manifest["max_tokens_per_request"] != 8192
    ):
        raise RuntimeError("Unexpected manifest configuration")

    papers = [
        json.loads(line)
        for line in SOURCE.read_text(encoding="utf-8").splitlines()
    ]

    ids = [p["paper_id"] for p in papers]
    if len(ids) != 10 or len(set(ids)) != 10:
        raise RuntimeError("Invalid paper count or duplicate IDs")

    for paper in papers:
        text = paper["input_text"]
        if (
            not isinstance(text, str)
            or not text.strip()
            or len(text) > 6000
            or text != paper["title"] + "\n\n" + paper["abstract"]
        ):
            raise RuntimeError("Invalid paper input text")

    journal_exists = JOURNAL.exists()
    checkpoint_exists = CHECKPOINT.exists()

    if journal_exists != checkpoint_exists:
        raise RuntimeError("Incomplete checkpoint state. Stop for review.")

    journal = load_json(JOURNAL, {"attempts": []})
    checkpoint = load_json(
        CHECKPOINT,
        {
            "source_sha256": actual_hash,
            "model_id": EXPECTED_MODEL,
            "dimensions": EXPECTED_DIMENSIONS,
            "normalize": True,
            "results": {},
        },
    )

    attempts = journal["attempts"]
    results = checkpoint["results"]

    if (
        checkpoint["source_sha256"] != actual_hash
        or checkpoint["model_id"] != EXPECTED_MODEL
        or checkpoint["dimensions"] != EXPECTED_DIMENSIONS
        or checkpoint["normalize"] is not True
    ):
        raise RuntimeError("Checkpoint metadata mismatch")

    if (
        not isinstance(attempts, list)
        or not isinstance(results, dict)
        or len(attempts) > 10
        or not set(results).issubset(set(ids))
    ):
        raise RuntimeError("Invalid saved state")

    attempted_ids = [a["paper_id"] for a in attempts]
    if (
        len(set(attempted_ids)) != len(attempted_ids)
        or not set(attempted_ids).issubset(set(ids))
        or not set(results).issubset(set(attempted_ids))
    ):
        raise RuntimeError("Inconsistent request journal")

    remaining = [p for p in papers if p["paper_id"] not in attempted_ids]
    available_slots = 10 - len(attempts)
    selected = remaining[:min(args.max_new, available_slots)]

    print("Region:", EXPECTED_REGION)
    print("Model:", EXPECTED_MODEL)
    print("Dimensions:", EXPECTED_DIMENSIONS)
    print("Dataset SHA-256:", actual_hash)
    print("Previous attempts:", len(attempts))
    print("Saved embeddings:", len(results))
    print("Available request slots:", available_slots)
    print("Selected for this run:", len(selected))
    print("Paper IDs:", [p["paper_id"] for p in selected])

    if not args.execute:
        print("DRY RUN: No AWS API calls made.")
        return

    if not selected:
        print("No eligible requests remaining.")
        return

    import boto3
    from botocore.config import Config

    client = boto3.client(
        "bedrock-runtime",
        region_name=EXPECTED_REGION,
        config=Config(
            retries={"total_max_attempts": 1},
            connect_timeout=10,
            read_timeout=60,
        ),
    )

    # Establish both files before the first request.
    save_json_atomic(JOURNAL, journal)
    save_json_atomic(CHECKPOINT, checkpoint)

    for paper in selected:
        paper_id = paper["paper_id"]

        # Reserve a request slot BEFORE invoking Bedrock.
        attempts.append({
            "paper_id": paper_id,
            "status": "reserved",
            "timestamp_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
        })
        save_json_atomic(JOURNAL, journal)

        body = {
            "inputText": paper["input_text"],
            "dimensions": EXPECTED_DIMENSIONS,
            "normalize": True,
        }

        start = time.perf_counter()

        try:
            response = client.invoke_model(
                modelId=EXPECTED_MODEL,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )

            payload = json.loads(response["body"].read())
            vector, tokens, norm = validate_embedding(payload)

            latency = time.perf_counter() - start

            results[paper_id] = {
                "embedding": vector,
                "input_tokens": tokens,
                "latency_seconds": round(latency, 4),
                "vector_norm": norm,
            }

            save_json_atomic(CHECKPOINT, checkpoint)

            attempts[-1]["status"] = "saved"
            save_json_atomic(JOURNAL, journal)

            print(
                f"SUCCESS {paper_id} | "
                f"tokens={tokens} | "
                f"latency={latency:.3f}s | "
                f"norm={norm:.6f}"
            )

        except Exception as exc:
            attempts[-1]["status"] = "failed_or_uncertain"
            attempts[-1]["error_type"] = type(exc).__name__
            save_json_atomic(JOURNAL, journal)

            print(
                f"STOPPED at {paper_id}: "
                f"{type(exc).__name__}: {exc}"
            )
            raise SystemExit(1)

    total_tokens = sum(
        result["input_tokens"] for result in results.values()
    )

    print("\nRUN COMPLETE")
    print("Total saved embeddings:", len(results))
    print("Total saved input tokens:", total_tokens)
    print(
        "Estimated saved-token cost: "
        f"${total_tokens / 1_000_000 * 0.02:.8f}"
    )
    print("Actual AWS billing must be checked separately.")


if __name__ == "__main__":
    main()
