import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

BASE = Path("semantic_search/data")
SOURCE = BASE / "pilot_50.jsonl"
MANIFEST = BASE / "pilot_50_embedding_manifest.json"
STATE = BASE / "embeddings_50/state.json"

EXPECTED_SHA = (
    "c2038433b371e4a17fd8d3bbeec77ca0fe82b4bb9530d0325514ccb2c23c64c0"
)
MODEL = "amazon.titan-embed-text-v2:0"
REGION = "us-east-1"
DIMENSIONS = 512
MAX_REQUESTS = 50
MAX_CHARS = 6000
MAX_TOKENS = 8192
PRICE = 0.02


def save_atomic(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def validate_response(payload):
    vector = payload["embedding"]
    tokens = payload["inputTextTokenCount"]

    if not isinstance(vector, list) or len(vector) != DIMENSIONS:
        raise ValueError("Invalid embedding dimensions")
    if not all(
        type(x) in (float, int) and math.isfinite(x)
        for x in vector
    ):
        raise ValueError("Invalid embedding values")

    norm = math.sqrt(sum(x * x for x in vector))
    if abs(norm - 1.0) > 0.01:
        raise ValueError("Embedding norm outside tolerance")
    if type(tokens) is not int or not 0 < tokens <= MAX_TOKENS:
        raise ValueError("Invalid token count")

    return vector, tokens, norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-new", type=int, default=2)
    args = parser.parse_args()

    if not 1 <= args.max_new <= MAX_REQUESTS:
        raise ValueError("Invalid --max-new")

    raw = SOURCE.read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    if source_sha != EXPECTED_SHA:
        raise RuntimeError("Unexpected dataset fingerprint")

    expected = {
        "source_sha256": EXPECTED_SHA,
        "paper_count": 50,
        "aws_region": REGION,
        "model_id": MODEL,
        "dimensions": DIMENSIONS,
        "normalize": True,
        "max_requests": MAX_REQUESTS,
        "max_input_characters": MAX_CHARS,
        "max_tokens_per_request": MAX_TOKENS,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Manifest mismatch: {key}")

    papers = [json.loads(line) for line in raw.splitlines()]
    ids = [p["paper_id"] for p in papers]

    if len(ids) != 50 or len(set(ids)) != 50:
        raise RuntimeError("Invalid paper IDs")
    if ids != manifest["paper_ids"]:
        raise RuntimeError("Paper order differs from manifest")

    for paper in papers:
        text = paper["input_text"]
        if (
            not text.strip()
            or len(text) > MAX_CHARS
            or text != paper["title"] + "\n\n" + paper["abstract"]
        ):
            raise RuntimeError("Invalid paper text")

    if STATE.exists():
        state = json.loads(STATE.read_text(encoding="utf-8"))
    else:
        state = {
            "source_sha256": source_sha,
            "model_id": MODEL,
            "region": REGION,
            "dimensions": DIMENSIONS,
            "normalize": True,
            "attempts": [],
            "results": {},
        }

    for key, value in {
        "source_sha256": source_sha,
        "model_id": MODEL,
        "region": REGION,
        "dimensions": DIMENSIONS,
        "normalize": True,
    }.items():
        if state.get(key) != value:
            raise RuntimeError(f"State mismatch: {key}")

    attempts = state["attempts"]
    results = state["results"]

    attempted_ids = [a["paper_id"] for a in attempts]

    if (
        len(attempts) > MAX_REQUESTS
        or len(set(attempted_ids)) != len(attempted_ids)
        or not set(attempted_ids).issubset(set(ids))
        or not set(results).issubset(set(attempted_ids))
    ):
        raise RuntimeError("Invalid checkpoint state")

    # Do not automatically resume after failures or uncertain calls.
    if any(a["status"] != "saved" for a in attempts):
        raise RuntimeError(
            "Unresolved request attempt. Manual review required."
        )

    if set(results) != set(attempted_ids):
        raise RuntimeError("Attempt/result mismatch")

    saved_tokens = 0
    for paper_id, result in results.items():
        _, tokens, _ = validate_response({
            "embedding": result["embedding"],
            "inputTextTokenCount": result["input_tokens"],
        })
        saved_tokens += tokens

    available = MAX_REQUESTS - len(attempts)
    remaining = [p for p in papers if p["paper_id"] not in results]
    selected = remaining[:min(args.max_new, available)]

    print("Dataset SHA-256:", source_sha)
    print("Region:", REGION)
    print("Model:", MODEL)
    print("Dimensions:", DIMENSIONS)
    print("Previous attempts:", len(attempts))
    print("Saved embeddings:", len(results))
    print("Saved input tokens:", saved_tokens)
    print("Remaining request slots:", available)
    print("Selected for this run:", len(selected))
    print("Paper IDs:", [p["paper_id"] for p in selected])

    if not args.execute:
        print("DRY RUN PASSED. AWS requests made: 0")
        return

    if not selected:
        print("No new requests available.")
        return

    import boto3
    from botocore.config import Config

    client = boto3.client(
        "bedrock-runtime",
        region_name=REGION,
        config=Config(
            retries={"total_max_attempts": 1},
            connect_timeout=10,
            read_timeout=60,
        ),
    )

    save_atomic(STATE, state)

    for paper in selected:
        paper_id = paper["paper_id"]
        attempt = {
            "paper_id": paper_id,
            "status": "reserved",
            "timestamp_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
        }

        # Persist reservation before the API request.
        attempts.append(attempt)
        save_atomic(STATE, state)

        start = time.perf_counter()

        try:
            response = client.invoke_model(
                modelId=MODEL,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "inputText": paper["input_text"],
                    "dimensions": DIMENSIONS,
                    "normalize": True,
                }),
            )

            payload = json.loads(response["body"].read())
            vector, tokens, norm = validate_response(payload)
            latency = time.perf_counter() - start

            results[paper_id] = {
                "embedding": vector,
                "input_tokens": tokens,
                "latency_seconds": round(latency, 4),
                "vector_norm": norm,
            }
            attempt["status"] = "saved"
            save_atomic(STATE, state)

            print(
                f"SUCCESS {paper_id} | "
                f"tokens={tokens} | "
                f"latency={latency:.3f}s | "
                f"norm={norm:.6f}"
            )

        except Exception as exc:
            attempt["status"] = "failed_or_uncertain"
            attempt["error_type"] = type(exc).__name__
            save_atomic(STATE, state)

            print(f"STOPPED: {type(exc).__name__}: {exc}")
            raise SystemExit(1)

    total_tokens = sum(
        r["input_tokens"] for r in results.values()
    )

    print("\nRUN COMPLETE")
    print("Saved embeddings:", len(results))
    print("Recorded attempts:", len(attempts))
    print("Total saved tokens:", total_tokens)
    print(f"Estimated token cost: ${total_tokens * PRICE / 1e6:.8f}")
    print("Actual AWS charges must be checked separately.")


if __name__ == "__main__":
    main()
