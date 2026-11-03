import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

BASE = Path("semantic_search/data")
QUERY_DIR = BASE / "queries"
CACHE_PATH = QUERY_DIR / "query_cache.json"
JOURNAL_PATH = QUERY_DIR / "query_journal.json"

REGION = "us-east-1"
MODEL_ID = "amazon.titan-embed-text-v2:0"
DIMENSIONS = 512
NORMALIZE = True

MAX_REQUESTS = 1
MAX_QUERY_CHARS = 1000

QUERY = (
    "Machine learning methods for improving "
    "classification robustness and prediction accuracy"
)


def atomic_save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")

    with temp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp, path)


def load_json(path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def cache_key(query):
    payload = {
        "query": query,
        "model_id": MODEL_ID,
        "region": REGION,
        "dimensions": DIMENSIONS,
        "normalize": NORMALIZE,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_response(response):
    vector = response["embedding"]
    tokens = response["inputTextTokenCount"]

    if not isinstance(vector, list) or len(vector) != DIMENSIONS:
        raise ValueError("Invalid embedding dimensions")

    if not all(
        isinstance(x, (int, float)) and math.isfinite(x)
        for x in vector
    ):
        raise ValueError("Invalid vector values")

    norm = math.sqrt(sum(x * x for x in vector))

    if abs(norm - 1.0) > 0.01:
        raise ValueError("Embedding is not normalized")

    if type(tokens) is not int or not 0 < tokens <= 8192:
        raise ValueError("Invalid token count")

    return vector, tokens, norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if not QUERY.strip() or len(QUERY) > MAX_QUERY_CHARS:
        raise ValueError("Invalid query")

    key = cache_key(QUERY)

    cache_exists = CACHE_PATH.exists()
    journal_exists = JOURNAL_PATH.exists()

    if cache_exists != journal_exists:
        raise RuntimeError("Incomplete cache/journal state")

    cache = load_json(CACHE_PATH, {"entries": {}})
    journal = load_json(JOURNAL_PATH, {"attempts": []})

    entries = cache["entries"]
    attempts = journal["attempts"]

    if not isinstance(entries, dict) or not isinstance(attempts, list):
        raise RuntimeError("Invalid cache or journal")

    if len(attempts) > MAX_REQUESTS:
        raise RuntimeError("Request limit exceeded")

    if key in entries:
        vector, tokens, norm = validate_response(entries[key])

        print("CACHE HIT")
        print("Query:", QUERY)
        print("Dimensions:", len(vector))
        print("Tokens:", tokens)
        print("Norm:", round(norm, 6))
        print("AWS requests made: 0")
        return

    print("Query:", QUERY)
    print("Cache key:", key)
    print("Previous request attempts:", len(attempts))
    print("Remaining request slots:", MAX_REQUESTS - len(attempts))

    if not args.execute:
        print("DRY RUN: No AWS requests made.")
        return

    if len(attempts) >= MAX_REQUESTS:
        raise RuntimeError("No request slots remain")

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

    atomic_save(CACHE_PATH, cache)
    atomic_save(JOURNAL_PATH, journal)

    attempts.append({
        "cache_key": key,
        "status": "reserved",
        "timestamp_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),
    })
    atomic_save(JOURNAL_PATH, journal)

    body = {
        "inputText": QUERY,
        "dimensions": DIMENSIONS,
        "normalize": NORMALIZE,
    }

    start = time.perf_counter()

    try:
        response = client.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )

        payload = json.loads(response["body"].read())
        vector, tokens, norm = validate_response(payload)
        latency = time.perf_counter() - start

        entries[key] = {
            "embedding": vector,
            "inputTextTokenCount": tokens,
            "latency_seconds": round(latency, 4),
            "model_id": MODEL_ID,
            "region": REGION,
            "dimensions": DIMENSIONS,
            "normalize": NORMALIZE,
        }

        atomic_save(CACHE_PATH, cache)

        attempts[-1]["status"] = "saved"
        atomic_save(JOURNAL_PATH, journal)

        print("QUERY EMBEDDING SUCCESS")
        print("Dimensions:", len(vector))
        print("Input tokens:", tokens)
        print("Latency seconds:", round(latency, 4))
        print("Vector norm:", round(norm, 6))
        print("Estimated input cost USD:", tokens / 1_000_000 * 0.02)

    except Exception as exc:
        attempts[-1]["status"] = "failed_or_uncertain"
        attempts[-1]["error_type"] = type(exc).__name__
        atomic_save(JOURNAL_PATH, journal)

        print("STOPPED:", type(exc).__name__, str(exc))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
