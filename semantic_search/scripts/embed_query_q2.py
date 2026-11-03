import argparse
import json
import math
import os
import time
from pathlib import Path

BASE = Path("semantic_search/data/evaluation_50/Q2")
CONFIG_PATH = BASE / "query_config.json"
EMBEDDING_PATH = BASE / "query_embedding.json"
JOURNAL_PATH = BASE / "request_journal.json"


def save_new_json(path, data):
    """Create a JSON file without overwriting existing data."""
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def validate_embedding(payload):
    vector = payload["embedding"]
    tokens = payload["inputTextTokenCount"]

    if not isinstance(vector, list) or len(vector) != 512:
        raise ValueError("Expected a 512-dimensional embedding")

    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in vector
    ):
        raise ValueError("Invalid embedding values")

    norm = math.sqrt(sum(value * value for value in vector))

    if abs(norm - 1.0) > 0.01:
        raise ValueError("Embedding is not normalized")

    if type(tokens) is not int or not 0 < tokens <= 8192:
        raise ValueError("Invalid input token count")

    return norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config = json.loads(
        CONFIG_PATH.read_text(encoding="utf-8")
    )

    assert config["query_id"] == "Q2"
    assert config["region"] == "us-east-1"
    assert config["embedding_model"] == "amazon.titan-embed-text-v2:0"
    assert config["dimensions"] == 512
    assert config["normalize"] is True

    query = config["query"]
    assert isinstance(query, str)
    assert 0 < len(query.strip()) <= 1000

    print("Q2 EMBEDDING PREFLIGHT")
    print("=" * 45)
    print("Query:", query)
    print("Region:", config["region"])
    print("Model:", config["embedding_model"])
    print("Dimensions:", config["dimensions"])
    print("Authorized:", config["aws_requests_authorized"])
    print("Execute flag:", args.execute)

    # Never contact AWS when authorization is disabled.
    if not args.execute:
        print("DRY RUN PASSED")
        print("AWS requests made: 0")
        return

    if config["aws_requests_authorized"] is not True:
        raise RuntimeError(
            "AWS request blocked: Q2 authorization is FALSE"
        )

    if EMBEDDING_PATH.exists():
        raise RuntimeError(
            "Q2 embedding already exists. Refusing duplicate request."
        )

    if JOURNAL_PATH.exists():
        raise RuntimeError(
            "Q2 request journal already exists. "
            "Refusing another request."
        )

    import boto3
    from botocore.config import Config

    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        config=Config(
            retries={"total_max_attempts": 1},
            connect_timeout=10,
            read_timeout=60,
        ),
    )

    # Reserve the request before invoking Bedrock.
    journal = {
        "query_id": "Q2",
        "query": query,
        "status": "reserved",
        "timestamp_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),
    }

    save_new_json(JOURNAL_PATH, journal)

    body = {
        "inputText": query,
        "dimensions": 512,
        "normalize": True,
    }

    started = time.perf_counter()

    try:
        response = client.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )

        payload = json.loads(response["body"].read())
        norm = validate_embedding(payload)

        result = {
            "query_id": "Q2",
            "query": query,
            "model_id": config["embedding_model"],
            "region": config["region"],
            "dimensions": 512,
            "normalize": True,
            "embedding": payload["embedding"],
            "inputTextTokenCount": payload["inputTextTokenCount"],
            "latency_seconds": round(
                time.perf_counter() - started, 4
            ),
        }

        save_new_json(EMBEDDING_PATH, result)

        # The journal remains a permanent record of the attempt.
        print("Q2 EMBEDDING SAVED")
        print("Dimensions:", len(result["embedding"]))
        print("Input tokens:", result["inputTextTokenCount"])
        print("Vector norm:", round(norm, 6))
        print("Embedding file:", EMBEDDING_PATH)

    except Exception as exc:
        # Do not retry automatically after uncertain outcomes.
        print("REQUEST FAILED OR RESULT UNCERTAIN")
        print("Error type:", type(exc).__name__)
        print("Further requests blocked by request journal.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
