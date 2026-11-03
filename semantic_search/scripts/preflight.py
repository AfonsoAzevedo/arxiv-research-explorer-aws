import hashlib
import json
from pathlib import Path

SOURCE = Path("semantic_search/data/pilot_10.jsonl")
MANIFEST = Path("semantic_search/data/pilot_manifest.json")

EXPECTED_SHA256 = (
    "77a9b4592275b9d0d16e5f7d8b89eb4207c46abbab7b05c7afcf8f68fc31dd78"
)

MODEL_ID = "amazon.titan-embed-text-v2:0"
REGION = "us-east-1"
DIMENSIONS = 512
NORMALIZE = True

MAX_REQUESTS = 10
MAX_INPUT_CHARS = 6000
MAX_TOKENS_PER_REQUEST = 8192

PRICE_PER_MILLION_TOKENS_USD = 0.02

raw = SOURCE.read_bytes()
actual_hash = hashlib.sha256(raw).hexdigest()

assert actual_hash == EXPECTED_SHA256, "Dataset fingerprint mismatch"

papers = [json.loads(line) for line in raw.splitlines()]
assert len(papers) == MAX_REQUESTS

ids = [p["paper_id"] for p in papers]
assert len(set(ids)) == len(ids), "Duplicate paper IDs"

for paper in papers:
    text = paper["input_text"]
    assert isinstance(text, str) and text.strip()
    assert len(text) <= MAX_INPUT_CHARS
    assert text == paper["title"] + "\n\n" + paper["abstract"]

estimated_max_tokens = MAX_REQUESTS * MAX_TOKENS_PER_REQUEST
estimated_max_cost = (
    estimated_max_tokens
    / 1_000_000
    * PRICE_PER_MILLION_TOKENS_USD
)

manifest = {
    "source": str(SOURCE),
    "source_sha256": actual_hash,
    "paper_count": len(papers),
    "paper_ids": ids,
    "aws_region": REGION,
    "model_id": MODEL_ID,
    "dimensions": DIMENSIONS,
    "normalize": NORMALIZE,
    "max_requests": MAX_REQUESTS,
    "max_input_characters": MAX_INPUT_CHARS,
    "max_tokens_per_request": MAX_TOKENS_PER_REQUEST,
    "max_estimated_input_tokens": estimated_max_tokens,
    "estimated_max_input_cost_usd": estimated_max_cost,
    "price_assumption_usd_per_million_tokens": PRICE_PER_MILLION_TOKENS_USD,
    "execution_status": "NOT_STARTED",
}

MANIFEST.write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8",
)

print("PREFLIGHT PASSED")
print("SHA-256:", actual_hash)
print("Documents:", len(papers))
print("Region:", REGION)
print("Model:", MODEL_ID)
print("Dimensions:", DIMENSIONS)
print("Maximum requests:", MAX_REQUESTS)
print("Maximum input tokens (conservative):", estimated_max_tokens)
print(f"Maximum estimated token cost: ${estimated_max_cost:.6f}")
print("Manifest:", MANIFEST)
print("Bedrock requests made: 0")
