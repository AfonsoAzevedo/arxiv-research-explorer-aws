import hashlib
import json
from pathlib import Path

BASE = Path("semantic_search/data")
SOURCE = BASE / "pilot_50.jsonl"
SAMPLING_MANIFEST = BASE / "pilot_50_manifest.json"
OUTPUT = BASE / "pilot_50_embedding_manifest.json"

EXPECTED_SHA256 = (
    "c2038433b371e4a17fd8d3bbeec77ca0fe82b4bb9530d0325514ccb2c23c64c0"
)

MODEL_ID = "amazon.titan-embed-text-v2:0"
REGION = "us-east-1"
DIMENSIONS = 512
NORMALIZE = True

MAX_REQUESTS = 50
INITIAL_BATCH = 2
MAX_INPUT_CHARS = 6000
MAX_TOKENS_PER_REQUEST = 8192

PRICE_PER_MILLION_TOKENS_USD = 0.02

raw = SOURCE.read_bytes()
actual_hash = hashlib.sha256(raw).hexdigest()

sampling_manifest = json.loads(
    SAMPLING_MANIFEST.read_text(encoding="utf-8")
)

assert actual_hash == EXPECTED_SHA256
assert sampling_manifest["sha256"] == actual_hash

papers = [json.loads(line) for line in raw.splitlines()]
ids = [paper["paper_id"] for paper in papers]

assert len(papers) == 50
assert len(set(ids)) == 50

for paper in papers:
    assert paper["input_text"] == (
        paper["title"] + "\n\n" + paper["abstract"]
    )
    assert 0 < len(paper["input_text"]) <= MAX_INPUT_CHARS

max_tokens = MAX_REQUESTS * MAX_TOKENS_PER_REQUEST
max_cost = max_tokens / 1_000_000 * PRICE_PER_MILLION_TOKENS_USD

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
    "initial_batch": INITIAL_BATCH,
    "max_input_characters": MAX_INPUT_CHARS,
    "max_tokens_per_request": MAX_TOKENS_PER_REQUEST,
    "max_estimated_input_tokens": max_tokens,
    "estimated_max_input_cost_usd": max_cost,
    "price_assumption_usd_per_million_tokens": PRICE_PER_MILLION_TOKENS_USD,
    "execution_status": "NOT_STARTED",
}

OUTPUT.write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8",
)

print("50-PAPER EMBEDDING PREFLIGHT PASSED")
print("Documents:", len(papers))
print("Unique IDs:", len(set(ids)))
print("Dataset SHA-256:", actual_hash)
print("Model:", MODEL_ID)
print("Region:", REGION)
print("Dimensions:", DIMENSIONS)
print("Initial batch limit:", INITIAL_BATCH)
print("Total request limit:", MAX_REQUESTS)
print("Maximum estimated input tokens:", max_tokens)
print(f"Conservative estimated input cost: ${max_cost:.6f}")
print("Output manifest:", OUTPUT)
print("AWS requests made: 0")
