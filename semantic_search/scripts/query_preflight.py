import hashlib
import json
from pathlib import Path

BASE = Path("semantic_search/data")
MANIFEST = BASE / "pilot_manifest.json"
MAPPING = BASE / "faiss/paper_mapping.json"

QUERY = (
    "Machine learning methods for improving "
    "classification robustness and prediction accuracy"
)

MAX_QUERY_CHARS = 1000
MAX_NEW_REQUESTS = 1

manifest = json.loads(MANIFEST.read_text())
mapping = json.loads(MAPPING.read_text())

assert manifest["source_sha256"] == mapping["source_sha256"]
assert manifest["model_id"] == "amazon.titan-embed-text-v2:0"
assert manifest["aws_region"] == "us-east-1"
assert manifest["dimensions"] == 512
assert manifest["normalize"] is True

assert QUERY.strip()
assert len(QUERY) <= MAX_QUERY_CHARS
assert MAX_NEW_REQUESTS == 1

query_hash = hashlib.sha256(
    QUERY.encode("utf-8")
).hexdigest()

print("QUERY PREFLIGHT PASSED")
print("Query:", QUERY)
print("Query characters:", len(QUERY))
print("Query SHA-256:", query_hash)
print("Region:", manifest["aws_region"])
print("Model:", manifest["model_id"])
print("Dimensions:", manifest["dimensions"])
print("Maximum new requests:", MAX_NEW_REQUESTS)
print("Indexed papers:", len(mapping["paper_ids"]))
print("AWS requests made: 0")
