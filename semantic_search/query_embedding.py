
import hashlib
import json
import math

REGION = "us-east-1"
MODEL_ID = "amazon.titan-embed-text-v2:0"
DIMENSIONS = 512
NORMALIZE = True
MAX_QUERY_CHARS = 1000


def validate_query(query):
    """Validate text before an embedding request."""
    if not isinstance(query, str):
        raise ValueError("Query must be a string")

    query = query.strip()

    if not query:
        raise ValueError("Query cannot be empty")

    if len(query) > MAX_QUERY_CHARS:
        raise ValueError("Query exceeds 1000 characters")

    if any(ord(ch) < 32 and ch not in "\n\t" for ch in query):
        raise ValueError("Query contains invalid control characters")

    return query


def cache_key(query):
    """Bind query identity to the exact embedding configuration."""
    query = validate_query(query)

    configuration = {
        "query": query,
        "model_id": MODEL_ID,
        "region": REGION,
        "dimensions": DIMENSIONS,
        "normalize": NORMALIZE,
    }

    canonical = json.dumps(
        configuration,
        sort_keys=True,
        ensure_ascii=False,
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def validate_embedding_response(response):
    """Validate a Titan V2 response before saving or searching."""
    if not isinstance(response, dict):
        raise ValueError("Invalid response format")

    vector = response.get("embedding")
    tokens = response.get("inputTextTokenCount")

    if not isinstance(vector, list) or len(vector) != DIMENSIONS:
        raise ValueError("Expected 512 embedding dimensions")

    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in vector
    ):
        raise ValueError("Invalid embedding values")

    norm = math.sqrt(sum(value * value for value in vector))

    if not math.isfinite(norm) or abs(norm - 1.0) > 0.01:
        raise ValueError("Embedding must be normalized")

    if type(tokens) is not int or not 0 < tokens <= 8192:
        raise ValueError("Invalid input token count")

    return {
        "embedding": vector,
        "inputTextTokenCount": tokens,
        "vector_norm": norm,
    }


def load_cached_embedding(query, cache_dir):
    """Return a validated cached response, or None if absent."""
    from pathlib import Path

    key = cache_key(query)
    cache_dir = Path(cache_dir)
    path = cache_dir / f"{key}.json"

    if not path.exists():
        return None

    record = json.loads(path.read_text(encoding="utf-8"))

    if record["cache_key"] != key:
        raise ValueError("Cache key mismatch")

    if record["query"] != validate_query(query):
        raise ValueError("Cached query mismatch")

    return validate_embedding_response(record["response"])


def save_cached_embedding(query, response, cache_dir):
    """Save a validated response without overwriting existing data."""
    import os
    from pathlib import Path

    query = validate_query(query)
    validated = validate_embedding_response(response)

    key = cache_key(query)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    path = cache_dir / f"{key}.json"
    temporary = cache_dir / f".{key}.tmp"

    if path.exists():
        existing = load_cached_embedding(query, cache_dir)

        if existing != validated:
            raise ValueError("Conflicting cached embedding")

        return path

    record = {
        "cache_key": key,
        "query": query,
        "response": {
            "embedding": validated["embedding"],
            "inputTextTokenCount": validated["inputTextTokenCount"],
        },
    }

    payload = json.dumps(
        record,
        indent=2,
        allow_nan=False,
    ) + "\n"

    fd = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())

        # Publish exclusively: never replace another writer's entry.
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing = load_cached_embedding(query, cache_dir)
            if existing != validated:
                raise ValueError("Conflicting concurrent cache entry")

        dir_fd = os.open(cache_dir, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    finally:
        if temporary.exists():
            temporary.unlink()

    return path
