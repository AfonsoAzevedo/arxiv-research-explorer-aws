# Generates SHA-256 hashes for cache identifiers
import hashlib

# Reads and writes JSON data
import json

# Performs mathematical operations to validate embedding vectors
import math

REGION = "us-east-1"
MODEL_ID = "amazon.titan-embed-text-v2:0"
DIMENSIONS = 512
NORMALIZE = True
MAX_QUERY_CHARS = 1000  # Limit research queries to 1000 characters


def validate_query(query):
    """Validate and clean a research query"""
    if not isinstance(query, str):
        raise ValueError("Query must be a string")

    query = query.strip()

    # Input validation
    if not query:
        raise ValueError("Query cannot be empty")

    if len(query) > MAX_QUERY_CHARS:
        raise ValueError("Query exceeds 1000 characters")

    if any(ord(ch) < 32 and ch not in "\n\t" for ch in query):
        raise ValueError("Query contains invalid control characters")

    return query


def cache_key(query):
    """Bind query identity to the exact embedding configuration"""
    query = validate_query(query)

    # Dictionary comparing the query and embedding config
    configuration = {
        "query": query,
        "model_id": MODEL_ID,
        "region": REGION,
        "dimensions": DIMENSIONS,
        "normalize": NORMALIZE,
    }

    # Convert configuration into JSON string
    canonical = json.dumps(
        configuration,
        sort_keys=True,  # Sort dictionary keys alphabetically
        ensure_ascii=False,  # Preserves Unicode characters
    )

    # Convert the JSON into bytes
    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()  # .hexdigest() returns the hash as a 64 char string
    # -> I use this hash for a consistent identifier for the cached embedding


def validate_embedding_response(response):
    """Validate a Titan V2 response before saving or searching"""
    if not isinstance(response, dict):
        raise ValueError("Invalid response format")

    # Extract the embedding and token count
    vector = response.get("embedding")
    tokens = response.get("inputTextTokenCount")

    # Check embedding dimensions
    if not isinstance(vector, list) or len(vector) != DIMENSIONS:
        raise ValueError("Expected 512 embedding dimensions")

    # Verify every element on the embedding
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in vector
    ):
        raise ValueError("Invalid embedding values")

    # Calculate vector magnitude
    norm = math.sqrt(sum(value * value for value in vector))

    # Verify normalization, I allow a tollerance
    if not math.isfinite(norm) or abs(norm - 1.0) > 0.01:
        raise ValueError("Embedding must be normalized")

    # Validate the token count
    if type(tokens) is not int or not 0 < tokens <= 8192:
        raise ValueError("Invalid input token count")

    return {
        "embedding": vector,  # The validated 512 dimensional vector
        "inputTextTokenCount": tokens,  # The amount of tokens
        "vector_norm": norm,  # The calculated vector magnitude
    }


def load_cached_embedding(query, cache_dir):
    """Return a validated cached response, or None if absent"""
    from pathlib import Path

    key = cache_key(query)  # Generate the cache identifier

    # Construct the filename
    cache_dir = Path(cache_dir)
    path = cache_dir / f"{key}.json"

    # Check if the file exists
    if not path.exists():
        return None

    # Read the cached JSON file
    record = json.loads(
        path.read_text(encoding="utf-8")
    )  # Converts the JSON string into a python object

    # Check the cache key
    if record["cache_key"] != key:
        raise ValueError("Cache key mismatch")

    # Verify the query
    if record["query"] != validate_query(query):
        raise ValueError("Cached query mismatch")

    # Validate and return the embedding
    return validate_embedding_response(record["response"])

# Function goal's: preserve valid cached embeddings and reduce the risk of incomplete or conflicting cache files.
def save_cached_embedding(query, response, cache_dir):
    """Save a validated response without overwriting existing data"""
    import os
    from pathlib import Path

    # Validate the query and embedding
    query = validate_query(query)
    validated = validate_embedding_response(response)

    # Prepare the cache directory
    key = cache_key(query)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)  # Creates the cache directory

    # Define the final and temporary filenames
    path = cache_dir / f"{key}.json"  # The final JSON cache file
    temporary = (
        cache_dir / f".{key}.tmp"
    )  # 	A temporary file used during saving (just to be safe if it crashes)

    # Handle an embedding that already exists
    if path.exists():
        existing = load_cached_embedding(query, cache_dir)

        if existing != validated:
            raise ValueError("Conflicting cached embedding")

        return path

    # Prepare the JSON record
    record = {
        "cache_key": key,
        "query": query,
        "response": {
            "embedding": validated["embedding"],
            "inputTextTokenCount": validated["inputTextTokenCount"],
        },
    }

    # Convert the record to JSON
    payload = (
        json.dumps(
            record,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )

    # Create the temporary file
    # Open the file for writing
    # Create the file if it doesn't exist
    # Fail if the file already exists
    # Give the file owner read and write permissions
    fd = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )

    # Publish the completed file
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())

        # Publish exclusively: never replace another writer's entry
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing = load_cached_embedding(query, cache_dir)
            if existing != validated:
                raise ValueError("Conflicting concurrent cache entry")

        # Synchronyze the directory
        dir_fd = os.open(cache_dir, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    # Clean up the temporary file
    finally:
        if temporary.exists():
            temporary.unlink()  # Deleles temporary file

    return path  # Return the saved file location
