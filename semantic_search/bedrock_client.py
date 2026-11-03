
"""Amazon Bedrock Runtime client configuration for semantic search."""

import boto3
from botocore.config import Config

from semantic_search.query_embedding import REGION


def create_bedrock_client():
    """Construct a Bedrock Runtime client without invoking a model."""
    return boto3.Session().client(
        "bedrock-runtime",
        region_name=REGION,
        config=Config(
            retries={
                "mode": "standard",
                "total_max_attempts": 1,
            },
            connect_timeout=10,
            read_timeout=60,
        ),
    )
