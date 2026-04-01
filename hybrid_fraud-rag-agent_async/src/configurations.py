import os
from dotenv import load_dotenv
import logging

# =========================
# LOAD ENV VARIABLES
# =========================
load_dotenv()


def _int_env(name: str, default: int) -> int:
    """Safely parse an environment variable as int, returning default on None/empty/invalid."""
    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


# Get environment variables with fallbacks
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = _int_env("REDIS_PORT", 6379)
CACHE_TTL = _int_env("CACHE_TTL", 3600)

# Optional OpenAI key — do not fail application startup when missing (useful for local testing)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_ENABLED = bool(OPENAI_API_KEY)

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
S3_BUCKET = os.getenv("S3_BUCKET")
local_path = os.getenv("LOCAL_PATH")
COHERE_KEY = os.getenv("COHERE_KEY")
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST")
WEAVIATE_PORT = _int_env("WEAVIATE_PORT", 8080)
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
WEAVIATE_GRPC_PORT = _int_env("WEAVIATE_GRPC_PORT", 50051)
KAFKA_URL = os.getenv("KAFKA_URL", "localhost:9092")


# Warn if OpenAI is disabled so it's obvious in the logs
if not OPENAI_ENABLED:
    logging.getLogger(__name__).warning(
        "OPENAI_API_KEY not set — OpenAI features will be disabled. Set OPENAI_API_KEY in the environment for full functionality."
    )
