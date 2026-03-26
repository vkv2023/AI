import os
from dotenv import load_dotenv

# =========================
# LOAD ENV VARIABLES
# =========================
load_dotenv()


# Get environment variables with fallbacks
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
CACHE_TTL = int(os.getenv("CACHE_TTL"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
S3_BUCKET = os.getenv("S3_BUCKET")
local_path = os.getenv("LOCAL_PATH")
COHERE_KEY = os.getenv("COHERE_KEY")
WEAVIATE_HOST=os.getenv("WEAVIATE_HOST")
WEAVIATE_PORT=int(os.getenv("WEAVIATE_PORT"))
WEAVIATE_API_KEY=os.getenv("WEAVIATE_API_KEY")
WEAVIATE_GRPC_PORT=int(os.getenv("WEAVIATE_GRPC_PORT"))


if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set")
