import os
from dotenv import load_dotenv

# =========================
# LOAD ENV VARIABLES
# =========================
load_dotenv()


class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    WEAVIATE_URL = os.getenv("WEAVIATE_URL")
    S3_BUCKET = os.getenv("S3_BUCKET")
    local_path = os.getenv("LOCAL_PATH")
    key = os.getenv("KEY")
    COHERE_KEY = os.getenv("COHERE_KEY")
    REDIS_HOST = os.getenv("REDIS_HOST")
    REDIS_PORT = int(os.getenv("REDIS_PORT"))
    CACHE_TTL = int(os.getenv("CACHE_TTL"))

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")


settings = Settings()
