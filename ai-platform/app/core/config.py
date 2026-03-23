import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    VECTOR_DB_URL = os.getenv("VECTOR_DB_URL", "http://localhost:8080")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")


settings = Settings()
