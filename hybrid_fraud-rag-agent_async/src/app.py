import os
import sys
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

# Ensure the root directory is in the path so 'src' is discoverable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.orchestrator.agent import handle_query

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# 1. Lifespan Manager: Cleanly start and stop connections
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Hybrid Fraud-RAG API...")
    # You can initialize shared clients here (Redis, Kafka Producer, etc.)
    yield
    logger.info("Shutting down API and closing connections...")


app = FastAPI(lifespan=lifespan)


@app.post("/query")
async def query(payload: dict):
    # Basic validation to prevent KeyErrors
    user_query = payload.get("query")
    if not user_query:
        logger.warning("Empty query received")
        raise HTTPException(status_code=400, detail="Missing 'query' in payload")

    try:
        logger.info(f"Processing query: {user_query[:50]}...")
        # handle_query will now use your RAG + Kafka logic
        result = await handle_query(user_query)
        return {"response": result}

    except Exception as e:
        logger.error(f"Error in orchestrator: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal processing error")


# Health check for Docker/Monitoring
@app.get("/health")
async def health():
    return {"status": "healthy"}