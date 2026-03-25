from fastapi import FastAPI
from src.orchestrator.agent import handle_query
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Processing request")

app = FastAPI()


# Use await here because handle_query is now async
@app.post("/query")
async def query(payload: dict):
    result = await handle_query(payload["query"])
    return {"response": result}
