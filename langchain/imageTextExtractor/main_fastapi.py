import os
import logging
import logging.config
import yaml
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, generate_latest
from starlette.responses import PlainTextResponse
from dotenv import load_dotenv

# Import components from our logic file
from ImageTextExtractor import ingest_data, pipeline_app

load_dotenv()

# Setup Logging - Ensure logs directory exists
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)

log_config_path = os.path.join(os.path.dirname(__file__), 'logging_config.yaml')
with open(log_config_path, 'r') as f:
    log_config = yaml.safe_load(f)

logging.config.dictConfig(log_config)
logger = logging.getLogger('main_fastapi')

app = FastAPI(title="Policy Document RAG Agent API")
logger.info("FastAPI application initialized")

# Metrics
app_request_count_total = Counter(
    'app_request_count_total', 'Total API requests', ['method', 'endpoint']
)
logger.debug("Prometheus metrics initialized")

# GLOBAL INITIALIZATION
# Using absolute path logic to ensure Docker finds the file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "data", "NIST_Securityframework.pdf")
logger.info(f"PDF path configured: {PDF_PATH}")


@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting up application - Initializing RAG Ingestion for: {PDF_PATH}")
    try:
        ingest_data(PDF_PATH)
        logger.info("Ingestion successful. Vector DB ready.")
    except Exception as e:
        logger.error(f"CRITICAL: Ingestion failed: {e}", exc_info=True)
        raise


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
async def process_query(request: QueryRequest):
    logger.info(f"Processing query: {request.question}")
    app_request_count_total.labels(method='POST', endpoint='/query').inc()
    logger.info("Metric incremented")

    try:
        # LangGraph entry point
        inputs = {"question": request.question, "retry_count": 0}
        logger.debug("Invoking LangGraph pipeline")
        final_output = pipeline_app.invoke(inputs)

        logger.info(f"Query processed successfully. Verified: {final_output.get('is_valid')}, Attempts: {final_output.get('retry_count')}")
        return {
            "answer": final_output["answer"],
            "metadata": {
                "verified": final_output.get("is_valid"),
                "attempts": final_output.get("retry_count")
            }
        }
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    logger.debug("Metrics endpoint accessed")
    return PlainTextResponse(generate_latest().decode('utf-8'))


@app.get("/")
async def root():
    logger.debug("Root endpoint accessed")
    return {"status": "online", "engine": "LangGraph + Weaviate"}
