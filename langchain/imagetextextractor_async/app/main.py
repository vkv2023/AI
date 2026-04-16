import os
import logging.config
import yaml
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, generate_latest, REGISTRY  # Added REGISTRY here
from starlette.responses import PlainTextResponse
from dotenv import load_dotenv

# --- OpenTelemetry Imports ---
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor

# Import components from your logic file
from src.rag.imagetextextractor_async import ingest_data
from src.rag.pipeline import pipeline_app

load_dotenv()

# 1. SETUP OPENTELEMETRY (Must happen before FastAPI init)
resource = Resource.create({
    "service.name": os.getenv("OTEL_SERVICE_NAME", "imageTextExt-agent")
})

provider = TracerProvider(resource=resource)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

# Use ConfigMap/Env value, fallback to Jaeger service

# jaeger_exporter = OTLPSpanExporter(
#     endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://jaeger:4317"),
#     insecure=True
# )

jaeger_exporter = OTLPSpanExporter(
    endpoint="http://jaeger:4318/v1/traces"
)

provider.add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

# Inject trace IDs into logs
LoggingInstrumentor().instrument(set_logging_format=True)

# 2. SETUP LOGGING

log_dir = os.getenv("LOG_DIR", "/app/logs")
os.makedirs(log_dir, exist_ok=True)

print(f"Logs will be written to: {log_dir}")

# Docker copies logging_config.yaml to /app/

log_config_path = "/app/logging_config.yaml"

with open(log_config_path, "r") as f:
    log_config = yaml.safe_load(f)

logging.config.dictConfig(log_config)
logger = logging.getLogger("main_fastapi")

# 3. INITIALIZE APP
app = FastAPI(title="Policy Document RAG Agent API")
logger.info("FastAPI application initialized")

# Instrument FastAPI for automatic route tracing
FastAPIInstrumentor.instrument_app(app)

# 4. METRICS
app_request_count_total = Counter(
    'app_request_count_total', 'Total API requests', ['method', 'endpoint']
)
# Pre-initialize for visibility in Prometheus
app_request_count_total.labels(method='POST', endpoint='/query')

# GLOBAL INITIALIZATION
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# PDF_PATH = os.path.join(BASE_DIR, "data", "NIST_Securityframework.pdf")

# Go up two levels from main.py to reach /app/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH = os.path.join(BASE_DIR, "data", "NIST_Securityframework.pdf")

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
    # This creates a manual span in Jaeger for the logic block
    with tracer.start_as_current_span("process_rag_query") as span:
        logger.info(f"Processing query: {request.question}")
        span.set_attribute("user.question", request.question)  # See the question in Jaeger

        app_request_count_total.labels(method='POST', endpoint='/query').inc()

        try:
            inputs = {"question": request.question, "retry_count": 0}
            final_output = pipeline_app.invoke(inputs)

            span.set_attribute("rag.verified", final_output.get("is_valid"))
            logger.info(f"Query processed. Verified: {final_output.get('is_valid')}")

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
    return PlainTextResponse(generate_latest(REGISTRY).decode('utf-8'))


@app.get("/")
async def root():
    return {"status": "online", "engine": "LangGraph + Weaviate"}
