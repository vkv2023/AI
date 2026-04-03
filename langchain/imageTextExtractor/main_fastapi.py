import os
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, generate_latest
from starlette.responses import PlainTextResponse
from dotenv import load_dotenv

# Import components from our logic file
from ImageTextExtractor import ingest_data, pipeline_app

load_dotenv()

app = FastAPI(title="Policy Document RAG Agent API")

# Metrics
app_request_count_total = Counter(
    'app_request_count_total', 'Total API requests', ['method', 'endpoint']
)

# GLOBAL INITIALIZATION
# Using absolute path logic to ensure Docker finds the file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "..", "data", "NIST_Cybersecurity_Framework.pdf")


@app.on_event("startup")
async def startup_event():
    print(f"Initializing RAG Ingestion for: {PDF_PATH}")
    try:
        ingest_data(PDF_PATH)
        print("Ingestion successful. Vector DB ready.")
    except Exception as e:
        print(f"CRITICAL: Ingestion failed: {e}")


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
async def process_query(request: QueryRequest):
    app_request_count_total.labels(method='POST', endpoint='/query').inc()

    try:
        # LangGraph entry point
        inputs = {"question": request.question, "retry_count": 0}
        final_output = pipeline_app.invoke(inputs)

        return {
            "answer": final_output["answer"],
            "metadata": {
                "verified": final_output.get("is_valid"),
                "attempts": final_output.get("retry_count")
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest().decode('utf-8'))


@app.get("/")
async def root():
    return {"status": "online", "engine": "LangGraph + Weaviate"}


# if __name__ == "__main__":
#     import uvicorn

    # uvicorn.run(app, host="0.0.0.0", port=8000)