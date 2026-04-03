from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
import os
from dotenv import load_dotenv
from prometheus_client import Counter, generate_latest
from starlette.responses import PlainTextResponse

# Import the necessary components from your existing pipeline file
from ImageTextExtractor import ingest_data, workflow, AgentState

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Prometheus Counter for API requests
app_request_count_total = Counter(
    'app_request_count_total',
    'Total number of API requests',
    ['method', 'endpoint']
)

# Initialize the LangChain pipeline components globally
# This ensures they are loaded once when the FastAPI app starts
retriever = ingest_data("../data/NIST_Cybersecurity_Framework.pdf")
pipeline_app = workflow.compile()


# Define a request model for the incoming query
class QueryRequest(BaseModel):
    question: str


@app.post("/query")
async def process_query(request: QueryRequest):
    """
    Receives a natural language query, processes it through the RAG pipeline,
    and returns the generated answer.
    """
    app_request_count_total.labels(method='POST', endpoint='/query').inc()
    inputs = {"question": request.question, "retry_count": 0}
    final_output = pipeline_app.invoke(inputs)
    return {"answer": final_output["answer"]}


@app.get("/")
async def root():
    return {"message": "Welcome to the Image Text Extractor API. Use the /query endpoint to process questions."}


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest().decode('utf-8'))


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
