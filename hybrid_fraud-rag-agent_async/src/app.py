from fastapi import FastAPI
from orchestrator.agent import handle_query

app = FastAPI()

@app.post("/query")
def query(payload: dict):
    return handle_query(payload["query"])