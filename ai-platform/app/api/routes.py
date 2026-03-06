from fastapi import FastAPI, APIRouter
from app.services.rag_pipeline import run_rag

router = APIRouter()

@router.post("/ask")
def ask_question(payload: dict):
    question = payload["question"]
    answer = run_rag(question)
    return {"answer": answer}
