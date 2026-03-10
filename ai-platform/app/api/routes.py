from fastapi import APIRouter, HTTPException
from app.schemas.request_models import QueryRequest
from app.services.rag_pipeline import RAGPipeline

router = APIRouter()
rag_pipeline = RAGPipeline()

@router.post("/ask")
async def ask_question(request: QueryRequest):
    """
    Process a user question through the RAG pipeline
    """
    try:
        answer = await rag_pipeline.run_rag(request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
