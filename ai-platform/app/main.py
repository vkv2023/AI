from fastapi import FastAPI
from app.api.routes import router
from app.core.config import settings

app = FastAPI(title="AI Platform")

# Include API routes
app.include_router(router)

@app.get("/health")
async def health_check():
    return {"status": "ok", "app": "ai-platform"}