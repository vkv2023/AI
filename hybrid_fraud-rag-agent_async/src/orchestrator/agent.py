# src/orchestrator/agent.py
from src.fraud_rag.rag_reasoner import get_rag_response
from src.services_detect.fraud_api import get_transaction_data
from src.caching.redis_cache import get_cache, set_cache


def detect_intent(query):
    # Quick intent routing logic
    if any(keyword in query.lower() for keyword in ["transaction", "payment", "id"]):
        return "API"
    return "RAG"


async def handle_query(query):
    # 1. AWAIT the cache check
    # If get_cache is 'async def', you MUST await it here
    cached = await get_cache(query)
    if cached:
        return {"source": "cache", "response": cached}

    intent = detect_intent(query)

    # 2. Route the request
    if intent == "RAG":
        # Ensure get_rag_response is 'async def'
        response = await get_rag_response(query)
    else:
        # Ensure get_transaction_data is 'async def'
        data = await get_transaction_data(query)
        response = f"Transaction Info: {data}"

    # 3. AWAIT the cache set
    # This ensures the string is written before the function returns
    await set_cache(query, response)

    return {"source": intent, "response": response}