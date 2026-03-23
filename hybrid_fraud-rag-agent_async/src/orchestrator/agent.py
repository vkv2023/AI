from rag.rag_reasoner import get_rag_response
from services.fraud_api import get_transaction_data
from cache.redis_cache import get_cache, set_cache

def detect_intent(query):
    if "transaction" in query or "payment" in query:
        return "API"
    return "RAG"

def handle_query(query):
    cached = get_cache(query)
    if cached:
        return {"source": "cache", "response": cached}

    intent = detect_intent(query)

    if intent == "RAG":
        response = get_rag_response(query)
    else:
        data = get_transaction_data(query)
        response = f"Transaction Info: {data}"

    set_cache(query, response)
    return {"source": intent, "response": response}