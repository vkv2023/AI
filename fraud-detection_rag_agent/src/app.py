from fastapi import FastAPI
from fraud_agent import retrieve_similar_cases
from rag_reasoner import analyze_risk
from redis_cache import get_cache, set_cache

app = FastAPI()


@app.post("/check_transaction")
def check_transaction(data: dict):
    description = data["description"]

    cached = get_cache(description)

    if cached:
        return {"cached": True, "result": cached}

    cases = retrieve_similar_cases(description)

    result = analyze_risk(description, cases)

    set_cache(description, result)

    return result
