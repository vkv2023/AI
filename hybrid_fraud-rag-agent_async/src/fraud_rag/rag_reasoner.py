from src.fraud_rag.weaviate_client import search_docs
from src.llm_core.openai_client import call_llm_with_context


# 1. Change to 'async def'
async def get_rag_response(query):
    # 2. Await the Weaviate search
    context = await search_docs(query)

    # 3. Use a structured prompt for the LLM
    # (Optional: Pass the raw list of docs if your call_llm_with_context handles formatting)
    prompt = f"""
    Use the context below to answer:

    {context}

    Question: {query}
    """

    # 4. Await the OpenAI response
    return await call_llm_with_context(query, context)