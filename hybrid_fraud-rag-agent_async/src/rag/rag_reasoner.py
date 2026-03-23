from rag.weaviate_client import search_docs
from llm.openai_client import call_llm

def get_rag_response(query):
    context = search_docs(query)

    prompt = f"""
    Use the context below to answer:

    {context}

    Question: {query}
    """

    return call_llm(prompt)