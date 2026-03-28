from datetime import datetime
from src.fraud_rag.weaviate_client import search_docs, store_fraud_event
from src.llm_core.openai_client import call_llm_with_context
from src.ingestion.kafka_producer import send_fraud_event


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
    llm_response = await call_llm_with_context(query, context)

    # Assuming the LLM response indicates fraud detection
    if "fraud detected" in llm_response.lower():  # This is a placeholder for actual fraud detection logic
        fraud_event_data = {
            "query": query,
            "context": [obj.properties for obj in context], # Extract properties from Weaviate objects
            "llm_response": llm_response,
            "timestamp": datetime.now().isoformat()
        }
        # Store in Weaviate
        await store_fraud_event(fraud_event_data)
        # Send to Kafka
        await send_fraud_event(fraud_event_data)

    return llm_response
