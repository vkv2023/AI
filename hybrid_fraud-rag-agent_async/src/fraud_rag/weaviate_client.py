import os
import weaviate
from dotenv import load_dotenv
from src.fraud_rag.embeddings import get_embedding
import src.configurations as conf

# WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")  # needed for v4 as it uses GRPC, not needed for docker
# For v4, we use use_async_with_local or connect_to_weaviate_cloud
client = weaviate.use_async_with_local(
    host=conf.WEAVIATE_HOST,
    port=conf.WEAVIATE_PORT,
    grpc_port=conf.WEAVIATE_GRPC_PORT
    # ,
)


async def search_docs(query_text):
    # 1. Ensure client is connected
    # (Note: with use_async_with_local, it's better to use 'async with'
    # inside the function or a startup script, but this check works too)
    if not client.is_connected():
        await client.connect()


    try:
        # 2. Get the vector for the user's query (Ensure get_embedding is awaited!)
        query_vector = await get_embedding(query_text)

        # 3. Access the collection
        fraud_events = client.collections.get("FraudEvent")

        # Change near_vector to hybrid
        response = await fraud_events.query.hybrid(
            query=query_text,  # Keyword search
            vector=query_vector,  # Vector search
            limit=5,  # Increase limit slightly
            alpha=0.3,  # 0.5 balances keywords and vector meaning, 0.3 more keyword search. removes hallucination
            # ADD THIS LINE:
            return_properties=["content"]
        )

        # 5. Check for results
        if not response.objects:
            print("No relevant fraud cases found.")
            return []  # Return empty list so the LLM client doesn't crash on a string

        # IMPORTANT: Return the list of objects directly.
        # This allows openai_client.py to access 'doc.properties["content"]'
        return response.objects

    except Exception as e:
        print(f"Weaviate Search Error: {e}")
        return []  # Return empty list on error to maintain data type consistency


async def store_fraud_event(event_data: dict):
    try:
        if not client.is_connected():
            await client.connect()

        fraud_events = client.collections.get("FraudEvent")
        response = await fraud_events.data.insert(
            properties=event_data
        )
        print(f"Fraud event stored in Weaviate with ID: {response.uuid}")
        return response.uuid
    except Exception as e:
        print(f"Error storing fraud event in Weaviate: {e}")
        return None

