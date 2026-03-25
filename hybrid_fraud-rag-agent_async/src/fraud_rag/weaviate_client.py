import os
import weaviate
from weaviate.classes.init import Auth
from dotenv import load_dotenv
from src.fraud_rag.embeddings import get_embedding

load_dotenv()

# ---------- CONFIG ----------
HOST = os.getenv("WEAVIATE_HOST", "localhost")
PORT = int(os.getenv("WEAVIATE_PORT", 8080))
GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", 50051))
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")  # needed for v4 as it uses GRPC, not needed for docker
# For v4, we use use_async_with_local or connect_to_weaviate_cloud
# This example assumes a standard URL connection
client = weaviate.use_async_with_local(
    host=HOST,
    port=PORT,
    grpc_port=GRPC_PORT
    # ,
    # auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")), # Optional: if using WCS
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
            alpha=0.3,  # 0.5 balances keywords and vector meaning
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
