import weaviate
import os
from dotenv import load_dotenv
from embeddings import get_embedding
load_dotenv()
# ---------- CONFIG ----------
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
print(WEAVIATE_URL)
client = weaviate.Client(url=WEAVIATE_URL)


def search_docs(query_text):
    # 1. Get the vector for the user's query
    query_vector = get_embedding(query_text)

    # 2. Execute the search
    response = (
        client.query
        .get("FraudEvent", ["content", "metadata"])
        .with_near_vector({"vector": query_vector})
        .with_limit(3)
        .do()
    )

    # 3. Check for Weaviate errors (This prevents the KeyError: 'data')
    if "errors" in response:
        error_msg = response["errors"][0]["message"]
        print(f"Weaviate Error: {error_msg}")
        return f"Error from database: {error_msg}"

    # 4. Safely extract results
    try:
        data = response.get("data", {}).get("Get", {}).get("FraudEvent", [])

        if not data:
            print("⚠️ No relevant fraud cases found.")
            return "No relevant context found."

        # Combine results into a single string for the LLM
        context = "\n---\n".join([item["content"] for item in data])
        return context

    except Exception as e:
        print(f"⚠️ Parsing Error: {e}")
        return "Failed to parse search results."