import weaviate
from config import WEAVIATE_URL

client = weaviate.Client(WEAVIATE_URL)

def search_docs(query):
    result = (
        client.query
        .get("Document", ["content"])
        .with_near_text({"concepts": [query]})
        .with_limit(2)
        .do()
    )

    docs = result["data"]["Get"]["Document"]
    return " ".join([d["content"] for d in docs])