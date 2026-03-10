import weaviate
from app.core.config import settings

class WeaviateClient:
    def __init__(self):
        self.client = weaviate.Client(settings.VECTOR_DB_URL)

    def search(self, query: str, top_k: int = 5):
        # Placeholder example
        return ["Doc1 snippet", "Doc2 snippet", "Doc3 snippet"]