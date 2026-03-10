from app.database.weaviate_client import WeaviateClient
from app.database.redis_cache import RedisCache

class HybridSearch:
    def __init__(self):
        self.vector_db = WeaviateClient()
        self.cache = RedisCache()

    def query(self, query_text: str, top_k: int = 5):
        # Check Redis cache first
        cached = self.cache.get(query_text)
        if cached:
            return cached

        # Vector search
        vector_results = self.vector_db.search(query_text, top_k=top_k)

        # TODO: BM25 or other hybrid scoring can be added here
        self.cache.set(query_text, vector_results)
        return vector_results