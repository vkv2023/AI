import cohere
from app.core.config import settings

class CohereReranker:
    def __init__(self):
        self.client = cohere.Client(settings.COHERE_API_KEY)

    def rerank(self, query: str, docs: list):
        # TODO: implement actual cohere reranking
        return docs  # placeholder