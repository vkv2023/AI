import requests
from app.core.config import settings

class QueryRewriter:
    """
    Rewrites or paraphrases user queries to improve retrieval quality.
    Uses Gemini API to rewrite the query.
    """
    GEMINI_URL = "https://api.gemini.ai/v1/completions"

    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY

    def rewrite(self, query: str) -> str:
        """
        Returns a rewritten version of the query.
        """
        prompt = f"Rewrite the following user query to be clear and precise for document retrieval:\n\nQuery: {query}\nRewritten Query:"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "gemini-pro",  # Replace with the actual Gemini model you want
            "prompt": prompt,
            "max_tokens": 100,
            "temperature": 0.0
        }

        response = requests.post(self.GEMINI_URL, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()

        # Gemini API typically returns {"text": "..."}
        rewritten_query = result.get("text", "").strip()
        return rewritten_query if rewritten_query else query