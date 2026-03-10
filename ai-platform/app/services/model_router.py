import requests
from app.core.config import settings

class ModelRouter:
    """
    Routes queries to Gemini LLM instead of Bedrock
    """
    GEMINI_URL = "https://api.gemini.ai/v1/completions"  # example endpoint

    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY

    def route(self, query: str, docs: list):
        context = "\n".join(docs)
        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "gemini-pro",      # replace with actual Gemini model
            "prompt": prompt,
            "max_tokens": 500,
            "temperature": 0.0
        }

        response = requests.post(self.GEMINI_URL, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()

        # Assuming Gemini returns {"text": "..."}
        return result.get("text", "")