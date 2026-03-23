# embeddings.py
from app.core.config import settings
import boto3


class Embeddings:
    def __init__(self):
        self.client = boto3.client("bedrock-runtime")

    def embed_text(self, text: str):
        response = self.client.invoke_model(
            modelId="amazon.titan-embed-text-v1",
            body=text
        )
        return response
