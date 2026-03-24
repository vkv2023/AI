import json
import time
from src.fraud_rag.weaviate_client import client
from src.fraud_rag.embeddings import get_embedding

CLASS_NAME = "FraudEvent"


# -----------------------------
# 1. Ensure Schema Exists
# -----------------------------
def create_schema():
    existing = client.schema.get()

    classes = [c["class"] for c in existing.get("classes", [])]

    if CLASS_NAME in classes:
        print("Schema already exists")
        return

    schema = {
        "class": CLASS_NAME,
        "description": "Transaction data for fraud analysis",
        "vectorizer": "none",  # we provide embeddings manually
        "properties": [
            {
                "name": "content",
                "dataType": ["text"],
                "description": "The raw transaction or logs text",
            },
            {
                "name": "metadata",
                "dataType": ["text"],
                "description": "Additional JSON metadata about the fraud case",
            }
        ]
    }

    client.schema.create_class(schema)
    print("Schema created")


# -----------------------------
# 2. Load Data
# -----------------------------
def load_data(file_path="data/fraud_cases.json"):
    with open(file_path, "r") as f:
        return json.load(f)


# -----------------------------
# 3. Ingest Data (Batch)
# -----------------------------
def ingest_data(data):
    print(f"Ingesting {len(data)} records...")

    with client.batch as batch:
        batch.batch_size = 10

        for item in data:
            content = item.get("content", "")

            # Generate embedding
            vector = get_embedding(content)

            obj = {
                "content": content,
                "metadata": str(time.time())
            }

            batch.add_data_object(
                obj,
                CLASS_NAME,
                vector=vector
            )

    print("Ingestion complete")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    create_schema()

    data = load_data()
    ingest_data(data)