import asyncio
import time
import json
import os
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.data import DataObject

# Ensure these imports point to your actual local file structure
from src.fraud_rag.weaviate_client import client
from src.fraud_rag.embeddings import get_embedding

CLASS_NAME = "FraudEvent"


async def create_schema():
    """Wipes and recreates the collection using v4 standards."""
    if not client.is_connected():
        await client.connect()

    # 1. Cleanup: delete existing collection to ensure a fresh schema
    if await client.collections.exists(CLASS_NAME):
        print(f"Deleting existing collection: {CLASS_NAME}...")
        await client.collections.delete(CLASS_NAME)

    # 2. Re-create using the correct 'none' vectorizer setting
    print(f"Creating collection: {CLASS_NAME}...")
    await client.collections.create(
        name=CLASS_NAME,
        # 'Vectorizer.none()' tells Weaviate you are providing vectors manually
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="content", data_type=DataType.TEXT),
            Property(name="metadata", data_type=DataType.TEXT),
        ]
    )
    print("✅ Schema created successfully.")


async def ingest_data(file_path="data/fraud_cases.json"):
    """Loads JSON and performs an async insert_many."""
    if not os.path.exists(file_path):
        print(f"❌ Error: Data file not found at {file_path}")
        return

    with open(file_path, "r") as f:
        data = json.load(f)

    print(f"Processing {len(data)} records for ingestion...")

    fraud_coll = client.collections.get(CLASS_NAME)
    objects_to_insert = []

    for item in data:
        content = item.get("content", "")

        # Await your async embedding function
        vector = await get_embedding(content)

        # Prepare the object with its vector
        objects_to_insert.append(
            DataObject(
                properties={
                    "content": content,
                    "metadata": str(time.time())
                },
                vector=vector
            )
        )

    # Use the async-native insert_many method
    print(f"Uploading batch to Weaviate...")
    response = await fraud_coll.data.insert_many(objects_to_insert)

    if response.has_errors:
        print("❌ Ingestion errors occurred:")
        for error in response.errors:
            print(f"- {error}")
    else:
        print(f"✅ Successfully ingested {len(objects_to_insert)} objects.")


async def main():
    try:
        if not client.is_connected():
            await client.connect()

        await create_schema()
        await ingest_data()

    except Exception as e:
        print(f"❌ Critical Error: {e}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())