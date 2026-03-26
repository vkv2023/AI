import asyncio
from weaviate.classes.query import Filter
from src.fraud_rag.weaviate_client import client


async def delete_all_records():
    try:
        print("Connecting to Weaviate...")
        await client.connect()

        # Get the specific collection
        fraud_coll = client.collections.get("FraudEvent")

        print("Starting deletion of all records in 'FraudEvent'...")

        # Use a 'like' filter with a wildcard to match everything
        # This clears the data but keeps your Property definitions intact
        response = await fraud_coll.data.delete_many(
            where=Filter.by_property("content").like("*"),
            verbose=True
        )

        print(f"--- Deletion Summary ---")
        print(f"Matched objects: {response.matches}")
        print(f"Successfully deleted: {response.successful}")
        print(f"Failed deletions: {response.failed}")

        if response.failed > 0:
            for err in response.objects:
                if not err.successful:
                    print(f"Error: {err.error}")

    except Exception as e:
        print(f"Critical Error during deletion: {e}")
    finally:
        await client.close()
        print("Connection closed.")


if __name__ == "__main__":
    asyncio.run(delete_all_records())