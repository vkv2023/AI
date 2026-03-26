import asyncio
from src.fraud_rag.weaviate_client import client


async def check_data():
    try:
        print("Connecting to Weaviate...")
        await client.connect()

        # Access the collection
        fraud_coll = client.collections.get("FraudEvent")

        # 1. Check Total Count
        # This confirms if all 7 objects are physically in the DB
        aggregation = await fraud_coll.aggregate.over_all(total_count=True)
        count = aggregation.total_count
        print(f"\n--- Verification Result ---")
        print(f"Total objects in Weaviate: {count}")

        # 2. Fetch Sample Data
        # This confirms the properties (content, label, etc.) were mapped correctly
        response = await fraud_coll.query.fetch_objects(limit=2)
        print("\nSample Records:")
        for obj in response.objects:
            print(f"- [{obj.properties.get('label')}] {obj.properties.get('content')}")

    except Exception as e:
        print(f"Verification Error: {e}")
    finally:
        await client.close()
        print("\nConnection closed.")


if __name__ == "__main__":
    asyncio.run(check_data())