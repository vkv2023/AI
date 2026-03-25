import asyncio
import json
from confluent_kafka import Consumer
from src.fraud_rag.weaviate_client import search_docs  # Your async function

c = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'fraud-group',
    'auto.offset.reset': 'earliest'
})

c.subscribe(['fraud'])


# src/ingestion/kafka_consumer.py

async def main_loop():
    try:
        while True:
            msg = c.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print(f"Consumer error: {msg.error()}")
                continue

            # --- THE FIX: Check if value is empty ---
            val = msg.value()
            if not val:
                print("Received an empty message, skipping...")
                continue

            try:
                # Now it's safe to decode
                data = json.loads(val.decode('utf-8'))
                print(f"Processing: {data}")

                # Check if the expected key exists
                if 'event' in data:
                    results = await search_docs(data['event'])
                    print(f"RAG Results found.")
                else:
                    print("JSON received but 'event' key is missing.")

            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON: {val}. Error: {e}")

    finally:
        c.close()

if __name__ == "__main__":
    # 3. Use the asyncio event loop to start the function
    asyncio.run(main_loop())