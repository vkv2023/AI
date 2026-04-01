import os
import asyncio
import json
from confluent_kafka import Consumer
from src.fraud_rag.weaviate_client import search_docs
import src.configurations as conf

# Configuration from Environment
KAFKA_URL = os.getenv("KAFKA_URL", "localhost:9092")
# KAFKA_URL = (conf.KAFKA_URL, "localhost:9092")  # Override with configuration value if set

conf = {
    'bootstrap.servers': KAFKA_URL,
    'group.id': 'fraud-group',
    'auto.offset.reset': 'earliest',
    'enable.auto.commit': True
}

c = Consumer(conf)
c.subscribe(['fraud'])


async def main_loop():
    print(f"Starting consumer on {KAFKA_URL}...")
    try:
        while True:
            # Poll for 1 second
            msg = c.poll(1.0)

            if msg is None:
                continue
            if msg.error():
                print(f"Consumer error: {msg.error()}")
                continue

            val = msg.value()
            if not val:
                continue

            try:
                data = json.loads(val.decode('utf-8'))
                print(f"Processing transaction: {data.get('event', 'unknown')}")

                # Trigger RAG search in Weaviate
                if 'event' in data:
                    results = await search_docs(data['event'])
                    print(f"RAG matching complete. Found {len(results)} similar cases.")

            except json.JSONDecodeError:
                print(f"Could not decode message: {val}")
            except Exception as e:
                print(f"Error in processing loop: {e}")

    finally:
        # Clean shutdown
        c.close()


if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("Consumer stopped by user.")