import os
import json
import asyncio
from aiokafka import AIOKafkaProducer
import src.configurations as conf


async def send_fraud_event(event_data):
    # 'kafka:29092' inside Docker, 'localhost:9092' on Windows
    KAFKA_URL = os.getenv("KAFKA_URL", "localhost:9092")
    # KAFKA_URL = (conf.KAFKA_URL, "localhost:9092")  # Override with configuration value if set

    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_URL,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    await producer.start()
    try:
        # High-speed async send
        await producer.send_and_wait("fraud", event_data)
        print(f"Successfully sent event to Kafka at {kafka_url}")
    except Exception as e:
        print(f"Failed to send to Kafka: {e}")
    finally:
        await producer.stop()