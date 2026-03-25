from aiokafka import AIOKafkaProducer
import json
import asyncio


async def send_fraud_event(event_data):
    producer = AIOKafkaProducer(bootstrap_servers='localhost:9092')
    await producer.start()
    try:
        # High-speed async send
        await producer.send_and_wait("fraud", json.dumps(event_data).encode('utf-8'))
        print("Event sent to Kafka")
    finally:
        await producer.stop()
