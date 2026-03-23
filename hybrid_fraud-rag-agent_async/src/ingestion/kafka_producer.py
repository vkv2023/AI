from confluent_kafka import Producer
import json

p = Producer({'bootstrap.servers': 'localhost:9092'})

p.produce('fraud', json.dumps({"event": "fraud_case"}))
p.flush()