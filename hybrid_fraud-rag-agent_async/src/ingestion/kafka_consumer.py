from confluent_kafka import Consumer
import json

c = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'grp',
    'auto.offset.reset': 'earliest'
})

c.subscribe(['fraud'])

while True:
    msg = c.poll(1.0)
    if msg:
        print("Received:", msg.value().decode())