import json
from embeddings import generate_embedding
from weaviate_client import client, create_schema

create_schema()

with open("../data/fraud_cases.json") as f:
    fraud_cases = json.load(f)

for case in fraud_cases:

    vector = generate_embedding(case["description"])

    client.data_object.create(
        data_object={
            "description": case["description"],
            "label": case["label"]
        },
        class_name="FraudCase",
        vector=vector
    )

print("Fraud cases indexed")