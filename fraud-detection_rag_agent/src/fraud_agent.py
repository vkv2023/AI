from embeddings import generate_embedding
from weaviate_client import client


def retrieve_similar_cases(transaction):
    vector = generate_embedding(transaction)

    result = client.query.get(
        "FraudCase",
        ["description", "label"]
    ).with_near_vector({
        "vector": vector
    }).with_limit(3).do()

    return result["data"]["Get"]["FraudCase"]
