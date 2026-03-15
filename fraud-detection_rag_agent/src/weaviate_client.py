import weaviate
# from weaviate.connect import ConnectionParams

client = weaviate.Client("http://localhost:8080")

client = weaviate.WeaviateClient(
    connection_params=Connection

def create_schema():
    schema = {
        "class": "FraudCase",
        "vectorizer": "none",
        "properties": [
            {"name": "description", "dataType": ["text"]},
            {"name": "label", "dataType": ["text"]}
        ]
    }

    if not client.schema.contains(schema):
        client.schema.create_class(schema)
