import os
import weaviate
from dotenv import load_dotenv
from openai import OpenAI
# Add Chunking Logic using, uncomment the below line
# from langchain.text_splitter import RecursiveCharacterTextSplitter


# ---------- CONFIG ----------
load_dotenv(dotenv_path=".env")

# ---------- INIT CLIENTS ----------
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = weaviate.Client(WEAVIATE_URL)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

CLASS_NAME = "Document"

# ---------- CREATE SCHEMA (if not exists) ----------
if not client.schema.exists(CLASS_NAME):
    schema = {
        "class": CLASS_NAME,
        "vectorizer": "text2vec-openai",
        "moduleConfig": {
            "text2vec-openai": {
                "model": "text-embedding-3-small"
            }
        },
        "properties": [
            {"name": "content", "dataType": ["text"]},
            {"name": "source", "dataType": ["text"]} # Add Metadata Filtering
        ]
    }
    client.schema.create_class(schema)

# ---------- INGEST SAMPLE DATA ----------
docs = [
    {"content": "Weaviate is used for vector similarity search."},
    {"content": "Spring Boot is popular for Java microservices."},
    {"content": "Java is a popular backend programming language."}
]

for doc in docs:
    client.data_object.create(doc, CLASS_NAME)

print("Documents inserted.")

# ---------- ASK QUESTION ----------
question = "What is Weaviate used for?"

result = (
    client.query
    .get(CLASS_NAME, ["content"])
    .with_near_text({"concepts": [question]})
    .with_hybrid(query="similarity search", alpha=0.5) #Add Hybrid Search (Vector + BM25)
    .with_limit(2)
    .do()
)

retrieved_docs = result["data"]["Get"][CLASS_NAME]
context = "\n".join([doc["content"] for doc in retrieved_docs])

completion = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Answer only using the provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]
)

print("\nAnswer:")
print(completion.choices[0].message.content)