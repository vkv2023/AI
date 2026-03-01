import os
import uuid
import boto3
import weaviate
from dotenv import load_dotenv
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# =========================
# LOAD ENV VARIABLES
# =========================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
S3_BUCKET = os.getenv("S3_BUCKET")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set")

# =========================
# INIT CLIENTS
# =========================
openai_client = OpenAI(api_key=OPENAI_API_KEY)
weaviate_client = weaviate.Client(WEAVIATE_URL)
s3 = boto3.client("s3")

CLASS_NAME = "EnterpriseDocument"

# =========================
# CREATE SCHEMA IF NOT EXISTS
# =========================
if not weaviate_client.schema.exists(CLASS_NAME):
    schema = {
        "class": CLASS_NAME,
        "vectorizer": "none",  # we provide embedding manually
        "properties": [
            {"name": "content", "dataType": ["text"]},
            {"name": "source", "dataType": ["text"]},
            {"name": "department", "dataType": ["text"]}
        ]
    }
    weaviate_client.schema.create_class(schema)
    print("Schema created")


# =========================
# STEP 1 — DOWNLOAD FROM S3
# =========================
def download_from_s3(key, local_path):
    s3.download_file(S3_BUCKET, key, local_path)
    print(f"Downloaded {key}")


# =========================
# STEP 2 — EXTRACT TEXT
# =========================
def extract_text(file_path):
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()


# =========================
# STEP 3 — CHUNKING
# =========================
def chunk_text(text, chunk_size=800, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


# =========================
# STEP 4 — EMBEDDING
# =========================
def embed_chunks(chunks):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks
    )
    return [d.embedding for d in response.data]


# =========================
# STEP 5 — STORE IN WEAVIATE
# =========================
def store_chunks(chunks, embeddings, source, department="general"):
    with weaviate_client.batch as batch:
        for chunk, vector in zip(chunks, embeddings):
            batch.add_data_object(
                data_object={
                    "content": chunk,
                    "source": source,
                    "department": department
                },
                class_name=CLASS_NAME,
                uuid=str(uuid.uuid4()),
                vector=vector
            )
    print("Chunks stored in Weaviate")


# =========================
# INGEST PIPELINE
# =========================
def ingest_document(s3_key, department="general"):
    local_file = "temp_document"
    download_from_s3(s3_key, local_file)

    text = extract_text(local_file)
    chunks = chunk_text(text)

    print(f"Total chunks: {len(chunks)}")

    embeddings = embed_chunks(chunks)
    store_chunks(chunks, embeddings, s3_key, department)


# =========================
# STEP 6 — HYBRID SEARCH
# =========================
def search(question, department_filter=None):
    query = (
        weaviate_client.query
        .get(CLASS_NAME, ["content", "source"])
        .with_hybrid(query=question, alpha=0.5)
        .with_limit(5)
    )

    if department_filter:
        query = query.with_where({
            "path": ["department"],
            "operator": "Equal",
            "valueText": department_filter
        })

    result = query.do()
    return result["data"]["Get"][CLASS_NAME]


# =========================
# STEP 7 — LLM ANSWER
# =========================
def generate_answer(question, retrieved_docs):
    context = "\n\n".join([doc["content"] for doc in retrieved_docs])

    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer strictly using the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )

    return completion.choices[0].message.content


# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":

    # ---------- INGEST ----------
    # Example S3 file
    # ingest_document("policies/finance_policy.pdf", department="finance")

    # ---------- QUERY ----------
    question = "What is the finance approval process?"
    retrieved = search(question, department_filter="finance")

    print("\nRetrieved Docs:")
    for r in retrieved:
        print("-", r["source"])

    answer = generate_answer(question, retrieved)

    print("\nAnswer:")
    print(answer)
