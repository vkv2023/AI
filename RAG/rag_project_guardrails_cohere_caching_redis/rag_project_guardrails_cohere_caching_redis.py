import uuid
import boto3
import cohere
import os
import weaviate
from dotenv import load_dotenv
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import redis


# =========================
# LOAD ENV VARIABLES
# =========================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
S3_BUCKET = os.getenv("S3_BUCKET")
local_path = os.getenv("LOCAL_PATH")
key = os.getenv("KEY")
COHERE_KEY = os.getenv("COHERE_KEY")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
CACHE_TTL = int(os.getenv("CACHE_TTL"))

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set")

if not COHERE_KEY:
    raise ValueError("COHERE_API_KEY is not set")

# =========================
# INIT CLIENTS
# =========================
openai_client = OpenAI(api_key=OPENAI_API_KEY)
weaviate_client = weaviate.Client(WEAVIATE_URL)
cohere_client = cohere.Client(COHERE_KEY)
CLASS_NAME = "JavaAwsDocuments"
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True
)

def get_cached_answer(question, department):
    key = f"{department}:{question}"
    return redis_client.get(key)


def cache_answer(question, department, answer):
    key = f"{department}:{question}"
    redis_client.set(key, answer, ex=CACHE_TTL)


# =========================
# ===== GUARDRAIL =====
# INPUT GUARDRAIL
# =========================
def input_guardrail(question):

    banned_patterns = [
        "ignore previous instructions",
        "reveal secrets",
        "api key",
        "password",
        "drop table",
        "delete database",
        "system prompt"
    ]

    for pattern in banned_patterns:
        if pattern in question.lower():
            return False, "Request blocked by security guardrail."

    return True, question


# =========================
# ===== GUARDRAIL =====
# RETRIEVAL GUARDRAIL
# =========================
def retrieval_guardrail(docs):

    filtered_docs = []

    banned_terms = [
        "ssn",
        "credit card",
        "secret",
        "password"
    ]

    for doc in docs:
        text = doc["content"].lower()
        if any(term in text for term in banned_terms):
            continue
        filtered_docs.append(doc)
    return filtered_docs


# =========================
# ===== GUARDRAIL =====
# OUTPUT GUARDRAIL
# =========================
def output_guardrail(answer):

    banned_patterns = [
        "password",
        "api_key",
        "secret",
        "confidential"
    ]

    for pattern in banned_patterns:
        if pattern in answer.lower():
            return "Response blocked due to security policy."

    return answer

# =========================
# ===== COHERE =====
#    RE-RANKED DOCS
# =========================
def reranked_documents(query, docs):
    documents = [doc["content"] for doc in docs]

    response = cohere_client.rerank(
        model = "rerank-english-v3.0",
        query=query,
        documents=documents,
        top_n=3
    )

    reranked_docs=[]

    for result in response.results:
        reranked_docs.append(docs[result.index])

    return reranked_docs

# =========================
# CREATE SCHEMA IF NOT EXISTS
# =========================
if not weaviate_client.schema.exists(CLASS_NAME):
    schema = {
        "class": CLASS_NAME,
        "vectorizer": "none",
        "properties": [
            {"name": "content", "dataType": ["text"]},
            {"name": "source", "dataType": ["text"]},
            {"name": "department", "dataType": ["text"]}
        ]
    }
    weaviate_client.schema.create_class(schema)
    print("Schema created", CLASS_NAME)


# =========================
# STEP 1 — DOWNLOAD FROM S3
# =========================
def download_from_s3(key, local_path):

    session = boto3.session.Session(
        profile_name="cli-user",
        region_name="us-east-1"
    )

    s3_client = session.client("s3")

    os.makedirs(local_path, exist_ok=True)
    full_path = os.path.join(local_path, key)

    if os.path.isfile(full_path):
        print("File already exists. Skipping download.")
        return full_path

    s3_client.download_file(S3_BUCKET, key, full_path)

    print(f"Downloaded {key}")
    return full_path


# =========================
# STEP 2 — EXTRACT TEXT
# =========================
def extract_text(full_path):

    if full_path.endswith(".pdf"):

        reader = PdfReader(full_path)

        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"

        return text

    else:

        with open(full_path, "r", encoding="utf-8") as f:
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
def ingest_document(s3_key, department="java"):

    file_path = download_from_s3(s3_key, local_path)

    print("Reading file:", file_path)

    text = extract_text(file_path)

    chunks = chunk_text(text)

    print(f"Total chunks: {len(chunks)}")

    embeddings = embed_chunks(chunks)

    store_chunks(chunks, embeddings, s3_key, department)


# =========================
# STEP 6 — VECTOR SEARCH
# =========================
def search(question, department_filter=None):

    question_embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    query = (
        weaviate_client.query
        .get(CLASS_NAME, ["content", "source"])
        .with_near_vector({"vector": question_embedding})
        .with_limit(5)
    )

    if department_filter:
        query = query.with_where({
            "path": ["department"],
            "operator": "Equal",
            "valueText": department_filter
        })

    result = query.do()

    try:
        return result["data"]["Get"][CLASS_NAME] or []
    except (KeyError, TypeError):
        return []


# =========================
# STEP 7 — LLM ANSWER
# =========================
def generate_answer(question, retrieved_docs):

    context = "\n\n".join([doc["content"] for doc in retrieved_docs])

    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            # {
            #     "role": "system",
            #     "content": """
            #     You are a secure enterprise assistant.
            #     Rules:
            #         1. Answer ONLY using the provided context
            #         2. If answer is not present say "I don't know"
            #         3. Never expose secrets or sensitive information
            #     """
            # },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    )

    return completion.choices[0].message.content


# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":

    while True:

        question = input("\nAsk a question (type 'exit' to quit): ").strip().lower()

        if question.lower() == "exit":
            print("Exiting...")
            break

        department = input("\nEnter department (e.g., java): ").strip().lower()

        # =========================
        # # 1. Check cache
        # =========================
        cached = get_cached_answer(question, department)

        if cached:
            print("Cache Hit")
            print("\nAnswer:")
            print(cached)
            continue

        # =========================
        # INPUT GUARDRAIL
        # =========================
        allowed, message = input_guardrail(question)

        if not allowed:
            print(message)
            continue

        # =========================
        # INGEST
        # =========================
        ingest_document("SofwareArchitectureDesignPattern.pdf", department=department)

        # =========================
        # QUERY VECTOR DB
        # =========================
        retrieved = search(question, department_filter=department)

        if not retrieved:
            print("No documents found.")
            continue

        # =========================
        # Re-ranking retrieved docs
        # =========================
        reranked_docs = reranked_documents(question,retrieved)

        # =====================================
        # RETRIEVAL GUARDRAIL on reranked docs
        # =====================================
        safe_docs = retrieval_guardrail(reranked_docs)

        if not safe_docs:
            print("All retrieved docs blocked by guardrail.")
            continue

        print("\nRetrieved Docs:")
        for r in safe_docs:
            print("-", r["source"])

        # =========================
        # LLM GENERATION
        # =========================
        answer = generate_answer(question, safe_docs)

        # =========================
        # OUTPUT GUARDRAIL
        # =========================
        final_answer = output_guardrail(answer)

        # 6. Store in cache
        cache_answer(question, department, final_answer)

        print("\nAnswer:")
        print(final_answer)