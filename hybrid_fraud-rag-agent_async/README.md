



# Fraud Detection RAG Agent (Hybrid GenAI System)

A **Fraud Detection Agent using RAG (Retrieval-Augmented Generation) 
helps detect suspicious transactions by combining:
     * Real-time transaction data (APIs)
     * Historical fraud cases (Weaviate vector DB)
     * AI reasoning (OpenAI LLM)

# Project Structure

fraud-rag-agent/
│
├── docker/
│   ├── docker-compose.yml        # Infra (Redis, Weaviate, Kafka)
│   └── requirements.txt
│
├── data/
│   └── fraud_cases.json          # Historical fraud dataset
│
├── src/
│   ├── app.py                    # FastAPI entrypoint
│   ├── configurations.py                 # Env configs (API keys, URLs)
│   │
│   ├── orchestrator/
│   │   └── agent.py              # Main routing logic (RAG vs API)
│   │
│   ├── llm_core/
│   │   └── openai_client.py      # OpenAI integration
│   │
│   ├── fraud_rag/
│   │   ├── embeddings.py         # Embedding generation
│   │   ├── weaviate_client.py    # Vector DB operations
│   │   ├── rag_reasoner.py       # Context + prompt builder
│   │
│   ├── cache/
│   │   └── redis_cache.py        # Caching layer
│   │
│   ├── ingestion/
│   │   ├── kafka_consumer.py     # Kafka ingestion service
│   │   ├── kafka_producer.py     # Send documents/events
│   │   └── ingest_data.py        # Initial data load
│   │
│   ├── services_detect/
│   │   └── fraud_api.py          # Mock payment/order APIs
│   │
│   └── utils/
│       └── helpers.py            # Common utilities
    └── tests/
│       └── test_api.py            # test_api 
│
└── README.md


# ️ System Architecture

    User → FastAPI → Orchestrator
                    ↓
     --------------------------------
     | RAG (Weaviate) | API | Cache |
     --------------------------------
                    ↓
              OpenAI LLM
                    ↓
                 Response

# Async:
Kafka → Consumer → Embeddings → Weaviate

# Infrastructure (Docker Only)

### docker-compose.yml runs:

* Weaviate → `http://localhost:8080`
* Redis → `localhost:6379`
* Kafka → `localhost:9092`

App runs separately (clean separation)

# How to Run the System

##   Start Infra

cd docker
docker compose up -d

## Load Initial Data (Optional)

cd ../src
python ingestion/ingest_data.py

## Start Kafka Consumer
python ingestion/kafka_consumer.py

## Run FastAPI App
python -m uvicorn app:app --reload
##  Open API
http://127.0.0.1:8000/docs

# Runtime Flow
## Fraud Detection Query
1. User sends transaction query
2. Orchestrator decides:
   * RAG (fraud patterns)
   * API (live transaction data)
3. Fetch context from Weaviate
4. Fetch real-time data (if needed)
5. Build prompt
6. Call OpenAI LLM
7. Cache result in Redis

## Data Ingestion Flow

1. Fraud case sent to Kafka
2. Consumer processes message
3. Chunk + generate embeddings
4. Store in Weaviate

# Key Design Highlights

* Hybrid AI → RAG + API grounding
* Event-driven ingestion (Kafka)
* Redis caching (low latency + cost)
* Modular architecture (clean separation)


# This is a hybrid GenAI fraud detection system combining RAG with real-time API grounding, backed by Redis caching and Kafka-driven ingestion, with OpenAI handling reasoning.”

# Future Enhancements
  * Semantic cache (embedding-based Redis)
  * Multi-agent fraud detection
  * Model routing (cost optimization)
  * Observability (logs + tracing)

                    Internet (Users)
                            │
                            ▼
                 ┌──────────────────────┐
                 │   API Gateway        │
                 │ (Auth, Rate Limit)   │
                 └─────────┬────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │   GenAI Orchestrator         │
            │ (ECS / EKS / FastAPI)       │
            └───────┬───────────┬─────────┘
                    │           │
        ┌───────────▼───┐   ┌───▼────────────┐
        │   RAG Flow     │   │ API Grounding  │
        │ (Weaviate DB)  │   │ (Microservices)│
        └───────┬────────┘   └──────┬────────┘
                │                   │
        ┌───────▼────────┐   ┌──────▼────────┐
        │ Vector Search  │   │ Order/Payment │
        │ + Embeddings   │   │ Services      │
        └───────┬────────┘   └───────────────┘
                │
                ▼
        ┌───────────────┐
        │   Prompt       │
        │ Augmentation   │
        └──────┬────────┘
               ▼
     ┌──────────────────────┐
     │ OpenAI ChatGPT API   │
     │ (LLM Inference)      │
     └─────────┬────────────┘
               ▼
           Response
               │
               ▼
        ┌───────────────┐
        │ Redis Cache   │
        │ (Response +   │
        │ Semantic)     │
        └───────────────┘


Async Pipeline (Kafka / MSK):
Docs → Kafka → Processor → Embeddings → Weaviate

[//]: # (Create Kafka Topic )

kafka-topics --create \
  --topic fraud-events \
  --bootstrap-server localhost:9092 \
  --partitions 1 \
  --replication-factor 1

===========================
Run the ingestion script 
===========================
PS C:\Vinod\code\python\AI\hybrid_fraud-rag-agent_async> python -m src.ingestion.ingest_data
http://localhost:8080
Schema already exists
Ingesting 7 records...
Ingestion complete
PS C:\Vinod\code\python\AI\hybrid_fraud-rag-agent_async>

==================================
to test your orchestrator service 
==================================
Terminal 1 (The Server): uvicorn src.app:app --reload
Terminal 2 (The Tester): python test_api.py

    FastAPI (app.py): Receives the JSON request.
    Orchestrator (agent.py): Coordinates the logic.
    RAG Reasoner (rag_reasoner.py): Triggers a search.
    Weaviate Client (weaviate_client.py): Fetches similar fraud patterns from your vector DB.
    LLM Core (llm_client.py): Takes the retrieved data and the query to generate a human-like fraud analysis.

=====================================================================================
Output
1- goes through API GW and store the reults in Redis
2- it goes and check in RAG vector DB, with no results. 
=======================================================================================
PS C:\Vinod\code\python\AI\hybrid_fraud-rag-agent_async\tests> python test_api.py
Sending Query: Check for suspicious patterns in transaction TXN_9988
Success!
Response: {
  "source": "API",
  "response": "Transaction Info: Transaction flagged: unusual location + high amount"
}
Sending Query: What is the fraud risk level for user 'Vinod'?   
Failed with status code: 500
Internal Server Error
PS C:\Vinod\code\python\AI\hybrid_fraud-rag-agent_async\tests>