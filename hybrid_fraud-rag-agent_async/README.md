



# 🚀 Fraud Detection RAG Agent (Hybrid GenAI System)

A **Fraud Detection Agent using RAG (Retrieval-Augmented Generation)** helps detect suspicious transactions 
by combining:

* Real-time transaction data (APIs)
* Historical fraud cases (Weaviate vector DB)
* AI reasoning (OpenAI LLM)

---

# 📁 Project Structure

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
│   ├── config.py                 # Env configs (API keys, URLs)
│   │
│   ├── orchestrator/
│   │   └── agent.py              # Main routing logic (RAG vs API)
│   │
│   ├── llm/
│   │   └── openai_client.py      # OpenAI integration
│   │
│   ├── rag/
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
│   ├── services/
│   │   └── fraud_api.py          # Mock payment/order APIs
│   │
│   └── utils/
│       └── helpers.py            # Common utilities
│
└── README.md
# ️ System Architecture (Simplified)
    User → FastAPI → Orchestrator
                    ↓
     --------------------------------
     | RAG (Weaviate) | API | Cache |
     --------------------------------
                    ↓
              OpenAI LLM
                    ↓
                 Response

Async:
Kafka → Consumer → Embeddings → Weaviate
---

# Infrastructure (Docker Only)

### docker-compose.yml runs:

* Weaviate → `http://localhost:8080`
* Redis → `localhost:6379`
* Kafka → `localhost:9092`

App runs separately (clean separation)

# ⚙️ How to Run the System

##   Start Infra

cd docker
docker compose up -d
---

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
---

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


# 💡  One-Liner

> “This is a hybrid GenAI fraud detection system combining RAG with real-time API grounding, 
> backed by Redis caching and Kafka-driven ingestion, with OpenAI handling reasoning.”


# Future Enhancements

* Semantic cache (embedding-based Redis)
* Multi-agent fraud detection
* Model routing (cost optimization)
* Observability (logs + tracing)

