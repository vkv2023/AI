ai-property-platform/   ← Project root (not necessarily a package)
│
├── app/                ← Python package (main application)
│   ├── __init__.py
│   ├── main.py              # FastAPI entrypoint
│
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py          # API endpoints
│   ├── core/
│   │   ├── config.py          # env configs
│   │   └── security.py        # API keys / auth
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── rag_pipeline.py    # main orchestration
│   │   ├── query_rewriter.py  # Gemini rewrite agent
│   │   ├── hybrid_search.py   # BM25 + Vector
│   │   ├── reranker.py        # Cohere reranking
│   │   └── model_router.py    # multi-model routing
│
│   ├── database/
│   │   ├── __init__.py
│   │   ├── redis_cache.py
│   │   └── weaviate_client.py
│
│   ├── guardrails/
│   │   ├── __init__.py
│   │   ├── input_guardrail.py
│   │   ├── retrieval_guardrail.py
│   │   └── output_guardrail.py
│
│   └── ingestion/
│       ├── __init__.py
│       ├── crawler.py
│       ├── parser.py
│       ├── chunker.py
│       └── embeddings.py
│
│   │
│   └── schemas/
│       └── request_models.py
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
    ├── requirements.txt
│
├── scripts/
│   ├── ingest_data.py
│   └── rebuild_index.py
│
├── tests/
│

├── .env
└── README.md



HIgh level Architecture 
==========================

                User
                 |
                CloudFront
                 |
                S3 Static Website
                 |
                FastAPI (Lightsail Docker)
                 |
 |-------------------------------|
 |                               |
Redis Cache                 Weaviate
 |                               |
 |------- Hybrid Search ---------|
                |
                v
          Cohere Rerank
                |
                v
        Model Router (Gemini/OpenAI)
                |
                v
              Answer


Production workflow  (CI/CD)
=============================

Developer Push
     ↓
GitHub
     ↓
GitHub Actions (CI/CD)
     ↓
Build Docker Image
     ↓
Push Image → AWS ECR
     ↓
Lightsail Server
docker pull image
docker-compose up
run the app 