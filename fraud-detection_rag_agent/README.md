A fraud detection agent using RAG (Retrieval-Augmented Generation) in a payment system helps detect suspicious transactions by combining real-time transaction data + historical fraud knowledge + AI reasoning.

fraud-rag-agent/
│
├
├── docker/
├     └──docker-compose.yml
├     └──requirements.txt
│
├── data/
│   └── fraud_cases.json
│
├── src/
│   ├── app.py
│   ├── embeddings.py
│   ├── weaviate_client.py
│   ├── redis_cache.py
│   ├── ingest_data.py
│   ├── fraud_agent.py
│   └── rag_reasoner.py
│
└── README.md

1- Start Infra (Docker) from docker folder:
    docker compose up -d
2- Go to Source Folder
    cd ..\src
3- Run API
    python -m uvicorn app:app --reload
4- Open API
    http://127.0.0.1:8000/docs


        Docker Compose → Infra only
            ↓
        Weaviate (8080)
        Redis (6379)
        
        Python App → runs separately
            ↓
        FastAPI (8000)