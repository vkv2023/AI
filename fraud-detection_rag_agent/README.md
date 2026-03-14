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