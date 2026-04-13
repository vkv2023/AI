This Directory contains all the work related to RAG, LLM, AI Agents, Generative AI and Agentic AI.

1- ai-platform: This directory contains the code for the RAG system, including the implementation of the vector database, embedding generation, and response generation.
2- fraud-detection: This directory contains code related to a fraud detection system, which may utilize the RAG system for retrieving relevant information to assist in fraud detection.
3- hybrid_fraud_rag-agent_async: This directory contains code for an asynchronous agent that combines the RAG system with other techniques for fraud detection.
4- RAG-agent: This directory contains code for an agent that utilizes the RAG system for various tasks, potentially including fraud detection or other applications.
5- langchain: This directory contains code related to the LangChain framework, which may be used for building and deploying language models and agents.

======================================================================================================================================
RAG (Retrieval-Augmented Generation)
=========================================================================================================

	Installing Weaviate (Open-Source Vector Database)
	Create docker-compose.yml
		 docker compose up -d 
		 http://localhost:8080/v1/meta or in curl curl http://localhost:8080/v1/meta
	Create .env file for having openai_API_KEY
	Create requirements.txt for the dependencies

	Install python client 
		pip install weaviate-client openai tiktoken
	Create Schema (Vectorized Class) 
	Insert Documents (Embeddings are automatically created using OpenAI)
	Query + Generate Answer (RAG)
		Retrieves top similar chunks
		Sends them to OpenAI GPT
		Returns generated answer
	
	Create venv 
	Activate venv
	python <projectname.py>
	
    We have : 
        Local vector DB
        Chunk before storing for better retrieval quality
        Metadata Filtering
        OpenAI embeddings 
        OpenAI generation
        Semantic search + Hybrid Search (Vector + BM25)
        Generative responses
        
    Using model:
        Embeddings:
            text-embedding-3-small
        Generation:
            gpt-4o-mini (fast + cheap)
            gpt-4o (better quality)

========================================================
        Enterprise Level Structure
========================================================

App/
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


