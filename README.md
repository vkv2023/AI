This Directory contains all the work related to RAG, LLM, AI Agents, Generative AI and Agentic AI.
======================================================================================================================================
RAG (Retrieval-Augmented Generation)
================================================================================================

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

AI/
│
├── ingestion/
│   ├── s3_loader.py
│   ├── chunker.py
│   ├── embedder.py
│   └── indexer.py
│
├── search/
│   ├── retrieve.py
│   ├── rerank.py
│   └── answer.py
│
├── RAG/
│   └── rag_project/
│
├── docker/
│   └── docker-compose.yml
│
├── .env
├── requirements.txt
└── README.md