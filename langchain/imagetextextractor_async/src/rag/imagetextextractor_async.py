import os
import logging.config
import yaml
from typing import List, TypedDict, Annotated
from dotenv import load_dotenv

# LangChain & Vector DB Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_weaviate import WeaviateVectorStore
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
import weaviate
from opentelemetry import trace
import weaviate.classes.init as wvc

tracer = trace.get_tracer(__name__)

load_dotenv()

# Setup Logging - Ensure logs directory exists
# log_config_path = os.path.join(os.path.dirname(__file__), 'logging_config.yaml')

log_config_path = os.getenv("LOG_CONFIG_PATH", "/app/logging_config.yaml")

with open(log_config_path, 'r') as f:
    log_config = yaml.safe_load(f)

# logging.config.dictConfig(log_config)
logger = logging.getLogger('ImageTextExtractor')

# Global reference for the retriever to be shared across graph nodes
retriever = None

# 1. SETUP WEAVIATE
logger.info(f"Connecting to Weaviate at {os.environ.get('WEAVIATE_HOST', 'localhost')}")
weaviate_host = os.getenv("WEAVIATE_HOST", "weaviate.image-text-extractor-rag-ai.svc.cluster.local")
client = weaviate.connect_to_custom(
    http_host=os.getenv("WEAVIATE_HOST", "weaviate_host"),
    http_port=int(os.getenv("WEAVIATE_HTTP_PORT", "8080")),
    grpc_host=os.getenv("WEAVIATE_HOST", "weaviate_host"),
    grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
    http_secure=False,
    grpc_secure=False
)

logger.info("Successfully connected to Weaviate")


class AgentState(TypedDict, total=False):
    question: str
    context: List[str]
    answer: str
    retry_count: int
    is_valid: bool


# 2. INGESTION FUNCTION
def ingest_data(file_path: str):
    logger.info(f"Resolved ingestion file path: {file_path}")
    global retriever
    logger.info(f"Starting data ingestion from: {file_path}")

    if not os.path.exists(file_path):
        logger.error(f"PDF not found at: {file_path}")
        raise FileNotFoundError(f"PDF not found at: {file_path}")

    logger.debug("Loading PDF documents...")
    loader = PyMuPDFLoader(file_path)
    raw_docs = loader.load()
    logger.info(f"Loaded {len(raw_docs)} documents")

    # Semantic Chunking
    logger.debug("Performing semantic chunking...")
    # text_splitter = SemanticChunker(OpenAIEmbeddings())
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    # docs = text_splitter.split_documents(raw_docs)
    docs = text_splitter.split_documents(raw_docs)
    logger.info(f"Chunked into {len(docs)} semantic chunks")

    # Vector Storage
    logger.debug("Storing documents in Weaviate vector store...")
    vectorstore = WeaviateVectorStore.from_documents(
        docs, OpenAIEmbeddings(), client=client, index_name="SecurityPolicy"
    )
    logger.info("Documents successfully stored in Weaviate")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    logger.info("Retriever initialized successfully")
    return retriever


# 3. GRAPH NODES

def retrieve_and_rerank(state: AgentState):
    # This creates a segment in your trace specifically for document retrieval
    with tracer.start_as_current_span("weaviate_retrieval"):
        logger.debug(f"Retrieving and reranking documents...")
    logger.debug(f"Retrieving and reranking documents for query: {state['question']}")

    if retriever is None:
        logger.error("Retriever not initialized")
        raise ValueError("Retriever not initialized. Ensure ingest_data() is called on startup.")

    logger.debug("Initializing Cohere reranker...")
    compressor = CohereRerank(model="rerank-english-v3.0", cohere_api_key=os.getenv("COHERE_API_KEY"))
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever,
    )

    # Use get_relevant_documents() method for retrievers
    docs = compression_retriever.get_relevant_documents(state["question"])
    logger.info(f"Retrieved and reranked {len(docs)} documents")
    return {"context": [d.page_content for d in docs]}


def generate_answer(state: AgentState):
    logger.debug(f"Generating answer for question: {state['question']}")
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    prompt = f"Context: {state['context']}\n\nQuestion: {state['question']}"
    response = llm.invoke(prompt)
    logger.info("Answer generated successfully")
    logger.debug(f"Answer: {response.content}")
    return {"answer": response.content}


def sub_agent_feedback(state: AgentState):
    logger.debug(f"Validating answer for question: {state['question']}")
    critic_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    verification_prompt = (
        f"Does this answer use ONLY the provided context? Answer 'YES' or 'NO'.\n"
        f"Context: {state['context']}\nAnswer: {state['answer']}"
    )
    check = critic_llm.invoke(verification_prompt).content
    valid = "YES" in check.upper()
    logger.info(f"Answer validation result: {'VALID' if valid else 'INVALID'} (Retry count: {state.get('retry_count', 0) + 1})")
    logger.debug(f"Validation response: {check}")
    return {"is_valid": valid, "retry_count": state.get("retry_count", 0) + 1}


# 4. ORCHESTRATION
def should_retry(state):
    if state["is_valid"] or state["retry_count"] >= 3:
        logger.debug("Pipeline complete - no more retries")
        return "end"
    logger.debug(f"Retrying retrieval (attempt {state['retry_count'] + 1}/3)")
    return "retry"

