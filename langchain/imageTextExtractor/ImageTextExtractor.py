import os
from typing import List, TypedDict
from dotenv import load_dotenv

# LangChain & Vector DB Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_weaviate import WeaviateVectorStore
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langgraph.graph import StateGraph, END
import weaviate

load_dotenv()

# Global reference for the retriever to be shared across graph nodes
retriever = None

# 1. SETUP WEAVIATE
WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://weaviate:8080")
auth_config = weaviate.auth.AuthApiKey(api_key=os.environ.get("WEAVIATE_API_KEY"))
client = weaviate.Client(url=WEAVIATE_URL, auth_client_pass=auth_config)


class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    retry_count: int
    is_valid: bool


# 2. INGESTION FUNCTION
def ingest_data(file_path: str):
    global retriever
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found at: {file_path}")

    loader = PyMuPDFLoader(file_path)
    raw_docs = loader.load()

    # Semantic Chunking
    text_splitter = SemanticChunker(OpenAIEmbeddings())
    docs = text_splitter.split_documents(raw_docs)

    # Vector Storage
    vectorstore = WeaviateVectorStore.from_documents(
        docs, OpenAIEmbeddings(), client=client, index_name="SecurityPolicy"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    return retriever


# 3. GRAPH NODES
def retrieve_and_rerank(state: AgentState):
    if retriever is None:
        raise ValueError("Retriever not initialized. Ensure ingest_data() is called on startup.")

    compressor = CohereRerank(model="rerank-english-v3.0")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    # Updated for LangChain 0.3 syntax
    docs = compression_retriever.invoke(state["question"])
    return {"context": [d.page_content for d in docs]}


def generate_answer(state: AgentState):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"Context: {state['context']}\n\nQuestion: {state['question']}"
    response = llm.invoke(prompt)
    return {"answer": response.content}


def sub_agent_feedback(state: AgentState):
    critic_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    verification_prompt = (
        f"Does this answer use ONLY the provided context? Answer 'YES' or 'NO'.\n"
        f"Context: {state['context']}\nAnswer: {state['answer']}"
    )
    check = critic_llm.invoke(verification_prompt).content
    valid = "YES" in check.upper()
    return {"is_valid": valid, "retry_count": state.get("retry_count", 0) + 1}


# 4. ORCHESTRATION
def should_retry(state):
    if state["is_valid"] or state["retry_count"] >= 3:
        return "end"
    return "retry"


workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_and_rerank)
workflow.add_node("generate", generate_answer)
workflow.add_node("validate", sub_agent_feedback)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "validate")
workflow.add_conditional_edges("validate", should_retry, {"end": END, "retry": "retrieve"})

pipeline_app = workflow.compile()