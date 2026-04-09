import logging.config
from langgraph.graph import StateGraph, END

from src.rag.imagetextextractor_async import (
    retrieve_and_rerank,
    generate_answer,
    sub_agent_feedback,
    should_retry,
    AgentState
)

logger = logging.getLogger('Pipeline')


logger.info("Building LangGraph workflow...")
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_and_rerank)
workflow.add_node("generate", generate_answer)
workflow.add_node("validate", sub_agent_feedback)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "validate")
workflow.add_conditional_edges("validate", should_retry, {"end": END, "retry": "retrieve"})

pipeline_app = workflow.compile()
logger.info("LangGraph workflow compiled successfully")
