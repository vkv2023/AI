— LangGraph is a low-level orchestration framework and runtime for building, 
    managing, and deploying long-running, stateful agents.

LangGraph is focused on the underlying capabilities important for agent orchestration:
    durable execution, streaming, human-in-the-loop, memory, persistence.

Use the Graph API if you prefer to define your agent as a graph of nodes and edges for complex.
Use the Functional API if you prefer to define your agent as a single function or with minimal code changes.

User → LangGraph → LLM → Tools → LLM → Output
    Token Limit, Predictable Cost, and Performance:

1. Input Limit (Safe Context Control) - Keep last N messages (simple + effective)
2. Summarization Node
3. 