import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI
import tiktoken

# from langchain.chat_models import init_chat_model
# model = init_chat_model("gpt-5-nano")
# response = model.invoke("Hello!")

MAX_TOKENS = 1200

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

print("LangSmith Key:", LANGCHAIN_API_KEY)

model = ChatOpenAI(model_name="gpt-5-nano",
                   openai_api_key=OPENAI_API_KEY,
                   temperature=0,
                   max_tokens=MAX_TOKENS)


def trim_to_tokens(text, max_tokens=1200, model="gpt-5-nano"):
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)

    return enc.decode(tokens[:max_tokens])


# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers and return the result."""
    return a * b


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers and return the result."""
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide numbers and return the result."""
    return a / b


# Augment the LLM with tools using LangGraph
tools = [multiply, add_numbers, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

# Define the State schema
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator


class CalculatorState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]  # dont overwrite the message history, append to it
    llm_calls: int


# Define model node
from langchain.messages import SystemMessage


def llm_call(state: CalculatorState) -> CalculatorState:
    """LLM decides whether to call a tool or not"""
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }


# Step 4: Define tool node
from langchain_core.messages import ToolMessage


def tool_node(state: CalculatorState) -> CalculatorState:
    """Executes tool calls from the last LLM message"""

    result = []

    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]

        observation = tool.invoke(tool_call["args"])

        result.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call["id"]
            )
        )

    return {"messages": result}


# Define logic to determine whether to end  the graph or not
from langgraph.graph import StateGraph, START, END
from typing import Literal


# Conditional edge function to route to the tool node or
# end based upon whether the LLM made a tool call
def should_continue(state: CalculatorState) -> Literal["tool_node", END]:
    """Route to tool or end based on LLM output"""

    last_message = state["messages"][-1]

    # Safe check if None returned from the LLM or if the message is not a tool message
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"

    return END


# Build agent workflow

agent_builder = StateGraph(CalculatorState)
# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()

from IPython.display import Image, display

# Show the agent
with open("images/agent_graph_API.png", "wb") as f:
    f.write(agent.get_graph(xray=True).draw_mermaid_png(max_retries=5, retry_delay=2.0))
    # display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

# ---- Run ----
from langchain.messages import HumanMessage

if __name__ == "__main__":
    while True:
        user_input = input("\nEnter your query (type 'exit' to quit): ")
        if user_input.lower() == "exit":
            print("Exiting...")
            break

        trimmed_input = trim_to_tokens(user_input, 1200)

        messages = [HumanMessage(content=trimmed_input)]
        result = agent.invoke({"messages": messages})

        # messages = [HumanMessage(content=user_input)]
        # result = agent.invoke({"messages": messages})

        last_message = result["messages"][-1]
        # Token count and metadata
        print(last_message.usage_metadata)

        print("\nAgent Response:")
        for m in result["messages"]:
            m.pretty_print()
