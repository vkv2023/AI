from langgraph.graph import StateGraph
from typing import TypedDict
from typing_extensions import TypedDict


# ---- Step 1: Define state schema ----
class NameState(TypedDict):
    name: str


# ---- Step 2: Define functions ----
# ---- Step 2: Define Nodes (functions now use state) ----
def first_name(state: NameState) -> NameState:
    # Pass input as-is
    return {"name": state["name"]}


def last_name(state: NameState) -> NameState:
    # Add last name in the input
    return {"name": state["name"] + " Kumar"}


# ---- Step 3: Build graph ----
workflow = StateGraph(NameState)

workflow.add_node("First_Name", first_name)
workflow.add_node("Last_Name", last_name)

workflow.add_edge("First_Name", "Last_Name")

workflow.set_entry_point("First_Name")
workflow.set_finish_point("Last_Name")

app = workflow.compile()

# ---- Step 4: Run ----
if __name__ == "__main__":
    print("Direct function call:")
    result = app.invoke({"name": "Vinod"})
    print(result)
