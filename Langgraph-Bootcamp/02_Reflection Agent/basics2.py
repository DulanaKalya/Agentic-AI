from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END
from typing import List, TypedDict, Annotated
from langgraph.graph.message import add_messages

from chains import generation_chain, reflection_chain

load_dotenv()

# 1. Define state schema
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

graph = StateGraph(State)

REFLECT = "reflect"
GENERATE = "generate"

# 2. Nodes
def generate_node(state: State) -> State:
    # generation_chain should return an AIMessage or similar
    result = generation_chain.invoke({"messages": state["messages"]})
    return {"messages": [result]}  # add_messages will merge

def reflection_node(state: State) -> State:
    # reflection_chain returns something with .content
    response = reflection_chain.invoke({"messages": state["messages"]})
    reflection_msg = HumanMessage(content=response.content)
    return {"messages": [reflection_msg]}  # add_messages will merge

graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflection_node)

graph.set_entry_point(GENERATE)

# 3. Conditional routing
def should_continue(state: State):
    if len(state["messages"]) > 2:
        return END
    return REFLECT

graph.add_conditional_edges(GENERATE, should_continue)
graph.add_edge(REFLECT, GENERATE)

# 4. Compile + visualize
app = graph.compile()
print(app.get_graph().draw_mermaid())
app.get_graph().print_ascii()
