from typing import List, Union
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage
from langgraph.graph import END, MessageGraph
from chains import revisor_chain, first_responder_chain
from tool import execute_tools
from schema import AnswerQuestion, ReviseAnswer
import os

graph = MessageGraph()
MAX_ITERATIONS = 4

from chains import revisor_chain, first_responder_chain

def draft_node(state: list[BaseMessage]) -> list[BaseMessage]:
    """
    Node for 'draft':
    - Takes the message history (state)
    - Calls first_responder_chain, which returns AnswerQuestion (Pydantic)
    - Wraps it into an AIMessage so LangGraph is happy
    """
    result: AnswerQuestion = first_responder_chain.invoke({"messages": state})

    msg = AIMessage(
        content=result.answer,
        additional_kwargs={
            "search_queries": result.search_queries,
            "reflection": result.reflection.model_dump() if hasattr(result, "reflection") else None,
        },
    )
    return [msg]




def revisor_node(state: list[BaseMessage]) -> list[BaseMessage]:
    # Count how many previous revisor outputs we already have
    past_revisions = sum(
        isinstance(m, AIMessage) and m.additional_kwargs.get("role") == "revisor"
        for m in state
    )

    result: ReviseAnswer = revisor_chain.invoke({"messages": state})

    msg = AIMessage(
        content=result.answer,
        additional_kwargs={
            "role": "revisor",
            "iteration": past_revisions + 1,
            "search_queries": result.search_queries,
            "reflection": (
                result.reflection.model_dump()
                if hasattr(result, "reflection")
                else None
            ),
            "references": result.references,
        },
    )
    return [msg]



# nodes
graph.add_node("draft", draft_node)
graph.add_node("execute_tools", execute_tools)
graph.add_node("revisor", revisor_node)


# edges
graph.add_edge('draft', 'execute_tools')
graph.add_edge('execute_tools', 'revisor')


def event_loop(state: List[BaseMessage]) -> str:
    # How many times has revisor run?
    num_iterations = sum(
        isinstance(m, AIMessage) and m.additional_kwargs.get("role") == "revisor"
        for m in state
    )

    # 1) Stop if we've hit the max number of revisions
    if num_iterations >= MAX_ITERATIONS:
        return END

    # 2) Also stop if there are no search_queries to run tools with
    last_ai = next((m for m in reversed(state) if isinstance(m, AIMessage)), None)
    if not last_ai:
        return END

    search_queries = last_ai.additional_kwargs.get("search_queries", [])
    if not search_queries:
        # Nothing more to search → no point looping
        return END

    # Otherwise, go to tools again
    return "execute_tools"

graph.add_conditional_edges('revisor', event_loop)
#graph.add_edge("revisor", END)

graph.set_entry_point('draft')

app = graph.compile()

print(os.environ.get('GROQ_API_KEY'))

#print(app.get_graph().draw_mermaid())
#app.get_graph().print_ascii()

response = app.invoke('Write about how small business can leverage AI to grow')
print(response)
