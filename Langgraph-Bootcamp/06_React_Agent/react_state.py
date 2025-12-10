import operator
from typing import Annotated,TypedDict,Union
from langchain_core.agents import AgentAction,AgentFinish

"""
    Represents one step of the agent where it chooses a tool to run.

    AgentAction(
        tool: str,         # name of the tool to run
        tool_input: str,   # input to that tool
        log: str = ""      # raw LLM output (optional)
    )

    from langchain_core.agents import AgentAction

    action = AgentAction(
        tool="calculator",
        tool_input="2 + 2",
        log="I will calculate 2 + 2"
    )

"""

"""
    from langchain_core.agents import AgentFinish
    finish = AgentFinish(
        return_values={"output": "The answer is 4"},
        log="I have completed the calculation."
    )

    AgentAction(
        tool="calculator",
        tool_input="2 + 2"
    )

    AgentFinish(
        return_values={"output": "The answer is 4"}
    )
"""

class AgentState(TypedDict):
    input: str
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

"""
        {
            "input": "What is 2 + 2?",
            "agent_outcome": AgentFinish(
                    return_values={"output": "The answer is 4"},
                    log="Finished."
            ),
            "intermediate_steps": [
            (
                AgentAction(
                    tool="calculator",
                    tool_input="2 + 2",
                    log="I will calculate 2 + 2"
                ),
            "4"  # tool result
        )
    ]
}
"""