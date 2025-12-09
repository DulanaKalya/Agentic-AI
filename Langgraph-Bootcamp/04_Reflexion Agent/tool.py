import json
from typing import List, Dict, Any
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, HumanMessage
from langchain_community.tools import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

tavily_tool = TavilySearchResults(max_results=1)

def execute_tools(state: List[BaseMessage]) -> List[BaseMessage]:
    last_ai_message: AIMessage = state[-1]
    
    # Extract tool calls from the AI message
    if not hasattr(last_ai_message, "tool_calls") or not last_ai_message.tool_calls:
        return []
    
    # Process the AnswerQuestion or ReviseAnswer tool calls to extract search queries
    tool_messages = []
    
    for tool_call in last_ai_message.tool_calls:
        if tool_call["name"] in ["AnswerQuestion", "ReviseAnswer"]:
            call_id = tool_call["id"]
            search_queries = tool_call["args"].get("search_queries", [])
            
            # Execute each search query using the tavily tool
            query_results = {}
            for query in search_queries:
                result = tavily_tool.invoke(query)
                query_results[query] = result
            
            # Create a tool message with the results
            tool_messages.append(
                ToolMessage(
                    content=json.dumps(query_results),
                    tool_call_id=call_id
                )
            )
    
    return tool_messages


# Example usage
test_state = [
    HumanMessage(
        content='Write about how small business can leverage AI to grow'
    ),
    AIMessage(
        content='', 
        tool_calls=[
            {
                'name': 'AnswerQuestion',
                'args': {
                    'answer': '', 
                    'search_queries': [
                            'capital of sri lanka,give single word answer', 
                            'who is anura kumara dissanayake,give single word answer'
                    ], 
                    'reflection': {
                        'missing': '', 
                        'superfluous': ''
                    }
                },
                'id': 'call_KpYHichFFEmLitHFvFhKy1Ra',
            }
        ],
    )
]
