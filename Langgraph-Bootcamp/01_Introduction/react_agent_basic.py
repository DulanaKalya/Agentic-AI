from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from langchain_core.prompts import PromptTemplate
from langchain_classic.agents import create_react_agent, AgentExecutor ,tool
import datetime

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Part 01

#result = llm.invoke("Give me the temperature of colombo,sri lanka now")
#print(result)


#Part 02

from langchain_tavily import TavilySearch
search_tool = TavilySearch()

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """ Returns the current date and time un the specified format"""
    current_time=datetime.datetime.now()
    formated_time = current_time.strftime(format)
    return formated_time

tools = [search_tool,get_system_time]


react_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(react_template)
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors="Reformat using Thought/Action/Observation steps or return a 'Final Answer:' line."
)
response = agent_executor.invoke({"input": "When was recent disaster in Sri Lanka and how many days ago was that from this instant and date of today"})
print(response["output"])