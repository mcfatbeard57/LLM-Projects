import functools, operator, requests, os, json
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
import gradio as gr

# Set environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangGraph Research Agents"

# Initialize model
llm = ChatOpenAI(model="gpt-4-turbo-preview")

# 1. Define custom tools
@tool("internet_search", return_direct=False)
def internet_search(query: str) -> str:
    """Searches the internet using DuckDuckGo."""
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=5)]
        return results if results else "No results found."

@tool("process_content", return_direct=False)
def process_content(url: str) -> str:
    """Processes content from a webpage."""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

tools = [internet_search, process_content]

# 2. Agents 
# Helper function for creating agents
def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

# Define agent nodes
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

# Create Agent Supervisor
members = ["Web_Searcher", "Insight_Researcher"]
system_prompt = (
    "As a supervisor, your role is to oversee a dialogue between these"
    " workers: {members}. Based on the user's request,"
    " determine which worker should take the next action. Each worker is responsible for"
    " executing a specific task and reporting back their findings and progress. Once all tasks are complete,"
    " indicate with 'FINISH'."
)

options = ["FINISH"] + members
function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {"next": {"title": "Next", "anyOf": [{"enum": options}] }},
        "required": ["next"],
    },
}

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}"),
]).partial(options=str(options), members=", ".join(members))

supervisor_chain = (prompt | llm.bind_functions(functions=[function_def], function_call="route") | JsonOutputFunctionsParser())

search_agent = create_agent(llm, tools, "You are a web searcher. Search the internet for information.")
search_node = functools.partial(agent_node, agent=search_agent, name="Web_Searcher")

insights_research_agent = create_agent(llm, tools, 
        """You are a Insight Researcher. Do step by step. 
        Based on the provided content first identify the list of topics,
        then search internet for each topic one by one
        and finally find insights for each topic one by one.
        Include the insights and sources in the final response
        """)
insights_research_node = functools.partial(agent_node, agent=insights_research_agent, name="Insight_Researcher")

# Define the Agent State, Edges and Graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

workflow = StateGraph(AgentState)
workflow.add_node("Web_Searcher", search_node)
workflow.add_node("Insight_Researcher", insights_research_node)
workflow.add_node("supervisor", supervisor_chain)

# Define edges
for member in members:
    workflow.add_edge(member, "supervisor")

conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
workflow.set_entry_point("supervisor")

graph = workflow.compile()

# Run the graph
for s in graph.stream({
    "messages": [HumanMessage(content="""Search for the latest AI technology trends in 2024,
            summarize the content. After summarise pass it on to insight researcher
            to provide insights for each topic""")]
}):
    if "__end__" not in s:
        print(s)
        print("----")

# final_response = graph.invoke({
#     "messages": [HumanMessage(
#         content="""Search for the latest AI technology trends in 2024,
#                 summarize the content
#                 and provide insights for each topic.""")]
# })

# print(final_response['messages'][1].content)