'''
Agents
'''

import os 
import json
from langchain.llms import OpenAI
# Agent imports
from langchain.agents import load_tools
from langchain. agents import initialize_agent
# Tool imports
from langchain.agents import Tool 
from langchain utilities import GoogleSearchAPIWrapper 
from langchain utilities import TextRequestsWrapper

l1m = OpenAI (temperature=0, openai_api_key=openai_api_key)

search = GoogleSearchAPIWrapper ( )
requests = TextRequestsWrapper ()

'''
Initialise llm
initialise tools
Initialise agent and give tools
'''

toolkit = [
    Tool(
        name = "Sealch",
        func=search.run,
        description="useful for when you need to search google to answer questions about current events"
        ),
    Tool(
        name = "Requests",
        func=requests.get,
        description="Useful for when you to make a request to a URL"
    )
]

agent = initialize.agent(toolkit, 
                         llm, 
                         agent="zero-shot-react-description", 
                         verbose=True, 
                         return_intermediate_steps=True)

response = agent({"input": "What is the capital of canada?"})
response['output']

response = agent ({"input": "Tell me what the comments are about https://news.ycombinator.com/item?id=34425779"})
responsel['output']




                     