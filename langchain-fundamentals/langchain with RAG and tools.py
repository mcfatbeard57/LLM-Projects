'''
This has RAG, tools langchain
Tools sch as wikipedia, Arxiv, AWS 
video: https://www.youtube.com/watch?v=2_gSXyt2108&list=PLZoTAELRMXVOQPRG7VAuHL--y97opD5GQ&index=7
'''

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
from dotenv import load_dotenv
load_dotenv()
import os
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor

# Wikipedia wrapper
api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

# dco loader
loader=WebBaseLoader("https://docs.smith.langchain.com/")
docs=loader.load()
documents=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs)
vectordb=FAISS.from_documents(documents,OpenAIEmbeddings())
retriever=vectordb.as_retriever()

# langsmith loader
retriever_tool=create_retriever_tool(retriever,"langsmith_search",
                      "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!")


## Arxiv Tool
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)


tools=[wiki,arxiv,retriever_tool]

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")

### Agents
agent=create_openai_tools_agent(llm,tools,prompt)
## Agent Executer
agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)

agent_executor.invoke({"input":"Tell me about Langsmith"})
agent_executor.invoke({"input":"What's the paper 1605.08386 about?"})