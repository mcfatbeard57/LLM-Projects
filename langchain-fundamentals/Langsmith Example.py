
"""
This is a Langsmith example.
Video Link: https://www.youtube.com/watch?v=3Gcm27l-uyQ
Setup Langsmith, open ai API key
"""


# imports
from langchain.openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import Stroutputparser
import os
from dotenv import load_dotenv

load_dotenv ()

os. environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_Key")
# this will trace everything in langsmith dashboard
os. environ ["LANGCHAINI TRACING_V2" ]="true"
os. environ[" LANGCHAIN_API_KEY" ]=os.getenv("LANGCHAIN_API_KEY")

prompt = ChatPromptTemplate(
    ("system", "You are helpful agent of the government. Lets take it down together"),
    ("user", "Quesitons:{question}\nContext:{context}"),
)

#define model and chain
model = ChatOpenAI(temperature=0,model="gpt-3.5-turbo")
output_parser = Stroutputparser()
chain = prompt| model | output_parser
# Give your question and context
question = ""
context = ""

output = chain.invoke({"question": question, "context": context})

