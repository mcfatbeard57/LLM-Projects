"""
Use langchian to create chatbot using open and paid llms
video : https://www.youtube.com/watch?v=5CJA1Hbutqc
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import Stroutputparser
import streamlit as st 
import os
from dotenv import load_dotenv
# third party integration is avaible in langchain_community
from langchain_community import Ollama


os. environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os. environ["LANGCHAIN_TRACING_V2" ]="true"
os. environ["LANGCHAIN API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

# Prompt template
prompt = ChatPromptTemplate.from_template(
    ("system", "You are a helpful assistant that translates hepls user with its questions."),
    ("user", "Question:{question}\nContext:{context}")
)

# Streamlit framework
st.title('Langchain Demo With OPENAI API') 
input_text=st.text_input("Search the topic u want")

#open AI llm
llm=ChatOpenAI(temperature=0,model="gpt-3.5-turbo")
# 11m=011ama(model="11ama2" )
# Output parser
output_parser = Stroutputparser()
#chain
chain = prompt|llm|output_parser

if input_text:
    output = chain.invoke({'question':input_text,'context':""})
    st.write(output)

