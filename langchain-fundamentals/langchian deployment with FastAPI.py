'''
Video: https://www.youtube.com/watch?v=XWB5DXP-DO8&list=PLZoTAELRMXVOQPRG7VAuHL--y97opD5GQ&index=4
This will have app, client, route and llm
'''

# ---- app.py ----
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langserve import add_routes
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.output_parsers import StructuredOutputParser
from langchain.output_parsers import ResponseSchema
import uvicorn
import os
from langchain_community.llms import Ollama

# get openai key

# start fastapi
app= FastAPI(
    title="Langchain Server", version="1.0",
    decsription="A simple API Server"
)

add_routes(
    app,
    # OpenAI(model="gpt-3.5-turbo"),
    ChatOpenAI(model="gpt-3.5-turbo"),
    path="/openai",
)


model=ChatOpenAI()
##ollama 1lama2
llm=Ollama (model="11ama2" )
prompt1=ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 word")
prompt2=ChatPromptTemplate.from_template("Write me a poem about {topic} with 100 word")

add_routes ( 
            app, 
            prompt1|model,
            path="/essay"
)

add_routes(
    app, 
    prompt2|llm,
    path="/poem"
)

if __name__ == "__main__":
    uvicorn.run(app,host="localhost",port=8000)
    
    
    
    

# ---- client.py ----
import requests
import streamlit as st

def get_OpenAI_response(input_text):
    response = requests.post(
        "http://localhost:8000/essay/invoke",
        json={"input":{'topic': input_text} },
    )

def get_ollama_response (input_text):
    response=requests.post(
        "http://localhost:8000/poem/invoke", 
        json={'input':{'topic':input_text}}
    )
st. title( 'Langchain Demo With LLAMA2 API')
input_text=st.text_input ("Write an essay on")
input_textl=st.text_input("Write a poem on" )

if input_text:
    st.write(get_OpenAI_response(input_text))
if input_textl:
    st.write(get_ollama_response(input_textl))
