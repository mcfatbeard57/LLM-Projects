from langchain import HuggingFaceHub, LLMChain
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
# import streamlit as st



prompt_template = PromptTemplate.from_template("Tell me a joke about {topic}")

prompt_template.invoke({"topic": "cats"})
'''
Part-1: Create code
- Give multiple code documentations in "advance RAG" & Use it to create code in any language
- Use agents for code type selection
- Advance version is another agent will look for code documentation online, download it and index it and then another agent use it.
- Start with react

Part-2: Chat with Git Repos 
Create a connection to a remote git repo
Option to download their code in local
Load an LLM from Ollama's hub

'''
