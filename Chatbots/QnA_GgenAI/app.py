# Q&A Chatbot
#from langchain.llms import OpenAI

# take environment variables from .env
from dotenv import load_dotenv
load_dotenv()  

# Imports
import streamlit as st
import os

import pathlib
import textwrap

import google.generativeai as genai
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

from IPython.display import display
from IPython.display import Markdown


def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))



## Function to load Gemini Pro/LLM model and get respones for questions
def get_gemini_response(question):
    # Load Model
    model = genai.GenerativeModel('gemini-pro')
    # Generate Response
    response = model.generate_content(question)
    return response.text


##initialize our streamlit app
st.set_page_config(page_title="Q&A Page; We will answer all your questions")
st.header("Goole Gemini Application")
input=st.text_input("Input: ",key="input")
 
submit=st.button("Away with the question")

## If ask button is clicked
if submit:
    response=get_gemini_response(input)
    st.subheader("Generated resposne as per Gemini is:\n")
    st.write(response)
