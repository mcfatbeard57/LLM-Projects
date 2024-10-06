from fastapi import FastAPI
from transformers import pipeline

## create a new FASTAPI app instance 
app=FastAPI ( )

# Initialize the text generation pipeline
pipe = pipeline("text2text-generation", model="google/flan-t5-small")

@app-get("/")
def home ():
  return {"message": "Hello World"}

# Define a function to handle the GET request at /generate |
@app-get("/generate")
def generate (text:str):
  ## use the pipeline to generate text from given input text
  output=pipe (text)
  
  ## return the generate text in Json reposne 
  return {"output" output [®][' generated_text
