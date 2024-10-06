# imports
from langchain.llms import OpenAI 
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

# load env files
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Note, the default model is already 'text-davinci-003' but I call it out here explicitly so you know where to change it later if you
llm = OpenAI (temperature=0, model_name = 'text-davinci-003', 
              openai_api_key=openai_api_key)


# Create our template
template = """
%INSTRUCTIONS:
Please summarize the following piece of text.
Respond in a manner that a 5 year old would understand.
%TEXT :
{text}
"""
# Create a LangChain prompt template that we can insert values to later
prompt = PromptTemplate (
    input_variables=["text"], 
    template=template)

# ------------------------ Short Summaries ------------------------
# pass in short text
confusing_text = ""
# create final prompt
final_prompt = prompt.format(text=confusing_text)
print(final_prompt)
# print llm response
print(llm(final_prompt))

# ------------------------ Long Summaries ------------------------
# load document into text
long_text=""
# use character splitter to split the long text into docs
text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=5000, chunk_overlap=350)
docs = text_splitter.create_documents([long_text])
# Get your chain ready to use
chain = load_summarize_chain(llm=llm, chain_type= 'map_reduce')
print(chain.run(docs))

