# Qna using docs as context

from langchain.llms import OpenAI

# # The vectorstore we'll be using
from langchain.vectorstores import FAISS
# The LangChain component we'll use to get the documents
from langchain.chains import RetrievalQA
# The easy document loader for text
from langchain. document_loaders import TextLoader
# The embedding engine that will convert our text to vectors
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI (temperature=0, openai_api_key=openai_api_key)

# use context + question for short answer
context=""" Rachel is 30 years old, 
Bob is 45 years old, 
Kevin is 65 years old"""
question = "Who is under 40 years old?"
print(llm(context + question))


# --------------------- use embeddings ---------------------

# load documnet
loader = TextLoader ('path-to-text-file')
doc = loader.load()
print (f"You have {len(doc)} document")
print (f"You have {len(doc[0].page_content)} characters in that document" )

# splitting text
text_splitter = RecursiveCharacterTextSplitter (chunk_size=3000, chunk_overlap=400)
docs = text_splitter.split_documents(doc)
# Get your embeddings engine ready
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
# Embed your documents and combine with the raw text in a pseudo db. Note: This will make an API call to OpenAI
docsearch = FAISS.from_documents(docs, embeddings)
# Create retrieval engine
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff"
                                 ,retriever=docsearch.as_retriever())
query = "ask your question related to documents"
qa.run(query)
