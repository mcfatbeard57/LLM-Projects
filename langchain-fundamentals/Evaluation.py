from langchain.embeddings.openai import OpenAIEmbeddings 
from langchain.vectorstores import FAISS 
from langchain.chains import RetrievalQA
# Model and doc loader
from langchain import OpenAI 
from langchain.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Eval!
from langchain.evaluation.qa import QAEvalChain
import os
from dotenv import load_dotenv


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI (temperature=0, openai_api_key=openai_api_key)

"""
Vector Dance:    
1. Load the documents
2. Create the embeddings
3. Create the FAISS index
4. Create the retriever
5. Create the chain
"""
loader = TextLoader ('path-to-text-file')
doc = loader.load()
# splitting text
text_splitter = RecursiveCharacterTextSplitter (chunk_size=3000, chunk_overlap=400)
docs = text_splitter.split_documents(doc)
# Get your embeddings engine ready
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
# Embed your documents and combine with the raw text in a pseudo db. Note: This will make an API call to OpenAI
docsearch = FAISS.from_documents(docs, embeddings)

chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(), input_key="question")
# get llm answers
predictions = chain.apply(question_answers)

eval_chain = QAEvalChain.from_llm(llm=llm, chain=chain)

# Have it grade itself. The code below helps the eval_chain know where the different parts
graded_outputs = eval_chain.evaluate(question_answers,
                                      predictions,
                                      question_key='question',
                                      prediction_key='result',
                                      answer_key='answer')

