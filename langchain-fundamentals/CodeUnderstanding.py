
# imports
 # Helper to read local files
import os
# Vector Support
from langchain.vectorstores import FAISS 
from langchain.embeddings.openai import OpenAIEmbeddings
# Model and chain
from langchain.chat_models import ChatOpenAI
# Text splitters
from langchain.text_splitter import CharacterTextSplitter 
from langchain.document_loaders import TextLoader
llm = ChatOpenAI (mode1= 'gpt-3.5-turbo', openai_api_key=openai_api_key)

#embeddings
embeddings = OpenAIEmbeddings (disallowed_special=(), openai_api_key=openai_api_key)

root_dir = 'data/thefuzz'
docs = []
# Go through each folder
for dirpath, dirnames, filenames in os.walk(root_dir):
    # Go through each file
    for file in filenames:
        try:
            # Load up the file as a doc and split
            loader = TextLoader(os.path. join(dirpath, file), encoding='utf-8')
            docs. extend (loader.load_and_split())
        except Exception as e:
            pass
        
docsearch = FAISS.from_documents(docs, embeddings)

# Get our retriever ready
qa = RetrievalA. from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever ( ))
query = "What function do I use if I want to find the most similar item in a list of items?"
output = qa.run(query)


