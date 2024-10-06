import os
import time
import arxiv
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings

# Create directory if not exists
dirpath = "arxiv_papers"
if not os.path.exists(dirpath):
    os.makedirs(dirpath)

# Search arXiv for papers related to "LLM"
client = arxiv.Client()
search = arxiv.Search(
    query="LLM",
    max_results=10,
    sort_order=arxiv.SortOrder.Descending
)

# Download and save the papers
for result in client.results(search):
    while True:
        try:
            result.download_pdf(dirpath=dirpath)
            print(f"-> Paper id {result.get_short_id()} with title '{result.title}' is downloaded.")
            break
        except (FileNotFoundError, ConnectionResetError) as e:
            print("Error occurred:", e)
            time.sleep(5)

# Load papers from the directory
papers = []
loader = DirectoryLoader(dirpath, glob="./*.pdf", loader_cls=PyPDFLoader)
try:
    papers = loader.load()
except Exception as e:
    print(f"Error loading file: {e}")
print("Total number of pages loaded:", len(papers)) 

# Concatenate all pages' content into a single string
full_text = ''
for paper in papers:
    full_text += paper.page_content

# Remove empty lines and join lines into a single string
full_text = " ".join(line for line in full_text.splitlines() if line)
print("Total characters in the concatenated text:", len(full_text)) 

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
paper_chunks = text_splitter.create_documents([full_text])

# Create Qdrant vector store
qdrant = Qdrant.from_documents(
    documents=paper_chunks,
    embedding=GPT4AllEmbeddings(),
    path="./tmp/local_qdrant",
    collection_name="arxiv_papers",
)
retriever = qdrant.as_retriever()

# Define prompt template
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Initialize Ollama LLM
ollama_llm = "llama2:7b-chat"
model = ChatOllama(model=ollama_llm)

# Define the processing chain
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)

# Add typing for input
class Question(BaseModel):
    __root__: str

# Apply input type to the chain
chain = chain.with_types(input_type=Question)
result = chain.invoke("Explain about Vision Enhancing LLMs")
print(result)
