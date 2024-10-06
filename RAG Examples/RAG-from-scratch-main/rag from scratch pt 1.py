! pip install langchain_community tiktoken langchain-openai langchainhub chromadb langchain

os.environ ['LANGCHAIN_TRACING_V2'] = 'true'
os. environ ['LANGCHAIN_ENPPOINT'] = 'https://api.smith.langchain.com'
os. environ ['LANGCHAIN_AP{_KEY'] = <>

import bs4
from langchain import hub 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma 
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.runnables import RunnablePassthrough 
from langchain_openai import ChatOpenAI,OpenAIEmbeddings



# part 2 Indexing

# Documents
question = "What kinds of pets do I like?"
document = "My favorite pet is a cat."

import tiktoken
def num_tokens_from_string(string: str, encoding_name: str) -> int:
  """Returns the number of tokens in a text string,"""
  encoding = tiktoken.get_encoding(encoding_name)
  num_tokens = len(encoding. encode(string))
  return num_tokens
num_tokens_from_string(question, "cl100k_base")


from langchain_openai import OpenAIEmbeddings
embd = OpenAIEmbeddings()
query_result = embd.embed_query(question)
document_result = embd. embed_query (document)
len (query_result)

import numpy as np
def cosine_similarity(vec1, vec2):
  dot_product = np.dot(vec1, vec2)
  norm_vec1 = np. linalg.norm(vec1)
  norm_vec2 = np. linalg.norm(vec2)
  return dot_product / (norm_vec1 * norm_vec2)
similarity = cosine_similarity(query_result, document_result)
print("Cosine Similarity:", similarity)


#### INDEXING ####
# Load blog
import bs4
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader(
  web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",), 
  bs_kwargs=dict(
    parse_only=b4.SoupStrainer(
      class=("post-content", "post-title", "post-header")
      )
    )
  )
blog_docs = loader.load()

#Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=300, chunk_over lap=50)
# Make splits
splits = text_splitter.split_documents(blog_docs)


# part 3 Retrieval

# Index
from langchain_openai import OpenAIEmbeddings 
from langchain_community.vectorstores import Chroma
vectorstore = Chroma. from_documents(documents=splits, embedding=0penAIEmbeddings ())
retriever = vectorstore.as_retriever(search_kwargs=("k": 1}) #k is number of nearby documents to be retirved

docs = retriever-get_relevant_documents("What is Task Decomposition?")



# part 4 Generation

from langchain_openai import ChatOpenAI 
from langchain.prompts import ChatPromptTemplate
# Prompt
{context}
template = '''Answer the question based only on the following context: Question: {question}'''
prompt = ChatPromptTemplate.from_template(template)
print(prompt)

# LLM
11m = ChatOpenAI (model_name="gpt-3.5-turbo", temperature=0)

# Chain
chain = prompt | 11m

# Run
chain. invoke({"context":docs, "question":"What is Task Decomposition?"})

from langchain import hub
prompt_hub_rag = hub.pull("rlm/rag-prompt")


from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
rag_chain = (
{"context": retriever, "question": RunnablePassthrough()}
| prompt
| l1m
| StrOutputParser ()
)
rag_chain.invoke("What is Task Decomposition?")












