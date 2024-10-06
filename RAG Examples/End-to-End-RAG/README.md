# End-to-End-RAG
This is a guide to build production level RAG
1. Architecture
	1. Monitoring(Langsmith)
	2. Retrieval(OpenAl, Pinecone)
	3. Generation(Groq, Llama3)
	4. Orchestration(Langchain)
	5. Guardrails(Nemo-Guardrails)
2. Flowchart
	1. Data Loading(Web Loader) -> Chunking(Text Spliiter) -> Vectorstore(Pinecone, OpenAl)
	2.  Input(User Query) -> Input Check(Guard Rails) -> Retriever(Pinecone)
	3. Generator(Groq, Llama3) -> Monitoring(Langsmith) - > Output
.
