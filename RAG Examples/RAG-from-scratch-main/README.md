# End2End-RAG



#GENAI #Study
#RAG

###### Frameworks
1. [[Langchain]]
2. [[LlamaIndex]]

###### VectorDBs
1. [[Azure Cassandra]]
2. [[ChromaDB]]
3. [[Faiss DB]]

###### Inference
1. [[GROQ]]


###### Embeddings:
1. Open AI Embeddings


###### Hyperparameters:
1. Chunk size, Overlap size
2. Metadata
3. Top N results
4. 'k' hyperparameter is used to retrieve 'k' nearby documents
5. Relevancy score


###### Notes:
1. Vector space is determined by semantic meaning in the document
2. In [[LangSmith]] we can check how the retrieval process is going on
3.  Most Vector DBs has meta data which can be queried upon
4. Can stich any complex flow using [[Langgraph]]
5. Any LLM works better with better Signal to Noise ratio

[RAG repo](https://github.com/mcfatbeard57/RAG-from-scratch)
##### RAG BASICS w Images:

[Basic RAG code](https://github.com/mcfatbeard57/RAG-from-scratch/blob/main/rag%20from%20scratch%20pt%201.py)

![[Pasted image 20240629141250.png | 300]]
![[Pasted image 20240629141656.png | 300]]
![[Pasted image 20240629144853.png | 300]]
![[Pasted image 20240629151347.png | 300]]






#### RAG Components:
1. Query construction
	1. [Query Construction Code](https://github.com/mcfatbeard57/RAG-from-scratch/blob/main/rag%20from%20scratch%20pt%203.py)
	2. Query Structuring:
		1. Convert Natural language question into structured query
		2. ![[Pasted image 20240703125544.png | 100]]
		3. Most vector DBs has metadata which can be queried and filtered. this is on top of chunks that are indexed
2. Query translation
	1. [Query Translation Code](https://github.com/mcfatbeard57/RAG-from-scratch/blob/main/rag%20from%20scratch%20pt%202.py)
	2. Multi Query: 
		1. Re-writing or rephrasing orignal question to list of question then retrieve relevant docs wrt each query
	3. RAG Fusion:
		1. Rewriting original query with multiple perspective to get wider set of relevant documents. Then use rank to combine those documents depending upon thier frequency and passing them to model.
		2. This is helpful if we are querying multiple vector stores or a large number of queries
		3. ![[Pasted image 20240703120019.png | 100]]
	4. Decomposition:
		1. Sub Question Decomposition: Use dynamic retrieval to answer initial question then use the QnA pair along with context to answer succeeding question. Use both preceding QnA pair to along with context of 3rd question  to answer that 3ed question
		2. Independent Question Decomposition: another approach is to answer all 3 questions individually and then concatenate them together
		3. ![[Pasted image 20240703122627.png | 100]]
	5. Step back Prompting:
		1. Create a step back or more abstract question counter to the actual question, can use few shot prompting to let LLM know what to do and then do RAG on both origanl and step back questions to get broader context and give this context along with orignal question to LLM
		2. Helpful in domain specific quesitons 
	6. HYDE:
		1. Map Question to hypothetical doc, using prompting and LLM, and use that for retrieval of context from Vectory DB. Thought process is that hypothetical doc would be closer to those docs in vector db in comparison to the original questions
		2. ![[Pasted image 20240703122744.png | 100]]
		3. END
3. Routing
	3. [Routing Code](https://github.com/mcfatbeard57/RAG-from-scratch/blob/main/rag%20from%20scratch%20pt%203.py)
	4. Routing the question to relevant DB, e.g. Vecotr DB, graph DB or relation DB. It could also be considered as routing to relevant prompt.
	5. Types of Routing:
		1. Logical Routing: Let LLMs reason out which is the relevant DB or relevant coding language. ![[Pasted image 20240703123009.png | 100]]
		2. Semantic Routing: Identify and Choose the best prompt, i.e. routing to different prompt. based on semantic semalirity ![[Pasted image 20240703123039.png | 100]]
		3. Logical Routing e.g. ![[Pasted image 20240703125216.png | 100]]
4. Indexing
	1. Chunk Optimisation
		1. 1. [Chunking Video](https://www.youtube.com/watch?v=8OJC21T2SL4)
		2. [Chunking Code](https://github.com/mcfatbeard57/RAG-from-scratch/blob/main/rag%20from%20scratch%20pt%204_1.py)
		3. Theory: Split data to better prepare for LLMs. goal is to get our data in a format where it can be retrieved for value later.
		4. Level 1: Character
			1. Pros: Easy & Simple
			2. Cons: Very rigid and doesn't take into account the structure of your text
			4. Chunk Size - The number of characters you would like in your chunks. 50, 100, 100,000, etc.
			5. Chunk Overlap - The amount you would like your sequential chunks to overlap. This is to try to avoid cutting a single piece of context into multiple pieces. This will create duplicate data across chunks.
		5. Level 2: Recursive Character
		6. Level 3: Document Specific
		7. Level 4: Semantic Splitting
		8. Level 5: Agentic Splitting
		9. Bonus Level: Indexing
	2. Multi Representation indexing
		1. Multi vector Retriever -> Multi Representation Indexing:
		1. You store summary of Docs in Vector DB and store docs in DOC db, then query using LLM to get relevant summary from vector DB and then the full doc corresponding to the summary from doc DB
		2. ![[Pasted image 20240703130659.png | 100]]
	3. Specialised Embeddings (Fine Tuning, ColBERT)
		5. [Indexing & ColBERT Code](https://github.com/mcfatbeard57/RAG-from-scratch/blob/main/rag%20from%20scratch%20pt%204.py)
		6. ColBERT
		1. RAGatouille is a way to easily use ColBERT
		2. It is a fast and accurate retrieval model, enabling scalable BERT-based search over large text collections in tens of milliseconds.
		3. Converting doc in tokens as well as question and finding the similarity between them
		4. ![[Pasted image 20240703140125.png | 100]]
	4. Hierarchal Embeddings (RAPTOR) Raptor
		1. [RAPTOR CODE](https://github.com/mcfatbeard57/RAG-from-scratch/blob/main/rag%20from%20scratch%20pt%204_2.py)
		2. Useful when questions require to leaf through multiple documents, then "k" hyper parameter might not be able to overcome as # of docs > k.
		3. Start with initial relevant docs/chunks (leafs), summarise them into clusters, do this recursively where either you hit a limit or end up with one big cluster
		4. Now you can pass these leafs for low level questions and the cluster summaries for high level questions to get the proper semantic meaning.
		5. [Deep Dive]([https://www.youtube.com/watch?v=jbGchdTL7d0](https://www.youtube.com/watch?v=jbGchdTL7d0))
		6. [Code]([https://github.com/langchain-ai/langchain/blob/master/cookbook/RAPTOR.ipynb](https://github.com/langchain-ai/langchain/blob/master/cookbook/RAPTOR.ipynb))
		7. ![[Pasted image 20240703131232.png | 100]]
	5. 
5. Retrieval
	1. [Retrieval Code](https://github.com/mcfatbeard57/RAG-from-scratch/blob/main/rag%20from%20scratch%20pt%205.py)
	2. Re ranking:
		1. Rerank initial docs which comes from semantic meaning. Can also use Cohere rank
	3. Challenges when Implementing RAG requires logical reasoning around these steps:
		1. We can ask when to retrieve
		2. When to re-write the question for better retrieval
		3. When to discard irrelevant retrieved documents and re-try retrieval
		4. FLOW : ![[Pasted image 20240703170923.png | 100]]
6. Generation

![[Pasted image 20240703115325.png | 500]]

![[Pasted image 20240703184453.png | 300]]



#### Advance RAG 
4. Active RAG: LLM decides when and what to retrieve based upon retrieval and / or generation.
	1. ![[Pasted image 20240703171043.png | 100]] ![[Pasted image 20240703184304.png | 100]]
5. Cognitive Architectures or CRAG
	1. [CODE](https://github.com/mcfatbeard57/RAG-from-scratch/blob/main/Corrective%20RAG.ipynb)
	2. State machines are a good way to implement active RAG
	3. LangGraph is a recently released way to build state machines
	4. Layout of diverse RAG flows and supports the more general process of "flow engineering"
	5. ![[Pasted image 20240703171215.png | 100]] ![[Pasted image 20240703174207.png | 100]] ![[Pasted image 20240703184400.png | 100]]
	6. [CRAG 1](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_crag.ipynb)
	7. Create a grader to grade your retrieval docs. when relevancy score for any retrieval crosses a threshold, then generate. If none passes the threshold then retrieve based on web search and use that as context.
7. Adaptive RAG
	1. [Code](https://github.com/mcfatbeard57/RAG-from-scratch/blob/main/Adaptive%20RAG%20w%20Cohere.ipynb)
	2. Query Analysis - Re writing and Routing
	3. Online Testing
	4. ![[Pasted image 20240703174319.png | 100]]


#### Benefits of RAG over longer context window
1. Better withReasoning and getting more facts right
2. Recency Buias: In context window, LLM might forget fact in the beginning of context but RAG can overcome this.
3. RAG today focused on precise retrieval of relevant doc chunks



#### References:
1. [RAG](https://www.youtube.com/watch?v=sVcwVQRHIc8)
2. [RAG repo](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_crag.ipynb)
3. [RAG from scratch](https://github.com/mcfatbeard57/RAG-from-scratch)

.
