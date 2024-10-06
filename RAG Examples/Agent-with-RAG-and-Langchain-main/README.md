# Agent-with-RAG-and-Langchain


This repo is about building a custom AI agent using Langchain and Retrieval Augmented Generation (RAG). The agent can summarize and respond to coding related issues from a GitHub repository.

Here are the steps on how to build the AI agent:

- Install required packages including python-dotenv, requests, Langchain, Langchain dastra-db, Langchain OpenAI, and Langchain Hub.
- Write a function to load issues from a GitHub repository. This function takes a list of issues as input and parses them into a format that can be used by the retrieval augmented generation model.
- Write another function to fetch issues from a GitHub repository. This function takes the owner and repository name as input and returns a list of issues.
- Create a Retriever tool that uses the vector store to search for similar issues. This tool takes a query as input and returns a list of similar issues.
- Create a Note tool that saves a note to a file. This tool takes a note as input and saves it to a file called notes.txt.
- Combine the Retriever tool and the Note tool into a single agent.

The agent can be a useful tool for developers who want to automate some of their tasks.
