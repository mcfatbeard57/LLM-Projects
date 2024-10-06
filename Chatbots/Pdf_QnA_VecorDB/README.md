# Q&A Demo Using Astra DB and LangChain with Vector Search

## Overview

This project demonstrates a question-answering system using Astra DB and LangChain, powered by Vector Search. The system leverages a Serverless Cassandra database with Vector Search capabilities to handle and query PDF documents efficiently. The aim is to provide an intuitive Q&A interface for extracting information from PDFs.

## Features

- **Vector Search:** Efficient search through PDF documents using vector embeddings.
- **LangChain Integration:** Utilizes LangChain for natural language processing and question-answering capabilities.
- **Astra DB:** Powered by Astra DB's Serverless Cassandra with Vector Search for scalable and fast data retrieval.

## Prerequisites

1. **Astra DB Account:**
   - Sign up for an Astra DB account [here](https://astra.datastax.com/).
   - Create a Serverless Cassandra database with Vector Search enabled.

2. **Database Token:**
   - Obtain a DB Token with the role `Database Administrator`.
   - Copy your `Database ID`.

3. **Python Environment:**
   - Python 3.8 or higher.
   - Jupyter Notebook installed.

## Installation

1. **Clone the Repository:**
   ```sh
   git clone https://github.com/mcfatbeard57/Pdf_QnA_VecorDB.git
   cd qa-demo-astra-langchain
   ```

2. **Install Dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Configuration

1. **Set Up Astra DB Connection:**
   - Create a file named `.env` in the root directory.
   - Add your Astra DB connection parameters to the `.env` file:
     ```env
     ASTRA_DB_ID=your_database_id
     ASTRA_DB_TOKEN=your_db_token
     ```

2. **Prepare Your PDFs:**
   - Place the PDF documents you want to query in the `pdfs` directory.

## Running the Project

1. **Start Jupyter Notebook:**
   ```sh
   jupyter notebook
   ```

2. **Open and Run the Notebook:**
   - Navigate to the `qa_demo.ipynb` notebook.
   - Follow the instructions in the notebook to upload your PDFs, initialize the vector search, and start asking questions.

## Usage

- **Uploading PDFs:** The notebook provides a section where you can upload PDF documents to the Astra DB.
- **Querying PDFs:** Use the Q&A interface to input your questions. The system will return relevant answers extracted from the uploaded PDFs.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.
