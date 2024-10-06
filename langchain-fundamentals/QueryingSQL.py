'''
talk to a db
Steps:
1. Find which table to use
2. Find which column to use
3. Construct the correct sql query
4. Execute that query
5. Get the result
6. Return a natural language response back
'''

# Imports and intialise llm
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
llm = OpenAI (temperature=0, openai_api_key=openai_api_key)

#initialise db
sqlite_db_path = 'data/San_Francisco Trees.db'
db = SQLDatabase. from_uri(f"sqlite:///{sqlite_db_path}")

db_chain = SQLDatabaseChain (llm=llm, database=db, verbose=True)
db_chain. run ("How many Species of trees are therg?")

