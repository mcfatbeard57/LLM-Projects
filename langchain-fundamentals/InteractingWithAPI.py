from langchain.chains import APIChain
from langchain.llms import OpenAI
llm = OpenAI (temperature=0, openai_api_key=openai_api_key)

api_docs = '''
BASE URL: https://restcountries.com/
API Documentation:
Trusted
The API endpoint /v3.1/name/ {name} Used to find informatin about a country. All URL parameters are listed below:
- name: Name of country - Ex: italy, france
The API endpoint /v3.1/currency/{currency} Uesd to find information about a region. All URL parameters are listed below:
- currency: 3 letter currency. Example: USD, COP
Woo! This is my documentation
'''

chain_new = APIChain. from_11m_and_api_docs (llm, api_docs, verbose=True)

chain_new.run('Can you tell me Information about france?')
chain_new.run( 'Can you tell me about the currency COP?')

