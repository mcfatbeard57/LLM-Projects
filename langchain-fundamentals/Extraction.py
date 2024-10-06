# Extract user intent from the message such as update db 

# To help construct our Chat Messages
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
# We will be using a chat model, defaults to gpt-3.5-turbo
from langchain.chat_models import ChatOpenAI
# To parse outputs and get structured data back
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

chat_model = ChatOpenAI(temperature=0, 
                        model= 'gpt-3.5-turbo', 
                        openai_api_key=openai_api_key)

# ----------------- Vanila extraction -----------------
# Make your prompt which combines the instructions w/ the fruit names
instructions ="""
You will be given a sentence with fruit names,
extract those fruit names and assign an emoji
Return the fruit name and emojis in a python dictionary
"""
fruit_names = "Apple, Pear, this is an kiwi"
# Make your prompt which combines the instructions w/ the fruit names
prompt = (instructions + fruit_names)
# Call the LLM
output = chat_model([HumanMessage(content=prompt)])
print (output.content)
print (type(output.content))

# Using Langchain response schema
# The schema I want out
response_schemas = [ResponseSchema(name="artist", 
                                   description="The name of the musical artist"), 
                    ResponseSchema(name="song", 
                                   description="The name of the song that the artist plays")
]
# The parser that will look for the LLM output in my schema and return it back to me
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# The format instructions that LangChain makes. Let's look at them 
format_instructions = output_parser.get_format_instructions() 
print (format_instructions)

# The prompt template that brings it all together
# Note: This is a different prompt template than before because we are using a Chat Model
prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template("Given a command from the user, extract the artist and song names \n \
                                                 {format_instructions}\n{user_prompt}")
        ],
    input_variables=["user_prompt"],
    partial_variables={ "format_instructions": format_instructions}
)

fruit_query = prompt.format_prompt(user_prompt="I really like So Young by Portugal. The Man")
print(fruit_query.messages[0].content)

fruit_output = chat_model(fruit_query.to_messages())
output = output_parser.parse(fruit_output.content)
print(output)
print(type(output))
