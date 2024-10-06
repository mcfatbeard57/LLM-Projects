from langchain.memory import CassandraChatMessageHistory, ConversationBufferMemory
from langchain.llms import OpenAI
from langchain import LLMChain, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI 
import json
import getpass
import os

# OPENAI_API_KEY = ""
os.environ["GOOGLE_API_KEY"] = getpass.getpass("")

template = """
You are now the guide of a mystical journey in the Whispering Woods. 
A traveler named Elara seeks the lost Gem of Serenity. 
You must navigate her through challenges, choices, and consequences, 
dynamically adapting the tale based on the traveler's decisions. 
Your goal is to create a branching narrative experience where each choice 
leads to a new path, ultimately determining Elara's fate. 

Here are some rules to follow:
1. Start by asking the player to choose some kind of weapons that will be used later in the game
2. Have a few paths that lead to success
3. Have some paths that lead to death. If the user dies generate a response that explains the death and ends in the text: "The End.", I will search for this text to end the game

Here is the chat history, use this to understand what to say next: {chat_history}
Human: {human_input}
AI:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"],
    template=template
)

# llm = OpenAI(openai_api_key=OPENAI_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-pro")
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    # memory=cass_buff_memory
)
chat_history = []  
choice = "start"

while True:
    # Join the chat history into a single string
    chat_history_str = "\n".join(chat_history)
    response = llm_chain.predict(human_input=choice, chat_history=chat_history_str)
    
    print(response.strip())
    
    # Append the human input and AI response to the chat history
    chat_history.append(f"Human: {choice}")
    chat_history.append(f"AI: {response.strip()}")

    if "The End." in response:
        break

    choice = input("Your reply: ")