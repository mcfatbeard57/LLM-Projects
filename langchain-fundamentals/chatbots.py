from langchain.llms import OpenAI 
from langchain import LLMChain 
from langchain.prompts.prompt import PromptTemplate
# Chat specific components
from langchain.memory import ConversationBufferMemory

template ="""
You are a chatbot that is unhelpful.
Your goal is to not help the user but only make jokes I Take what the user is saying and make a joke out of it
{chat_history}
Human: {human_ input}
Chatbot:
"""
prompt = PromptTemplate (
    input_variables=[
        "chat_history", "human_input"], 
    template=template
    )
memory = ConversationBufferMemory (memory_key="chat_history")

llm_chain = LLMChain(
    llm=OpenAI(openai_api__key=openai__api__key),
    prompt=prompt, 
    verbose=True, 
    memory=memory)
llm_chain.predict (human_input="Is an pear a fruit or vegetable?")
llm_chain.predict(human_input="What was one of the fruits I first asked you about?")

