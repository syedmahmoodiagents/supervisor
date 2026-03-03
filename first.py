import os
from dotenv import load_dotenv
from torch import subtract
load_dotenv()
hf = os.getenv("HF_TOKEN")

# print(hf)

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

endpoint = HuggingFaceEndpoint(repo_id='openai/gpt-oss-20b')
llm = ChatHuggingFace(llm=endpoint)

def subtract(a: int, b: int) -> int:
    "Subtract the second number from the first and return the result."
    return a - b

def add(a: int, b: int) -> int:
    "Add two numbers together and return the result."
    return a + b

agent = create_react_agent(llm, tools=[add, subtract])

query="find out What is 5 + 3 and then subtract 2 from the result?"

resp = agent.invoke({"messages": [HumanMessage(content=query)]})

# resp = agent.invoke({"input": })
print(resp['messages'][-1].content)