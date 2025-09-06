from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

import os
from dotenv import load_dotenv

load_dotenv()

api_gpt = os.environ.get("OPENAI_API_KEY")
api_tavily = os.environ.get("TAVILY_API_KEY")

llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=api_gpt)
search = TavilySearch(max_results=3, api_key=api_tavily)

tools = [search]

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant that has access to a variety of tools to help answer questions."),
    ("user", "Use the following tools to help answer the question: {input}\n\n{agent_scratchpad}")])

# agent = create_tool_calling_agent(llm, tools, prompt=prompt_template)
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# question = "What are the latest advancements in renewable energy technologies?"
# response = agent_executor.invoke({"input": question})
# print(response)
# print(" ------------------------------ Final Answer: ------------------------------- ")
# print(response['output'])


@tool
def covnert_temperature(value: float, to:str = "C") -> str:
    """Convert temperature between Celsius and Fahrenheit."""
    if to.upper() == "C":
        celsius = (value - 32) * 5.0/9.0
        return f"{value}°F is {celsius:.2f}°C"
    elif to.upper() == "F":
        fahrenheit = (value * 9.0/5.0) + 32
        return f"{value}°C is {fahrenheit:.2f}°F"
    else:
        return "Invalid target unit. Please use 'C' for Celsius or 'F' for Fahrenheit."
    
tools = [search, covnert_temperature]
agent = create_tool_calling_agent(llm, tools, prompt=prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
question = "Convert 100 degrees Fahrenheit to Celsius & fetch me the weather in boston, USA today."
response = agent_executor.invoke({"input": question})
print(response)
