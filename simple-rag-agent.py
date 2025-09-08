from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool


load_dotenv()

api_gpt = os.environ.get("OPENAI_API_KEY")
api_tavily = os.environ.get("TAVILY_API_KEY")

llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=api_gpt)
search = TavilySearch(max_results=3, api_key=api_tavily)

embeddings = OpenAIEmbeddings(openai_api_key=api_gpt)

docs = [Document(page_content="Langretrieval is a framework for developing applications powered by language models. It can be"), 
        Document(page_content="It can be used to build chatbots, Generative Question-Answering (GQA) systems, summarization tools, and much more."),
        Document(page_content="Langretrieval is designed to help developers build applications that are more advanced than those that can be built with a language model alone."),
        Document(page_content="Langretrieval provides a standard interface for all components, as well as abstractions for chains, agents, and memory.")]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
split_docs = text_splitter.split_documents(docs)

vectorstore = FAISS.from_documents(split_docs, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

rag_tool = create_retriever_tool(retriever=retriever, name="RAG-Tool", description="Useful for when you need to answer questions about langretrieval framework.")

tools = [search, rag_tool]
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant that has access to a variety of tools to help answer questions."),
    ("user", "Use the following tools to help answer the question: {input}\n\n{agent_scratchpad}")])


agent = create_tool_calling_agent(llm, tools, prompt=prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

question = "What is langretrieval? and what is the latest advancement in LLMs in the month of September 2025?"
response = agent_executor.invoke({"input": question})
print(response)
print(" ------------------------------ Final Answer: ------------------------------- ")
print(response['output'])

