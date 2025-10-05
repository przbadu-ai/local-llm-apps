import streamlit as st
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama
from agno.tools.arxiv import ArxivTools
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file if present

# Get the environment variables
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
API_KEY = os.getenv("OPENAI_API_KEY", "")
llm_model = os.getenv("LLM_MODEL", "llama3")

os.environ["OLLAMA_HOST"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# Set up the Streamlit app
st.title("Chat with Research Papers 🔎🤖")
st.caption(
    f"This app allows you to chat with arXiv research papers using {llm_model} running locally."
)


# Set the environment variables as needed
if LLM_PROVIDER == "ollama":
    model = Ollama(id=llm_model)
else:
    openai_access_token = API_KEY
    if not openai_access_token and LLM_PROVIDER == "openai":
        openai_access_token = st.text_input("OpenAI API Key", type="password")

    model = OpenAIChat(
        id=llm_model, max_tokens=1024, temperature=0.9, api_key=openai_access_token
    )


# Agent
assistant = Agent(model=model, tools=[ArxivTools()])

# Get the search query from the user
query = st.text_input("Enter the Search Query", type="default")

if query:
    # Search the web using the AI Assistant
    response = assistant.run(query, stream=False)
    st.write(response.content)
