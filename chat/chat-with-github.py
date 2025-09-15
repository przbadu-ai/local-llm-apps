import tempfile
from embedchain.pipeline import Pipeline as App
from embedchain.loaders.github import GithubLoader
import streamlit as st
from lib.llm_config import LlmConfig
import os
from dotenv import load_dotenv

load_dotenv()

github_token = os.getenv('GITHUB_PAT', '')

def get_loader():
    loader = GithubLoader(
        config={
            "token": github_token
        }
    )
    return loader

if "loader" not in st.session_state:
    st.session_state['loader'] = get_loader()

# Github loader
loader = st.session_state.loader

# LLM config
llm_config = LlmConfig()


def embedchain_bot(db_path, api_key):
    return llm_config.create_bot(db_path, api_key)

def load_repo(git_repo):
    global app
    # Add repo to the knowledge base
    app.add(f"repo:{git_repo} type:repo", data_type="github", loader=loader)
    st.success(f"Added {git_repo} to knowlege base!")

def make_db_path():
    ret = tempfile.mkdtemp(suffix="chroma")
    print(f"Created chroma DB at {ret}")
    return ret

st.title("Chat with Github Repository")
st.caption("This app allows you to chat with a Github Repo using local Ollama models")

openai_access_token = llm_config.api_key
if not openai_access_token and llm_config.provider == "openai":
    openai_access_token = st.text_input("OpenAI API Key", type="password")

# Initialize the embedchain App
if "app" not in st.session_state:
    st.session_state['app'] = embedchain_bot(make_db_path(), openai_access_token)

app = st.session_state.app

# Get the github repo from the user
git_repo = st.text_input("Enter the Github Repo", type="default")

if git_repo and ("repos" not in st.session_state or git_repo not in st.session_state.repos):
    if "repos" not in st.session_state:
        st.session_state["repos"] = [git_repo]
    else:
        st.session_state.repos.append(git_repo)
    load_repo(git_repo)

# Ask a question about the Github Repo
prompt = st.text_input("Ask any question about the Github Repo")

# Chat with the Github Repo
if prompt:
    answer = st.session_state.app.chat(prompt)
    st.write(answer)

