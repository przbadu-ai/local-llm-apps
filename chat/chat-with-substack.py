import streamlit as st
from lib.llm_config import LlmConfig
from embedchain import App
import tempfile

llm_config = LlmConfig()

def embedchain_bot(db_path, api_key):
    return llm_config.create_bot(db_path, api_key)

st.title("Chat with Substack Newsletter 📝")
st.caption("This app allows you to chat with Substack newsletter using OpenAI API")

openai_access_token = llm_config.api_key
if not openai_access_token and llm_config.provider == "openai":
    openai_access_token = st.text_input("OpenAI API Key", type="password")

# Create a temporary directory to store the database
db_path = tempfile.mkdtemp()

# Create an instance of Embedchain App
app = embedchain_bot(db_path, openai_access_token)

# Get the Substack blog URL from the user
substack_url = st.text_input("Enter Substack Newsletter URL", type="default")

if substack_url:
    # Add the Substack blog to the knowledge base
    app.add(substack_url, data_type='substack')
    st.success(f"Added {substack_url} to knowledge base!")

    # Ask a question about the Substack blog
    query = st.text_input("Ask any question about the substack newsletter!")

    # Query the Substack blog
    if query:
        result = app.query(query)
        st.write(result)