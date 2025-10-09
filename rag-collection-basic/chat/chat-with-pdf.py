# Import necessary libraries
import os
import tempfile
import streamlit as st
from embedchain import App
from lib.llm_config import LlmConfig

llm_config = LlmConfig()


# Define the embedchain_bot function
def embedchain_bot(db_path, api_key):
    return llm_config.create_bot(db_path, api_key)


st.title("Chat with PDF")
st.caption(
    "This app allows you to chat with a PDF using Llama3 running locally wiht Ollama!"
)

openai_access_token = llm_config.api_key
if not openai_access_token and llm_config.provider == "openai":
    openai_access_token = st.text_input("OpenAI API Key", type="password")

# Create a temporary directory to store the PDF file
db_path = tempfile.mkdtemp()

# Create an instance of the embedchain App
app = embedchain_bot(db_path, openai_access_token)

# Upload a PDF file
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

# If a PDF file is uploaded, add it to the knowledge base
if pdf_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(pdf_file.getvalue())
        app.add(f.name, data_type="pdf_file")
    os.remove(f.name)
    st.success(f"Added {pdf_file.name} to knowledge base!")

# Ask a question about the PDF
prompt = st.text_input("Ask a question about the PDF")

# Display the answer
if prompt:
    answer = app.chat(prompt)
    st.write(answer)

