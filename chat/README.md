## 📄 Chat with [PDF, Youtube, Substack, Email, Website, ...]

LLM + RAG app to chat with PDF, YouTube, Substack, ets. The app uses Retrieval Augmented Generation (RAG) to provide accurate answers to questions based on the content of the uploaded PDF.

We are using in-app chroma db for vector database, and using ollama to provide local LLM models.

### Features

- Upload a PDF document, YouTube [URL/video id], Substack url
- Ask questions about the resource
- Get accurate answers using RAG and the selected LLM

### How to get Started?

1. Clone the GitHub repository

```bash
git clone https://github.com/przbadu/local-llm-apps.git
cd local-llm-apps/chat
```

2. Configure environment variables

```bash
cp .env.example .env
```

Also make sure to use your available LLM models for embedding and chat. Default values are:

- For Embedding: `nomic-embed-text`
- For Chat: `gpt-oss` (default 20b model)

3. Install [uv](https://docs.astral.sh/uv/)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Install the required dependencies

```bash
source .venv/bin/activate
uv sync
```

4. Run the Streamlit App

**chat with YouTube**

```bash
streamlit run chat-with-youtube.py
```

**chat with PDF**

```bash
streamlit run chat-with-pdf.py
```

