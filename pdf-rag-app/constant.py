CHROMA_DB_PATH = "chroma" # Directory to persist the Chroma database.
DATA_PATH = "data" # Directory containing the source documents.
SEARCH_K = 3 # Number of similar documents to retrieve.

TEMPERATURE = 0.1 # LLM temperature setting.
TOP_P = 0.75 # LLM top-p setting.

LLM_MODEL = "mistral" # "mistral", "qwen3:1.7b", "gpt-oss:20b", "qwen3:8b", "qwen2.5:latest" LLM model path or name.
EMBEDDING_MODEL = "all-minilm:l6-v2" # "embeddinggemma" or "nomic-embed-text" for sentence-transformers

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
