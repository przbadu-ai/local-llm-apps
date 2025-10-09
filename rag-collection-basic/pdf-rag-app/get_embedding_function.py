from langchain_ollama import OllamaEmbeddings

from constant import EMBEDDING_MODEL

# Function to get the embedding function using OllamaEmbeddings
# with the specified model "embeddinggemma", or "nomic-embed-text" as a fallback.
def get_embedding_function():
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )
    return embeddings
