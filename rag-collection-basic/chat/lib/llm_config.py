from embedchain import App
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file if present

BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:8081/v1")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
API_KEY = os.getenv("OPENAI_API_KEY", "")

if LLM_PROVIDER == "ollama":
    llm_model = os.getenv("LLM_MODEL", "llama3:instruct")
else:
    llm_model = os.getenv("LLM_MODEL", "gpt-4-turbo")


print(
    f"Using LLM Provider: {LLM_PROVIDER}, Model: {llm_model}, Embedding Model: {EMBEDDING_MODEL}"
)
print(f"base_url: {BASE_URL}")


class LlmConfig:
    """Helper class to initialize LLM with embedding using Ollama/OpenAI provider configuration."""

    def __init__(self):
        self.base_url = BASE_URL
        self.model = llm_model
        self.embed_model = EMBEDDING_MODEL
        self.provider = LLM_PROVIDER
        self.api_key = API_KEY

    def create_bot(self, db_path: str, api_key: str) -> App:
        if self.provider == "ollama":
            config = {
                "llm": {
                    "provider": "ollama",
                    "config": {
                        "model": self.model,
                        "max_tokens": 250,
                        "temperature": 0.5,
                        "stream": True,
                        "base_url": self.base_url,
                    },
                },
                "vectordb": {"provider": "chroma", "config": {"dir": db_path}},
                "embedder": {
                    "provider": "ollama",
                    "config": {"model": self.embed_model, "base_url": self.base_url},
                },
            }
        else:
            config = {
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": self.model,
                        "temperature": 0.5,
                        "api_key": api_key,
                    },
                },
                "vectordb": {"provider": "chroma", "config": {"dir": db_path}},
                "embedder": {"provider": "openai", "config": {"api_key": api_key}},
            }

        return App.from_config(config=config)
