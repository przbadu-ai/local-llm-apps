from embedchain import App
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file if present

base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
llm_provider = os.getenv('LLM_PROVIDER', 'ollama')
embedding_model = os.getenv('EMBEDDING_MODEL', 'nomic-embed-text')
api_key = os.getenv('OPENAI_API_KEY', '')

if llm_provider == 'ollama':
    llm_model = os.getenv('LLM_MODEL', 'llama3:instruct')
else:
    llm_model = os.getenv('LLM_MODEL', 'gpt-4-turbo')


print(f"Using LLM Provider: {llm_provider}, Model: {llm_model}, Embedding Model: {embedding_model}")
print(f"base_url: {base_url}")

class LlmConfig:
    """Helper class to initialize LLM with embedding using Ollama/OpenAI provider configuration."""
    def __init__(self):
        self.base_url = base_url
        self.model = llm_model
        self.embed_model = embedding_model
        self.provider = llm_provider
        self.api_key = api_key

    def create_bot(self, db_path: str) -> App:
        if self.provider == "ollama":
            config={
                "llm": {"provider": "ollama", "config": {"model": self.model, "max_tokens": 250, "temperature": 0.5, "stream": True, "base_url": self.base_url}},
                "vectordb": {"provider": "chroma", "config": {"dir": db_path}},
                "embedder": {"provider": "ollama", "config": {"model": self.embed_model, "base_url": self.base_url}},
            }
        else:
            config={
                "llm": {"provider": "openai", "config": {"model": self.model, "temperature": 0.5, "api_key": self.api_key}},
                "vectordb": {"provider": "chroma", "config": {"dir": db_path}},
                "embedder": {"provider": "openai", "config": {"api_key": self.api_key}},
            }

        return App.from_config(config=config)
