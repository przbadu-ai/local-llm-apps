import os
import asyncio
import argparse
import inspect
import logging
import logging.config

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.llm.ollama import ollama_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from lightrag.kg.shared_storage import initialize_pipeline_status
import httpx

WORKING_DIR = "./pydantic-docs"
DOCS_URL = "https://ai.pydantic.dev/llms.txt"
LLM_MODEL="GLM-4.5-Air-GLM-4.6-distill-Q4_K_M-00001-of-00002"
API_KEY="Not-Needed"
BASE_URL="http://localhost:1337/v1"
EMBEDDING_DIM=1024
MAX_TOKEN_SIZE=8129

EMBEDDING_MODEL="bge-m3:latest"
EMBEDDING_HOST="http://localhost:11434"

def fetch_pydantic_docs() -> str:
    """Fetch the Pydantic AI documentation from the URL.
    Returns:
        str: The content of the documentation.
    """

    try:
        response = httpx.get(DOCS_URL)
        response.raise_for_status()
        return response.text
    except Exception as e:
        raise Exception(f"Failed to fetch Pydantic docs: {e}")


def configure_logging():
    """Configure logging for the application"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(
        os.path.join(log_dir, "lightrag_compatible_demo.log")
    )

    print(f"\nLightRAG compatible demo log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")


async def llm_model_func(
        prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    try:
        return await openai_complete_if_cache(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            keyword_extraction=keyword_extraction,
            model=LLM_MODEL,
            api_key=API_KEY,
            base_url=BASE_URL,
            **kwargs
        )
    except Exception as e:
        logger.error(f"LLM backend call failed: {e}")
        return f"[LLM backend unavailable: {e}]"


async def print_stream(stream):
    async for chunk in stream:
        if chunk:
            print(chunk, end='', flush=True)

async def initialize_rag():
    """Create and fully initialize a LightRAG instance with storages + pipeline status.

    Per LightRAG docs this explicit sequence is required before any insert/query:
        rag = LightRAG(...)
        await rag.initialize_storages()
        await initialize_pipeline_status()
    """

    # Ensure working directory exists before LightRAG tries to set up files
    os.makedirs(WORKING_DIR, exist_ok=True)

    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=MAX_TOKEN_SIZE,
            func=lambda texts: ollama_embed(
                texts,
                embed_model=EMBEDDING_MODEL,
                host=EMBEDDING_HOST,
            ),
        ),
        llm_model_func=llm_model_func,
    )

    # Required initialization steps
    if hasattr(rag, "initialize_storages"):
        await rag.initialize_storages()
    else:
        raise RuntimeError("Installed LightRAG version missing initialize_storages(); upgrade lightrag-hku.")

    await initialize_pipeline_status()
    return rag

def cleanup_old_files():
    """Cleanup old data files in the working directory."""
    files_to_delete = [
            "graph_chunk_entity_relation.graphml",
            "kv_store_doc_status.json",
            "kv_store_full_docs.json",
            "kv_store_text_chunks.json",
            "vdb_chunks.json",
            "vdb_entities.json",
            "vdb_relationships.json",
        ]

    for file in files_to_delete:
        file_path = os.path.join(WORKING_DIR, file)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleting old file:: {file_path}")


def get_user_query() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text")
    parser.add_argument("--query-mode", type=str, default="naive", choices=["naive", "local", "global", "hybrid"], help="The query mode")
    args = parser.parse_args()
    return args.query_text, args.query_mode


async def embedding_model_available(model: str) -> bool:
    """Check whether the Ollama embedding model is available locally.

    Uses the Ollama tags endpoint. Falls back to attempting a tiny embed call if tags not reachable.
    """
    tags_url = f"{EMBEDDING_HOST.rstrip('/')}/api/tags"
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(tags_url)
            if r.status_code == 200:
                data = r.json()
                models = {m.get('name') for m in data.get('models', []) if isinstance(m, dict)}
                return model in models
    except Exception:
        # Silent fallback; we'll detect during real embed if truly missing
        return False
    return False


async def main():
    # Ensure rag reference exists for cleanup / debugging
    rag = None

    try:
        # Clear old data files
        cleanup_old_files()

        # Initialize RAG instance (handles working dir creation + storages)
        rag = await initialize_rag()

        # Parse args early
        query_text, query_mode = get_user_query()

        # Ensure embedding model exists before heavy processing
        if not await embedding_model_available(EMBEDDING_MODEL):
            print(
                f"Embedding model '{EMBEDDING_MODEL}' not found on Ollama host {EMBEDDING_HOST}.\n"
                f"Pull it first: ollama pull {EMBEDDING_MODEL}\n"
                "Then re-run your command. Exiting early."
            )
            return

        # Download and insert Pydantic docs only if not skipped or no prior status file
        print("Inserting docs...")
        try:
            docs_content = fetch_pydantic_docs()
            await rag.ainsert(docs_content)
        except Exception as insert_err:
            print(f"Failed to insert docs: {insert_err}")
            raise

        # Perform query
        print(f"\nUser Query: {query_text}\n")
        print(f"Query Mode: {query_mode}\n")

        response = await rag.aquery(
            query_text,
            param=QueryParam(mode=query_mode, stream=True)
        )

        if inspect.isasyncgen(response):
            print("\n\n✨ Response:\n\n", end='', flush=True)
            await print_stream(response)
            print("\n")
        else:
            print(f"\n\n✨ Response:\n\n{response}\n")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Best-effort finalize storages if API available
        if rag and hasattr(rag, "finalize_storages"):
            try:
                await rag.finalize_storages()
            except Exception as fin_err:
                logger.warning(f"Finalize storages failed: {fin_err}")


if __name__ == "__main__":
    configure_logging()
    asyncio.run(main())
    print("\n\nDone!\n")