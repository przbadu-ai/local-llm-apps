# Local LLM Apps

This repository contains collection of different RAG and Local LLM techniques that will use your local LLM server mostly
using `ollama`, but it can be easily migrated to use `OpenAI` compitable servers like `llamacpp`, `vllm` by switching to `openai` library.

## Dependencies

- `langchain` -> for building RAG app.
- `chroma` -> Vector database
- `ollama` -> For using local ollama models for `embedding` and `chat` functions.
- `openai` -> [optional] if you want to switch to either OpenAI, or OpenAI compitable API servers.
- `deepeval` -> various metrics for evaluating the performance of a RAG system.
- `json` -> working with JSON.
- `rank_bm25` -> Re-ranker
- `fitz`
- `numpy`

## Prerequisite

- Install UV astral, this is used to manage python environment


## Directory structure

- `evaluation/` -> contains evaluation metrices
- `data` -> contains data to work with, this is where you should move your document files that you want to chat with
- `rag-collection-scripts` -> Scripts containing various RAG techniques that you can run using `python`.
