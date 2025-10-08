# RAG App

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Clone the project and switch directory to `rag-app-pdf/` directory
3. Activate the environment, install required packages, also install chromadb

```sh
source .venv/bin/activate
uv sync
```

## Run the app

1. Copy your documents (PDF) inside `data/` directory and run:

```sh
python main.py "<query>" [--reset]
```

> NOTE: `--reset` is optional arguments, you can use this if you want to reset and reindex vector db
