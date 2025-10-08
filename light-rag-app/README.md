1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Clone the project and switch directory to `rag-app-pdf/` directory
3. Activate the environment, install required packages, also install chromadb

```sh
source .venv/bin/activate
uv sync
```

## We are using Ollama for embedding model so it expect you to have ollama and required model:

```sh
ollama pull bge-m3:latest
```

## Run the app

1. Copy your documents (PDF) inside `data/` directory and run:

```sh
python main.py "<query>" [--query-mode]
```

- `--query-mode`: supported values are `"naive", "local", "global", "hybrid"`, default: `"naive"`
