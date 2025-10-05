import argparse

from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from populate_database import main as populate_database

from get_embedding_function import get_embedding_function

from constant import CHROMA_DB_PATH, PROMPT_TEMPLATE, SEARCH_K, LLM_MODEL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text")
    parser.add_argument("--reset", action="store_true", help="Reset the database before querying")
    args = parser.parse_args()
    query_text = args.query_text

    # Generate indexing to vector db
    populate_database(args.reset)

    # search and query the RAG system
    query_rag(query_text)


def query_rag(query_text: str) -> None:
    """Query the RAG system with the provided text."""
    embedding_function = get_embedding_function()
    db = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embedding_function,
    )

    # Search the database for relevant documents.
    results = db.similarity_search_with_score(query_text, k=SEARCH_K)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(
        context=context_text,
        question=query_text,
    )
    print("✨ Prompt:\n", prompt)

    model = OllamaLLM(model=LLM_MODEL)
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text



if __name__ == "__main__":
    main()
