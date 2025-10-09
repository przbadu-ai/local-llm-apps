import argparse
import os
import shutil

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma

from get_embedding_function import get_embedding_function

from constant import CHROMA_DB_PATH, DATA_PATH


def main(reset=False):
    """Populate the Chroma database with documents from the data directory."""
    if reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents() -> list[Document]:
    """Load documents from the data directory."""
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]) -> None:
    """Add document chunks to the Chroma database."""
    # Initialize the Chroma database with the embedding function.
    db = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=get_embedding_function(),
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the chunks in the database.
    existing_items = db.get(include=[]) # IDs are included by default.
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in the database: {len(existing_ids)}")

    # Only add documents that don't exist in the database.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("No new documents to add.")


def calculate_chunk_ids(chunks: list[Document]) -> list[Document]:
    """
    Calculate unique IDs for each document chunk.
    The ID format is: <Page source> : <Page Number> : <Chunk Index>
    Example: "data/monopoly.pdf:6:2"
    """

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page metadata.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database() -> None:
    """Clear the chroma database."""
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database")
    args = parser.parse_args()
    main(reset=args.reset)