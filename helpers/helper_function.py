from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from typing import List
from rank_bm25 import BM25Okapi
import fitz
import asyncio
import random
import textwrap
import numpy as np
from enum import Enum

from config.llm_config import *


def replace_t_with_space(list_of_documents):
    """
    Replace all instances of '\t' with a space in the text of each document.

    Args:
        list_of_documents (List[Document]): List of Document objects.

    Returns:
        List[Document]: List of Document objects with '\t' replaced by space.
    """
    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace('\t', ' ')
    return list_of_documents

def text_wrap(text, width=120):
    """
    Wrap text to a specified width.

    Args:
        text (str): The input text to be wrapped.
        width (int): The maximum line width.

    Returns:
        str: The wrapped text.
    """
    return textwrap.fill(text, width=width)

def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    """
    Encodes a PDF book into a vector store using Ollama embeddings.

    Args:
        path (str): Path to the PDF file.
        chunk_size (int): Size of each text chunk.
        chunk_overlap (int): Overlap between text chunks.

    Returns:
        Chroma: A Chroma vector store containing the encoded text chunks.
    """
    # Load the PDF documents
    loader = PyPDFLoader(path)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    # Create embeddings and vector store
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(cleaned_texts, embeddings)

    return vectorstore


def encode_from_string(content, chunk_size=1000, chunk_overlap=200):
    """
    Encodes text content into a vector store using Ollama embeddings.

    Args:
        content (str): The input text content.
        chunk_size (int): Size of each text chunk.
        chunk_overlap (int): Overlap between text chunks.

    Returns:
        Chroma: A Chroma vector store containing the encoded text chunks.

    Raises:
        ValueError: If the content is empty.
        RuntimeError: If there is an error during encoding.
    """

    if not isinstance(content, str) or not content.strip():
        raise ValueError("Content must be a non-empty string.")
    
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
                         
    if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
        raise ValueError("chunk_overlap must be a non-negative integer.")

    try:
        # Split the content into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=True
        )
        chunks = text_splitter.create_documents([content])

        # Assign metadata to each chunk
        for chunk in chunks:
            chunk.metadata['relevance_score'] = 1.0  # Default relevance score

        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        vectorstore = Chroma.from_documents(chunks, embeddings)

    except Exception as e:
        raise RuntimeError(f"An error occurred during the encoding process: {str(e)}")

    return vectorstore


def retrieve_context_per_question(question, chunks_query_retriever):
    """
    Retrieve relevant content and unique URLs for a given question using the chunks query retriever.

    Args:
        question (str): The input question.
        chunks_query_retriever (List[str]): List of text chunks to search.

    Returns:
        A tuple containing:
        - A string with the concatenated content of relevant chunks.
        - A list of unique URLs extracted from the relevant chunks.
    """
    docs = chunks_query_retriever.invoke(question)

    context = [doc.page_content for doc in docs]

    return context, list(set([doc.metadata.get("source", "") for doc in docs if "source" in doc.metadata]))


class QuestionAnswerFromContext(BaseModel):
    """
    Model to generate an answer to a query based on a given context.
    
    Attributes:
        answer_based_on_content (str): The generated answer based on the context.
    """
    answer_based_on_content: str = Field(description="Generates an answer to a query based on a given context.")


def create_question_answer_from_context_chain(llm):
    """
    Create a question-answering chain that generates answers based on provided context.

    Args:
        llm: The language model to be used for generating answers.

    Returns:
        A chain that takes context and question as input and produces an answer.
    """
    question_answer_from_context_llm = llm

    question_answer_prompt_template = """
    For the question below, provide a concise and suffice answer based ONLY on the provided context:

    <Context> 
    {context}
    </Context>

    <Question>
    {question}
    </Question>
    """
    question_answer_from_context_prompt = PromptTemplate(
        template=question_answer_prompt_template,
        input_variables=["context", "question"],
    )

    question_answer_from_context_cot_chain = question_answer_from_context_prompt |question_answer_from_context_llm.with_structured_output(
        QuestionAnswerFromContext
    )

    return question_answer_from_context_cot_chain


def answer_question_from_context(
    question: str,
    context: str,
    question_answer_from_context_chain,
) -> str:
    """
    Answer a question using the given context by invoking a chain of reasoning.

    Args:
        question: The question to be answered.
        context: The context to be used for answering the question.

    Returns:
        A dictionary containing the answer, context, and question.
    """
    input_data = {
        "question": question,
        "context": context
    }
    print("Answering the question from the retrieved context...")

    output = question_answer_from_context_chain.invoke(input_data)
    answer = output.answer_based_on_content
    return {"answer": answer, "context": context, "question": question}

def show_context(context: list):
    """
    Display the context in a readable format.

    Args:
        context (list): The list of context items to be displayed.
    """
    for i, c in enumerate(context):
        print(f"Context {i+1}:")
        print(c)
        print("\n" + "-"*80 + "\n")

def read_pdf_to_string(path: str) -> str:
    """
    Reads a PDF document from the specified path and extracts its text content as a string.

    Args:
        path (str): The file path to the PDF document.

    Returns:
        str: The extracted text content of all pages in the PDF.

    The function uses the 'fitz' library (PyMuPDF) to open the PDF document, iterate over each page,
    extract the text, and concatenate it into a single string.
    """
    doc = fitz.open(path)
    content = ""

    for page_num in range(len(doc)):
        page = doc[page_num]
        content += page.get_text()

    return content


def bm25_retrieval(bm25: BM25Okapi, cleaned_texts: List[str], query: str, k: int = 5) -> List[str]:
    """
    Retrieve the top-k relevant documents from a collection using BM25.

    Args:
        bm25 (BM25Okapi): The BM25 API instance for document retrieval.
        cleaned_texts (List[str]): The list of cleaned text documents.
        query (str): The search query.
        k (int, optional): The number of top documents to retrieve. Defaults to 5.

    Returns:
        List[str]: The list of top-k relevant documents.
    """
    # Tokenize the query
    query_tokens = query.split()

    # Get the BM25 scores for the query against the cleaned texts
    scores = bm25.get_scores(query_tokens)

    # Get the indices of the top-k documents
    top_k_indices = np.argsort(scores)[::-1][:k]

    # Retrieve the top-k documents
    top_k_documents = [cleaned_texts[i] for i in top_k_indices]

    return top_k_documents

async def exponential_backoff(attempt):
    """
    Implements exponential backoff with a jitter.
    
    Args:
        attempt: The current retry attempt number.
        
    Waits for a period of time before retrying the operation.
    The wait time is calculated as (2^attempt) + a random fraction of a second.
    """
    # Calculate the wait time with exponential backoff and jitter
    wait_time = (2 ** attempt) + random.uniform(0, 1)
    print(f"Rate limit hit. Retrying in {wait_time:.2f} seconds...")

    # Asynchronously sleep for the calculated wait time
    await asyncio.sleep(wait_time)

async def retry_with_exponential_backoff(coroutine, max_retries=5):
    """
    Retries a coroutine using exponential backoff upon encountering a RateLimitError.
    
    Args:
        coroutine: The coroutine to be executed.
        max_retries: The maximum number of retry attempts.
        
    Returns:
        The result of the coroutine if successful.
        
    Raises:
        The last encountered exception if all retry attempts fail.
    """
    for attempt in range(max_retries):
        try:
            # Attempt to execute the coroutine
            return await coroutine
        except RateLimitError as e:
            # If the last attempt also fails, raise the exception
            if attempt == max_retries - 1:
                raise e

            # Wait for an exponential backoff period before retrying
            await exponential_backoff(attempt)
    # If max retries are reached without success, raise an exception
    raise Exception("Max retries reached")


class EmbeddingProvider(Enum):
    OLLAMA = "ollama"

class ModelProvider(Enum):
    OLLAMA = "ollama"

def get_langchain_embedding_provider(provider: EmbeddingProvider, model_id: str = None):
    """
    Returns an embedding provider based on the specified provider and model ID.

    Args:
        provider (EmbeddingProvider): The embedding provider to use.
        model_id (str): Optional -  The specific embeddings model ID to use .

    Returns:
        A LangChain embedding provider instance.

    Raises:
        ValueError: If the specified provider is not supported.
    """
    if provider == EmbeddingProvider.OLLAMA:
        return OllamaEmbeddings(model=EMBEDDING_MODEL)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")
