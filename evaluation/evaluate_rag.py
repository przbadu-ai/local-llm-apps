"""
RAG Evaluation Script

This script evaluates the performace of a Retrieval-Augmented Generation (RAG) system
using various metrics from the deepeval library.

Dependencies:
- deepeval
- langchain_ollama
- json

Custom modules:
- helper_functions (for RAG-specific operations)
"""

import json
from typing import List, Dict, Any

# from deepeval import assert_test
from deepeval.metrics import GEval, FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models.base_model import DeepEvalBaseLLM

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.llm_config import *


# Custom Ollama model wrapper for deepeval
class OllamaModel(DeepEvalBaseLLM):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.llm = OllamaLLM(model=model_name, temperature=0)

    def load_model(self):
        return self.llm

    def generate(self, prompt: str) -> str:
        return self.llm.invoke(prompt)

    async def a_generate(self, prompt: str) -> str:
        return self.llm.invoke(prompt)

    def get_model_name(self) -> str:
        return self.model_name

# from helper_functions import (
#     create_question_answer_from_context_chain,
#     answer_question_from_context,
#     retrieve_context_per_question,
# )


def create_deep_eval_test_cases(
    questions: List[str],
    gt_answers: List[str],
    generated_answers: List[str],
    retrieved_documents: List[str],
) -> List[LLMTestCase]:
    """
    Create a list of LLMTestCase objects for evaluation.

    Args:
        questions (List[str]): List of input questions.
        gt_answers (List[str]): List of ground truth answers.
        generated_answers (List[str]): List of generated answers.
        retrieved_documents (List[str]): List of retrieved documents.

    Returns:
        List[LLMTestCase]: List of LLMTestCase objects.
    """
    return [
        LLMTestCase(
            input=question,
            expected_output=gt_answer,
            actual_output=generated_answer,
            retrieval_context=retrieved_document,
        )
        for question, gt_answer, generated_answer, retrieved_document in zip(
            questions, gt_answers, generated_answers, retrieved_documents
        )
    ]


# Initialize Ollama model for evaluation
ollama_eval_model = OllamaModel(model_name=EVALUATION_LLM_MODEL)

# Define Evaluation metrics
correctness_metric = GEval(
    name="Correctness",
    model=ollama_eval_model,
    evaluation_params=[
        LLMTestCaseParams.EXPECTED_OUTPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    evaluation_steps=[
        "Determine whether the actual output is factually correct based on the expected output."
    ],
)


faithfulness_metric = FaithfulnessMetric(
    threshold=0.7, model=ollama_eval_model, include_reason=False
)


relevance_metric = ContextualRelevancyMetric(
    threshold=1, model=ollama_eval_model, include_reason=True
)


def evaluate_rag(retriever, num_questions: int = 5) -> Dict[str, Any]:
    """
    Evaluates a RAG system using predefined test cases and metrics.

    Args:
        retriever: The retriever object to fetch context documents.
        num_questions (int): Number of questions to evaluate.

    Returns:
        Dict[str, Any]: A dictionary containing evaluation results.
    """

    # Initialize LLM
    llm = OllamaLLM(model=LLM_MODEL, temperature=0)

    # Create evaluation prompt
    eval_prompt = PromptTemplate.from_template(
        """
        Evaluate the following retrieval results for the question.

        Question: {question}
        Retrieved Context: {context}

        Rate on a scale of 1-5 (5 being best) for:
        1. Relevance: How relevant is the retrieved context to the question?
        2. Completeness: Does the context contain all necessary information?
        3. Conciseness: Is the retrieved context focused and free of irrelevant information?

        Provide ratings in JSON format:
        """
    )

    # Create evaluation chain
    eval_chain = eval_prompt | llm | StrOutputParser()

    # Generate test questions
    question_gen_prompt = PromptTemplate.from_template(
        "Generate {num_questions} diverse test questions about climate change:"
    )
    question_chain = question_gen_prompt | llm | StrOutputParser()

    questions = question_chain.invoke({"num_questions": num_questions}).split("\n")

    # Evaluate each question
    results = []
    for question in questions:
        # Get retrieved context
        context = retriever.invoke(question)
        context_text = "\n".join([doc.page_content for doc in context])

        # Evaluate results
        eval_result = eval_chain.invoke({"question": question, "context": context_text})
        results.append(eval_result)

    return {
        "questions": questions,
        "results": results,
        "average_score": calculate_average_scores(results),
    }


def calculate_average_scores(results: List[Dict]) -> Dict[str, float]:
    """
    Calculate average scores for relevance, completeness, and conciseness.

    Args:
        results (List[Dict]): List of evaluation results.

    Returns:
        Dict[str, float]: A dictionary with average scores.
    """
    total_relevance = 0
    total_completeness = 0
    total_conciseness = 0
    num_results = len(results)

    for result in results:
        scores = json.loads(result)
        total_relevance += scores.get("Relevance", 0)
        total_completeness += scores.get("Completeness", 0)
        total_conciseness += scores.get("Conciseness", 0)

    return {
        "average_relevance": total_relevance / num_results if num_results > 0 else 0,
        "average_completeness": (
            total_completeness / num_results if num_results > 0 else 0
        ),
        "average_conciseness": (
            total_conciseness / num_results if num_results > 0 else 0
        ),
    }


if __name__ == "__main__":
    # Initialize retriever
    embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMADB_PATH,
        embedding_function=embedding_model,
    )
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})

    # Evaluate RAG system
    evaluation_results = evaluate_rag(retriever, num_questions=5)
    print("Evaluation Results:", evaluation_results)
