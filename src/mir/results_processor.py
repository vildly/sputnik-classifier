from abc import ABC, abstractmethod
import json
import re
import os
import asyncio
from dotenv import load_dotenv
from pylo import get_logger
from ragas_test import evaluate_data
from db import (
    add_one,
    connect_to_database,
    find_by_id,
    get_collection,
)

# Load environment variables
load_dotenv()
logger = get_logger()


class Config:
    def __init__(self, job_id: str, openrouter_models: list, openai_models: list):
        self.job_id = job_id
        self.openrouter_models = openrouter_models
        self.openai_models = openai_models


config = Config(
    job_id="67e2ffcb1f64f880441adec4",
    openrouter_models=["google/gemini-2.0-flash-001"],
    openai_models=["gpt-4o"],
)


def strip_markdown_fences(text):
    """Remove Markdown code fences from text."""
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text


class ExtractorInterface(ABC):
    @abstractmethod
    def extract(self, model_result, model_key: str) -> str:
        """
        Extracts the JSON content string from a model's result.
        """
        pass


class OpenRouterExtractor(ExtractorInterface):
    def extract(self, model_result, model_key: str) -> str:
        """
        Extraction function for OpenRouter responses.
        Expected structure:
          { "model-name": { "choices": [ { "message": { "content": "results" } } ] } }
        """
        return model_result.get("choices", [{}])[0].get("message", {}).get("content", "")


class OpenAIExtractor(ExtractorInterface):
    def extract(self, model_result, model_key: str) -> str:
        """
        Extraction function for OpenAI responses.
        Expected structure:
          { "model-name": "results" }
        """
        return model_result if isinstance(model_result, str) else model_result.get(model_key, "")


def process_model_results(job_doc, model_key: str, articles: list, extractor: ExtractorInterface) -> list:
    """
    Process a single model's results using the provided extractor function.
    Returns a list of results dictionaries.
    """
    job_id = job_doc.get("_id")
    # Collect all entries matching this model_key from the job document.
    model_results_list = [entry.get(model_key) for entry in job_doc.get("model", []) if model_key in entry]

    if not model_results_list:
        logger.warning(f"({model_key}) No results found for job ID: {job_id}")
        return []

    all_results = []
    for model_result in model_results_list:
        # Extract the JSON string from the result structure.
        content = extractor.extract(model_result, model_key)
        try:
            parsed_content = json.loads(strip_markdown_fences(content))
            logger.info(f"({model_key}) Job ID: {job_id}, parsed content: {parsed_content}")
        except json.JSONDecodeError as e:
            logger.error(f"({model_key}) Job ID: {job_id}, failed to parse JSON: {e}")
            continue

        # NOTE:
        # Iterate over parsed keys; each key is expected to be an index.
        for key, category in parsed_content.items():
            try:
                index = int(key)
            except ValueError:
                logger.error(f"({model_key}) Job ID: {job_id}, encountered non-integer key: {key}")
                continue

            if index >= len(articles):
                logger.warning(f"({model_key}) Job ID: {job_id}, index {index} out of range for articles, skipping")
                logger.warning("The model may have returned too many results or hallucinated")
                continue

            article = articles[index]
            all_results.append(
                {
                    "user_input": article["user_input"],
                    "reference": article["reference"],
                    "agent_response": category,
                }
            )

    return all_results


async def process_and_store_results_combined(openrouter_models: list, openai_models: list):
    """
    Process all job entries and combine OpenRouter and OpenAI results into the same DB object.
    """
    jobs_col = get_collection(db="jobs", collection="v1")
    results_col = get_collection(db="results", collection="v1")
    data_col = get_collection(db="data", collection="v1")

    # No need to check if the job exists as the method will raise an error if it doesn't.
    job_doc = find_by_id(col=jobs_col, doc_id=config.job_id)

    job_id = job_doc.get("_id")
    data_id = job_doc.get("data_id")
    if not data_id:
        logger.error(f"Job ID: {job_id} is missing the data_id field")
        return

    data_doc = find_by_id(col=data_col, doc_id=data_id)

    articles = data_doc.get("articles", [])
    results_structure = {"models": {}}

    # Process OpenRouter models.
    for model_key in openrouter_models:
        logger.info(f"({model_key}) Starting evaluation (OpenRouter)")
        results = process_model_results(job_doc, model_key, articles, OpenRouterExtractor())
        if results:
            ragas_avg, ragas_metrics = evaluate_data(results)
            results_structure["models"][model_key] = {
                "model_id": model_key,
                "results": results,
                "ragas_avg": ragas_avg,
                "ragas_metrics": ragas_metrics,
            }

    # Process OpenAI models.
    for model_key in openai_models:
        logger.info(f"({model_key}) Starting evaluation (OpenAI)")
        results = process_model_results(job_doc, model_key, articles, OpenAIExtractor())
        if results:
            ragas_avg, ragas_metrics = evaluate_data(results)
            results_structure["models"][model_key] = {
                "model_id": model_key,
                "results": results,
                "ragas_avg": ragas_avg,
                "ragas_metrics": ragas_metrics,
            }

    # Only add a result document to the DB if any models produced results.
    if results_structure["models"]:
        add_result = add_one(col=results_col, data=results_structure)
        logger.info(f"(DB) Added result document with ID: {add_result.inserted_id} for Job ID: {job_id}")


async def main():
    connect_to_database(connection_string=os.getenv("MONGODB_URI"))
    await process_and_store_results_combined(config.openrouter_models, config.openai_models)


if __name__ == "__main__":
    asyncio.run(main())
