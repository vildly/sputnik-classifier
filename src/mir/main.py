from typing import List
import os
import asyncio
import json

from pylo import get_logger
from dotenv import load_dotenv

from db import (
    add_one,
    connect_to_database,
    find_by_id,
    get_collection,
    update_by_id,
)
from openrouter import openrouter_chat, openai_chat

# Load environment variables before getting the logger
load_dotenv()
logger = get_logger()


class Config:
    def __init__(self, data_id: str, prompt: str, openrouter_models: List[str], openai_models: List[str]):
        self.data_id = data_id
        self.prompt = prompt
        self.openrouter_models = openrouter_models
        self.openai_models = openai_models


config = Config(
    data_id="67e2d15667e690e63ad74a5c",
    prompt="categorize the data into the categories provided",
    openrouter_models=["google/gemini-2.0-flash-001"],
    openai_models=["gpt-4o"],
)


async def do_task(openrouter_models, openai_models, jobs_col, query, job_doc):
    tasks = []
    for model in openrouter_models:
        logger.info(f"Starting model '{model}'")
        # 1) New Openrouter call
        task = asyncio.create_task(openrouter_chat(model, query))
        tasks.append(task)

    for model in openai_models:
        logger.info(f"Starting model '{model}'")
        # 2) New OpenAI call
        task_oa = asyncio.create_task(openai_chat(model, query))
        tasks.append(task_oa)

    # As each task completes, update the job document.
    for completed in asyncio.as_completed(tasks):
        model, res = await completed
        update_by_id(
            col=jobs_col,
            doc_id=job_doc.inserted_id,
            update_data={"model": {"$each": [{model: res}]}},
            operator="$push",
        )


async def process_batch(batch, base_query, categories, jobs_col, job_doc):
    """Process the accumulated batch if it's not empty."""
    if not batch:
        return
    input_data = json.dumps(batch, ensure_ascii=False)
    full_query = base_query.format(input_data=input_data, categories=categories)
    await do_task(config.openrouter_models, config.openai_models, jobs_col, full_query, job_doc)


async def main(batch_len: int = 10000):
    """
    Main function to process the data in batches.

    :param batch_len: The maximum length of the batch.

    :return: None
    """
    cnt_len = 0
    data_len = 0

    connect_to_database(connection_string=os.getenv("MONGODB_URI"))

    data_col = get_collection(db="data", collection="v1")
    data_doc = find_by_id(col=data_col, doc_id=config.data_id)
    articles_length = len(data_doc["articles"])

    categories = ", ".join(data_doc["categories"])

    # Static base query with a placeholder for input_data and categories
    base_query = """
    You are an expert in text categorization. Please follow the instructions carefully.

    I will provide you with a string that represents data in a JSON-like format. This string may resemble a Python dictionary, but your task is to parse it and produce valid JSON. Each input text (string value) must be categorized according to the categories I provide. You must choose only one category from the list for each text, and you must not invent new categories.

    Your tasks are:
    1. Parse the string to identify the keys and values.
    2. Convert any single quotes (') used for keys or string values into double quotes (") to comply with strict JSON rules.
    3. Validate that the structure is a valid JSON object (no trailing commas, all keys and values properly quoted, etc.).
    4. For each key in the JSON, assign exactly one of the categories provided. Do not use any category that is not in the list.
    5. It is crucial that your response is only JSON and nothing else. No commentary is needed.

    The expected JSON format is as follows:
    {{
      "key1": "categoryChosenFromList",
      "key2": "categoryChosenFromList",
      ...
    }}

    For example, if given the string:
    {{
      "0": "Some short text to categorize",
      "1": "Another text to categorize"
    }}

    You should output valid JSON with the same keys but where each value is ONE category from the list.

    IMPORTANT: Only use ONE OF the categories provided, and do not add or modify categories beyond those listed. Choose the most appropriate category for each text.

    Categories you must choose from: {categories}

    Here is the JSON string to be processed:
    {input_data}  # This is the placeholder
    """

    # Dictionary to accumulate input data
    input_dict = {}

    jobs_col = get_collection(db="jobs", collection="v1")
    job_doc = add_one(
        col=jobs_col,
        data={"data_id": config.data_id, "prompt": config.prompt, "model": []},
    )
    if not job_doc.inserted_id:
        logger.error("Job document not created")
        return

    while cnt_len < articles_length:
        for i in range(articles_length):
            current_article_text = data_doc["articles"][i]["user_input"]
            data_len += len(current_article_text)
            cnt_len += 1

            if data_len > batch_len:
                logger.info(f"Max length reached on index {i}")
                data_len = 0

                # Process the current batch.
                await process_batch(input_dict, base_query, categories, jobs_col, job_doc)
                logger.info(f"Processed batch ending at index {i}")

                # Reset input_dict for the next batch.
                input_dict = {}
            else:
                # Populate the dictionary.
                input_dict[str(i)] = current_article_text

    # NOTE:
    # Process any remaining data that did not reach the MAX_DATA_LENGTH threshold.
    # This ensures that no accumulated articles are left unprocessed.
    # Does nothing if the input_dict is empty.
    logger.info("Processing the remaining batch")
    await process_batch(input_dict, base_query, categories, jobs_col, job_doc)
    logger.info("Processed the remaining batch")


if __name__ == "__main__":
    asyncio.run(main())
