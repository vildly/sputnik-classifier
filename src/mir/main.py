# main.py
from typing import List
import re
import os
import asyncio

from pylo import get_logger
from dotenv import load_dotenv
import json

from db import (
    add_one,
    connect_to_database,
    find_by_id,
    get_collection,
    update_by_id,
)
from openrouter import openrouter_chat, openai_chat


# Loading the env before getting logger if any variables have been set for it!
load_dotenv()
logger = get_logger()


class Body:
    def __init__(self, data_id: str, prompt: str, openrouter_models: List[str], openai_models: List[str]):
        self.data_id = data_id
        self.prompt = prompt
        self.openrouter_models = openrouter_models
        self.openai_models = openai_models


body = Body(
    data_id="67dd621c95dba9ac576eb821",
    prompt="categorize the data into the categories provided",
    openrouter_models=["meta-llama/llama-3.3-70b-instruct:free", "google/gemini-2.0-pro-exp-02-05:free"],
    openai_models=["gpt-4o"],
)


async def do_task(openrouter_models, openai_models, jobs_col, query, job_doc):
    tasks = []
    for model in openrouter_models:
        # 1) New Openrouter call
        task = asyncio.create_task(openrouter_chat(model, query))
        tasks.append(task)

    for model in openai_models:
        # 2) New OpenAI call
        task_oa = asyncio.create_task(openai_chat(model, query))
        tasks.append(task_oa)

    # As each task completes, update the job document.
    for completed in asyncio.as_completed(tasks):
        model, res = await completed
        print(f"Got answer: {res}; ")
        logger.info(f"Completed: {model}; adding to the job document")
        result = update_by_id(
            col=jobs_col,
            doc_id=job_doc.inserted_id,
            update_data={"model": {"$each": [{model: res}]}},
            operator="$push",
        )
        # Optionally log the result to debug
        logger.info(f"Update result: {result.modified_count if result else 'No result logged'}")


def strip_markdown_fences(text):

    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text


def load_json_data(job_id: str) -> str:

    string = ""
    col = get_collection(db="jobs", collection="v1")
    data_doc = find_by_id(col=col, doc_id=job_id)

    for i in range(len(data_doc["model"])):
        for model in body.openrouter_models + body.openai_models:
            try:
                if data_doc["model"][i][model]:

                    string += strip_markdown_fences(data_doc["model"][i][model]["choices"][0]["message"]["content"]) + ",\n"

            except KeyError:
                pass  # Just catch key error

    print(string)

    return ""


async def main():
    MAX_DATA_LENGTH = 100000
    cnt_len = 0
    data_length = 0

    connect_to_database(connection_string=os.getenv("MONGODB_URI"))

    data_col = get_collection(db="data", collection="v1")
    data_doc = find_by_id(col=data_col, doc_id=body.data_id)
    articles_length = len(data_doc["articles"])

    categories = ", ".join(data_doc["categories"])

    # Static base query with a placeholder for input_data
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
        data={"data_id": body.data_id, "prompt": body.prompt, "model": []},
    )

    while cnt_len < articles_length:
        for i in range(articles_length):

            current_article_text = data_doc["articles"][i]["user_input"]
            data_length += len(current_article_text)
            cnt_len += 1

            if data_length > MAX_DATA_LENGTH:
                print(f"Max length reached on index {i}")
                data_length = 0

                # Convert the dictionary to a JSON string
                input_data = json.dumps(input_dict, ensure_ascii=False)

                # Create the full query by inserting the current JSON data
                full_query = base_query.format(input_data=input_data, categories=categories)
                await do_task(body.openrouter_models, body.openai_models, jobs_col, full_query, job_doc)
                print(full_query)
                print("HERE>>>>>>>>>>>>>>> " + str(i))
                # Reset input_dict for the next batch
                input_dict = {}

            else:
                # Populate the dictionary
                input_dict[str(i)] = current_article_text

    # Process any remaining data at the end
    if input_dict:
        input_data = json.dumps(input_dict, ensure_ascii=False)
        full_query = base_query.format(input_data=input_data, categories=categories)
        # print(full_query)


if __name__ == "__main__":
    asyncio.run(main())
