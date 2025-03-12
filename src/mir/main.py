# main.py
from typing import List

import os
import asyncio

from pylo import get_logger
from dotenv import load_dotenv

from clean import process_json_data
from db import (
    add_one,
    connect_to_database,
    find_by_id,
    get_collection,
    update_by_id,
)
from openrouter import chat


# Loading the env before getting logger if any variables have been set for it!
load_dotenv()
logger = get_logger()


class Body:
    def __init__(self, data_id: str, prompt: str, models: List[str]):
        self.data_id = data_id
        self.prompt = prompt
        self.models = models


body = Body(
    data_id="67cad4bf1aa383247994482c",
    prompt="categorize the data into the categories provided",
    models=["deepseek/deepseek-r1:free", "google/gemini-2.0-pro-exp-02-05:free"],
)


async def main():
    try:
        connect_to_database(connection_string=os.getenv("MONGODB_URI"))

        # Prepare a job document to store the results
        jobs_col = get_collection(db="jobs", collection="v1")
        job_doc = add_one(
            col=jobs_col,
            data={"data_id": body.data_id, "prompt": body.prompt},
        )

        # Prepare the data for the models:
        data_col = get_collection(db="data", collection="v1")
        data_doc = find_by_id(col=data_col, doc_id=body.data_id)

        # --- Integrate the data cleaning functions ---
        # Our document's "data" field is a list of records (each with a "description")
        # We pass that list to process_json_data along with the key to process.
        keys_to_clean = ["description"]

        # Process the list of descriptions.
        # If you want to use Swedish stop words, you might pass language="swedish"
        cleaned_df = process_json_data(
            json_input=data_doc.get("data", []), keys=keys_to_clean, language="swedish"
        )

        # Option 1: If you want to join all individual clean texts into one string:
        joined_clean_text = " ".join(cleaned_df["clean_text"].tolist())
        data_doc["clean_text"] = joined_clean_text

        # Option 2: Alternatively, you could keep the cleaned DataFrame records
        # and, for example, send the list instead of a joined string:
        # data_doc["clean_text"] = cleaned_df["clean_text"].tolist()

        # Concatenate the prompt to the document (if desired).
        data_doc["prompt"] = body.prompt

        # --- Create asynchronous tasks to call your model endpoints ---
        # Create tasks that return a tuple (model, result)
        tasks = []
        for model in body.models:
            task = asyncio.create_task(chat(model, str(data_doc)))
            tasks.append(task)

        # As each task completes, update the job document.
        for completed in asyncio.as_completed(tasks):
            model, res = await completed
            logger.info(f"Completed: {model}; adding to the job document")
            update_by_id(
                col=jobs_col,
                doc_id=job_doc.inserted_id,
                update_data={"model": {"$each": [{model: res}]}},
                operator="$push",
            )
    except Exception as exc:
        # Logs the error and full traceback
        logger.exception(exc)


if __name__ == "__main__":
    asyncio.run(main())
