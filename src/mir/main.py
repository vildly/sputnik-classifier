# main.py
from typing import List

import os
import json
import asyncio

from bson import json_util

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from pylo import get_logger
from dotenv import load_dotenv

from .db import (
    add_one,
    connect_to_database,
    find_by_id,
    get_collection,
    update_by_id,
)
from .openrouter import chat
from .log_middleware import LogRequestsMiddleware


# Loading the env before getting logger if any variables have been set for it!
load_dotenv()
logger = get_logger()


try:
    connect_to_database(connection_string=os.getenv("MONGODB_URI"))
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(LogRequestsMiddleware)

    @app.get("/")
    def get_root(id: str):
        job_col = get_collection(db="jobs", collection="v1")
        job_doc = find_by_id(col=job_col, doc_id=id)
        return JSONResponse(
            status_code=200, content=json.loads(json_util.dumps(job_doc))
        )

    class JobModel(BaseModel):
        data_id: str
        prompt: str
        models: List[str]

    @app.post("/")
    async def post_root(body: JobModel):
        jobs_col = get_collection(db="jobs", collection="v1")
        job_doc = add_one(
            col=jobs_col,
            data={"data_id": body.data_id, "prompt": body.prompt},
        )

        # Prepare the data for the models:
        data_col = get_collection(db="data", collection="v1")
        data_doc = find_by_id(col=data_col, doc_id=body.data_id)
        # Concatenate the prompt to the data sent to the model
        data_doc["prompt"] = body.prompt

        # Create tasks that return a tuple (model, result)
        tasks = []
        for model in body.models:
            task = asyncio.create_task(chat(model, str(data_doc)))
            tasks.append(task)

        # As each task completes, update the job document.
        for completed in asyncio.as_completed(tasks):
            model, res = await completed
            update_by_id(
                col=jobs_col,
                doc_id=job_doc.inserted_id,
                update_data={"model": {"$each": [{model: res}]}},
                operator="$push",
            )

        return JSONResponse(status_code=201, content={"job": str(job_doc.inserted_id)})

except Exception as exc:
    # Logs the error and full traceback
    logger.exception(exc)
