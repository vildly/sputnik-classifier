# main.py
from typing import Union, Dict, List, Any

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from pylo import get_logger
from dotenv import load_dotenv

from .db import add_one, connect_to_database, get_collection, update_by_id
from .clean import clean_text_to_numpy, process_json_data
from .log_middleware import LogRequestsMiddleware
from .openrouter import chat


# See:
# https://openrouter.ai/models?max_price=0&order=top-weekly
MODELS: List[str] = [
    "google/gemini-2.0-pro-exp-02-05:free",
    "deepseek/deepseek-r1:free",
]


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

    class JobModel(BaseModel):
        raw: Union[Dict[str, Any], List[Any]]
        prompt: str

    @app.post("/")
    def post_root(body: JobModel):
        col = get_collection(db="jobs", collection="v1")
        job = add_one(
            col=col,
            data={"raw": body.raw, "pre": "", "prompt": body.prompt, "models": []},
        )

        # TODO: Preprocess/clean "raw" data before passing it to the models and add it
        # to the db
        pre = process_json_data(
            json_input=body.raw, keys=["title", "description"], language="english"
        )
        numpy_arr = clean_text_to_numpy(pre)
        # Convert the DataFrame 'pre' to a list of dictionaries.
        # pre_serializable = pre.to_dict(orient="records")
        logger.debug(pre)
        logger.debug(numpy_arr)

        # update_by_id(
        #     col=col,
        #     doc_id=job.inserted_id,
        #     update_data={"pre": pre_serializable},
        #     operator="$push",
        # )

        for model in MODELS:
            # TODO: Use "pre" data here concatenated with the prompt
            res = chat(model=model, prompt=body.prompt)
            update_by_id(
                col=col,
                doc_id=job.inserted_id,
                update_data={"models": {"$each": [{model: res}]}},
                operator="$push",
            )

        return JSONResponse(status_code=201, content={"job": str(job.inserted_id)})

except Exception as exc:
    # Logs the error and full traceback
    logger.exception(exc)
