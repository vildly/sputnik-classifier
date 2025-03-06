# main.py
import os
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from db import connect_to_database
from clean import get_dependencies
from openrouter import request
from pylo import get_logger
from dotenv import load_dotenv


# Loading the env before getting logger if any variables have been
# set for it!
load_dotenv()
logger = get_logger()


def _process_error(exc) -> JSONResponse:
    """
    Process the error and return the appropriate response
    """
    # Log the error with stack trace
    logger.error(f"server_exc={exc}")
    logger.exception(exc)

    # Return a JSON response:
    if isinstance(exc, ValueError):
        # Perhaps return a static string here?
        return JSONResponse(status_code=400, content={"error": str(exc)})

    # Default response for any other error
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


try:
    # get_dependencies()
    # connect_to_database(connection_string=os.getenv("MONGODB_URI"))
    app = FastAPI()

    # Cross-Origin Resource Sharing (CORS) Middleware
    #
    # Adjust the "allow_origins" list to allow requests from specific domains,
    # as * (all) allows requests from any domain.
    #
    # Update any methods and headers as needed!
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """
        Middleware to log requests and responses using "HTTP" log level
        """
        start_time = time.time()

        # Log request info
        client_ip = request.client.host if request.client else "missing"
        logger.info(f"IP: {client_ip}")
        logger.info(f"Method: {request.method} {request.url.path}")

        try:
            # Process the request
            response = await call_next(request)
        except Exception as exc:
            return _process_error(exc)

        # Calculate processing time
        process_time = (time.time() - start_time) * 1000
        formatted_process_time = f"{process_time:.2f}ms"

        # Log response info and process time
        logger.info(f"Status: {response.status_code}")
        logger.info(f"Time: {formatted_process_time}")

        return response

    @app.get("/")
    def get_root():
        return JSONResponse(status_code=200, content={"hello": "world"})

    class RootModel(BaseModel):
        model: str
        prompt: str

    @app.post("/")
    def post_root(body: RootModel):
        res = request(model=body.model, prompt=body.prompt)
        return JSONResponse(status_code=201, content={"job": res})


except Exception as exc:
    logger.exception(exc)  # Logs the error and full traceback
