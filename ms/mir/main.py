# main.py
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import nltk
from pylo import get_logger

logger = get_logger()


def _get_dependencies() -> None:
    # Important as NLTK searches in these directories:
    # - $HOME/nltk_data
    # - $HOME/<path-to-project>/.venv/nltk_data
    # - $HOME/<path-to-project>/.venv/share/nltk_data
    # - $HOME/<path-to-project>/.venv/lib/nltk_data
    # - /usr/share/nltk_data
    # - /usr/local/share/nltk_data
    # - /usr/lib/nltk_data
    # - /usr/local/lib/nltk_data
    download_dir = "./.venv/lib/nltk_data"
    nltk.download("stopwords", download_dir=download_dir)
    nltk.download("punkt_tab", download_dir=download_dir)
    nltk.download("punkt", download_dir=download_dir)


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
    _get_dependencies()
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
    def hello():
        return JSONResponse(status_code=200, content={"hello": "world"})


except Exception as exc:
    logger.exception(exc)  # Logs the error and full traceback
