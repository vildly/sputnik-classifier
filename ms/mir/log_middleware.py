# log_middleware.py
import time
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from fastapi.responses import JSONResponse
from pylo import get_logger

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


class LogRequestsMiddleware(BaseHTTPMiddleware):
    """
    Custom middleware to log the request details and response time.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
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

        # Log response info and process time
        logger.info(f"Status: {response.status_code}")
        logger.info(f"Time: {process_time:.2f}ms")

        return response
