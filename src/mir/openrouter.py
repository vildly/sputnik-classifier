import os
import httpx
from pylo import get_logger

logger = get_logger()


async def chat(model: str, prompt: str) -> tuple[str, str]:
    """
    Sends a chat request to the Openrouter API for the given model and prompt.

    Returns:
        A tuple (model, result) where 'result' is either the API response (JSON or text)
        or an error message if an exception occurred or if the response status is not 200.
    """
    logger.info(f"({model}) OR-Prompt: {prompt}")

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as client:
            response = await client.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.getenv('OR_SECRET')}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
    except Exception as exc:
        error_msg = f"({model}) Exception: {exc}"
        logger.warning(error_msg)
        return model, error_msg

    logger.info(f"({model}) OR-Status: {response.status_code}")

    # Get response body—if JSON, parse it; otherwise, return text.
    content_type = response.headers.get("Content-Type", "")
    if "application/json" in content_type:
        body = response.json()
    else:
        body = response.text

    # Check for non-200 response and return an error message if needed.
    if response.status_code != 200:
        error_msg = f"({model}) Error {response.status_code}: {body}"
        logger.warning(error_msg)
        return model, error_msg

    return model, body
