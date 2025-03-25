import os
import httpx
from openai import AsyncOpenAI, APIError
from pylo import get_logger

logger = get_logger()


async def openrouter_chat(model: str, prompt: str) -> tuple[str, str]:
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


async def openai_chat(model: str, prompt: str) -> tuple[str, str]:
    """
    Sends a chat request to the OpenAI API for the given model and prompt.

    Returns:
        A tuple (model, result) where 'result' is either the generated text output
        or an error message if an exception occurred.
    """
    logger.info(f"({model}) OA-Prompt: {prompt}")

    # Initialize the async client with your OpenAI API key
    client = AsyncOpenAI(api_key=os.getenv("OA_SECRET"))

    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
    except APIError as exc:
        # Handle OpenAI-specific exceptions
        error_msg = f"({model}) OpenAI API error: {exc}"
        logger.warning(error_msg)
        return model, error_msg
    except Exception as exc:
        # Catch any other exceptions
        error_msg = f"({model}) Unexpected exception: {exc}"
        logger.warning(error_msg)
        return model, error_msg

    # We don't get a direct HTTP status code in the official library, so just log success.
    logger.info(f"({model}) OA-Status: Success")

    # Extract the model's generated text; ensure we return it as a string.
    try:
        output_text = completion.choices[0].message.content
    except (AttributeError, IndexError) as exc:
        error_msg = f"({model}) Error parsing response: {exc}"
        logger.warning(error_msg)
        return model, error_msg

    # Ensure the result is text
    body = str(output_text or "")

    logger.info(f"({model}) OA-Response: {body}")
    return model, body
