import os
import httpx
from pylo import get_logger

logger = get_logger()


async def chat(model: str, prompt: str) -> str:
    logger.info(f"({model}) OR-Prompt: {prompt}")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('OR_SECRET')}",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            },
        )

    content_type = response.headers.get("Content-Type", "")
    if "application/json" in content_type:
        body = response.json()
    else:
        body = response.text

    logger.info(f"({model}) OR-Status: {response.status_code}")
    if not response.status_code == 200:
        message = f"({model}) OR-Exc: {body}"
        logger.warning(message)
        raise ValueError(message)

    return body
