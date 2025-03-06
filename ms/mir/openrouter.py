import os
import requests
import json

from pylo import get_logger


logger = get_logger()


def chat(model: str, prompt: str) -> str:
    logger.info(f"OR-Model: {model}")
    logger.info(f"OR-Prompt: {prompt}")

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('OR_SECRET')}",
            # "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
            # "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
        },
        data=json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }),
    )

    content_type = response.headers.get("Content-Type", "")
    if "application/json" in content_type:
        body = response.json()
    else:
        body = response.text

    logger.info(f"OR-Status: {response.status_code}")
    if not response.ok:
        message = f"OR-Exc:{body}"
        logger.warning(message)
        raise ValueError(message)

    return body
