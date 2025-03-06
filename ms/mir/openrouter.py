import os
import requests
import json

from pylo import get_logger


logger = get_logger()


def request(model: str, prompt: str) -> str:
    logger.info(model)
    logger.info(prompt)

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

    logger.info(response.status_code)
    if not response.ok:
        body = response.text
        logger.warning(body)
        raise ValueError(f"OR: {body}")

    json_serialized_response = json.dumps(response.json(), indent=4)
    logger.info(json_serialized_response)
    return json_serialized_response
