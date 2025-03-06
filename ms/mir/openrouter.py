import os
import requests
import json

from pylo import get_logger
from dotenv import load_dotenv


logger = get_logger()
load_dotenv()

if __name__ == "__main__":
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('OR_SECRET')}",
            # "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
            # "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
        },
        data=json.dumps({
            "model": "google/gemini-2.0-pro-exp-02-05:free",
            "messages": [{"role": "user", "content": "What is the meaning of life?"}],
        }),
    )
    logger.info(response.status_code)
    # Pretty-print the JSON to the console
    pretty_json = json.dumps(response.json(), indent=4)
    logger.info(pretty_json)
