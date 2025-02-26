# main.py
import os
import asyncio
import nltk
from dotenv import load_dotenv
from pylo import get_logger

from db import add_one, connect_to_database, find_one, get_collection
from rabbitmq import listen


logger = get_logger()


def get_dependencies() -> None:
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


if __name__ == "__main__":
    try:
        load_dotenv()
        connect_to_database(os.getenv("MONGODB_URI"))

        # TESTING PURPOSES:
        col = get_collection("db", "preprocessed")
        add_one_res = add_one(col=col, data={"data": "hello world"})
        find_one_res = find_one(col=col, query={"data": "hello world"})
        logger.debug(add_one_res)
        logger.debug(find_one_res)

        asyncio.run(listen())
    except Exception as exc:
        logger.exception(exc)
