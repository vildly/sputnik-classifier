import os
import nltk
from dotenv import load_dotenv

from db import connect_to_database


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
    load_dotenv()
    connect_to_database(os.getenv("MONGODB_URI"))

    # col = get_collection("db", "preprocessed")
    # print(add_one(col=col, data={"data": "hello world"}))
    # print(find_one(col=col, query={"data": "hello world"}))
