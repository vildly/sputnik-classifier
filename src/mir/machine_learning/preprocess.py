import re
import numpy as np
from nltk.downloader import download
from nltk.tokenize import word_tokenize
from typing import List

# Download necessary NLTK package
download("punkt")


def clean_text(text: str) -> str:
    """
    Removes email headers and performs basic text cleaning.

    Args:
        text (str): The input text.

    Returns:
        str: Cleaned text with headers removed and non-alphanumeric characters filtered out.
    """
    if not isinstance(text, str):
        return ""

    # Use NumPy array for efficient operations
    text_array = np.array(list(text))  # Convert text to NumPy array for fast processing

    # Remove email headers using regex
    text_str = "".join(text_array)
    text_str = re.sub(
        r"^(From|Subject|Organization|Lines|In-Reply-To):.*$",
        "",
        text_str,
        flags=re.MULTILINE,
    )

    # Remove non-alphanumeric characters (keeping spaces)
    text_str = re.sub(r"[^a-zA-Z0-9\s]", "", text_str)

    # Convert to lowercase
    return text_str.lower()


def tokenize_text(texts: List[str]) -> List[List[str]]:
    """
    Tokenizes a list of text strings into word tokens using NumPy for optimization.

    Args:
        texts (List[str]): A list of input text strings.

    Returns:
        List[List[str]]: A list of tokenized word lists for each input string.
    """
    cleaned_texts = np.vectorize(clean_text)(texts)  # Vectorized text cleaning
    return [word_tokenize(text) for text in cleaned_texts]
