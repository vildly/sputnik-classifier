from typing import Union
import re
import numpy as np
import nltk
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Load tokenizer once (instead of per document)
nltk.download("punkt")
TOKENIZER = nltk.tokenize.TreebankWordTokenizer()


def clean_text(texts: np.ndarray) -> np.ndarray:
    """
    Cleans an array of input text by removing headers and non-alphanumeric characters.

    Args:
        texts (np.ndarray): A NumPy array of text strings.

    Returns:
        np.ndarray: A NumPy array of cleaned text strings.
    """
    if texts.dtype != object:
        raise ValueError("The NumPy array must contain string objects.")

    # Precompile regex patterns for efficiency
    header_pattern = re.compile(
        r"^(From|Subject|Organization|Lines|In-Reply-To):.*$", re.MULTILINE
    )
    non_alphanum_pattern = re.compile(r"[^a-zA-Z0-9\s]")

    # Vectorized text cleaning
    cleaned_texts = np.empty_like(texts, dtype=object)

    for i in tqdm(range(len(texts)), desc="Cleaning Text", unit="doc"):
        text = texts[i]
        text = header_pattern.sub("", text)  # Remove headers
        text = non_alphanum_pattern.sub("", text)  # Remove non-alphanumeric characters
        cleaned_texts[i] = text.lower()  # Convert to lowercase

    return cleaned_texts


def tokenize_text(texts: Union[np.ndarray, list], num_workers: int = 8) -> np.ndarray:
    """
    Efficiently tokenizes a NumPy array (or list) of text strings using multi-threading.

    Args:
        texts (np.ndarray or list): Array of input **pre-cleaned** text strings.
        num_workers (int): Number of parallel CPU threads.

    Returns:
        np.ndarray: A NumPy array of tokenized word lists for each input string.
    """
    if isinstance(texts, list):
        texts = np.array(texts, dtype=object)  # Convert list to NumPy array

    if texts.dtype != object:
        raise ValueError("The NumPy array must contain string objects.")

    # Parallelize tokenization using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        tokenized_texts = list(
            tqdm(
                executor.map(TOKENIZER.tokenize, texts),
                total=len(texts),
                desc="Tokenizing Texts",
                unit="doc",
            )
        )

    return np.array(tokenized_texts, dtype=object)
