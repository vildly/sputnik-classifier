import os
import json
from typing import List, Union

import numpy as np
import pandas as pd
from nltk.downloader import download
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, word_tokenize


# -----------------------------------------------------------------------------
# NLTK Dependency Download
# -----------------------------------------------------------------------------
download_dir = os.getenv("NLTK_DATA")
download(info_or_id="stopwords", download_dir=download_dir)
download(info_or_id="punkt", download_dir=download_dir)
download(info_or_id="punkt_tab", download_dir=download_dir)


# -----------------------------------------------------------------------------
# Text Cleaning Helpers
# -----------------------------------------------------------------------------
def _remove_stop_words(text: str, spacer: str = "", language: str = "english") -> str:
    """
    Tokenizes the text and removes stop words.
    """
    stop_words = set(stopwords.words(language))
    tokens = word_tokenize(text=text, language=language)
    return spacer.join([w for w in tokens if w.lower() not in stop_words])


def _remove_symbols(text: str, spacer: str = "") -> str:
    """
    Tokenizes text using a regex that keeps only word characters.
    """
    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text)
    return spacer.join(tokens)


# -----------------------------------------------------------------------------
# Main Processing Function
# -----------------------------------------------------------------------------
def process_json_data(
    json_input: Union[str, dict, list], keys: List[str], language: str = "english"
) -> pd.DataFrame:
    """
    Processes JSON data (from a file path, a JSON-string, dict, or list) by:
      - Flattening it via pandas.json_normalize.
      - Filtering only the specified keys.
      - Cleaning the text (lowercase, removing symbols, and stop words).

    Returns:
      A pandas DataFrame with a new column 'clean_text' including the cleaned text.
    """
    if not keys:
        raise ValueError("No keys provided.")

    # Load JSON data from file, JSON string, or pass-through dict/list.
    if isinstance(json_input, str):
        if os.path.exists(json_input):
            with open(json_input, "r", encoding="utf-8") as f:
                json_data = json.load(f)
        else:
            try:
                json_data = json.loads(json_input)
            except Exception as e:
                raise ValueError(
                    "Input string is neither a valid file path nor valid JSON."
                ) from e
    elif isinstance(json_input, (dict, list)):
        json_data = json_input

    # Flatten the JSON structure.
    df = pd.json_normalize(json_data)

    def process_row(row: pd.Series) -> str:
        texts = []
        for key in keys:
            if key not in row:
                continue

            value = row[key]

            # Only process values if they are scalar (str, int, or float).
            # If not scalar (e.g. a Series, list, or dict), skip.
            if not isinstance(value, (str, int, float)):
                continue

            # Convert the value to a string and process.
            text = str(value).lower()
            text = _remove_symbols(text, spacer=" ")
            text = _remove_stop_words(text, spacer=" ", language=language)
            if text.strip():
                texts.append(text.strip())
        return " ".join(texts).strip() if texts else ""

    # Apply the cleaning function to each row.
    df["clean_text"] = df.apply(process_row, axis=1)

    # Use .loc[] to ensure a DataFrame is returned.
    filtered_df: pd.DataFrame = df.loc[df["clean_text"] != ""].reset_index(drop=True)
    return filtered_df


# -----------------------------------------------------------------------------
# Optional: Convert Clean Text to a NumPy Array
# -----------------------------------------------------------------------------
def clean_text_to_numpy(df: pd.DataFrame) -> np.ndarray:
    """
    Returns a NumPy array of the clean_text column.
    """
    if "clean_text" not in df.columns:
        raise ValueError("DataFrame does not contain a 'clean_text' column.")
    return df["clean_text"].to_numpy()
