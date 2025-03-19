# svm.py
#
# Partially used guide:
# https://developer.ibm.com/tutorials/awb-classifying-data-svm-algorithm-python/
from typing import Dict, Generator, Set, Tuple, List
from string import punctuation
import os
import concurrent.futures
import nltk
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.decomposition import PCA
import matplotlib.colors as colors
import matplotlib.pyplot as plt


def read_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        return file.read()


def load_dir(path: str) -> Tuple[np.ndarray, np.ndarray]:
    print(f"Loading files from {path}")
    print("This may take a while...")
    initial_size: int = 20
    file_paths: np.ndarray = np.empty(initial_size, dtype=object)
    file_labels: np.ndarray = np.empty(initial_size, dtype=object)
    file_count: int = 0

    for root, dirs, _ in os.walk(path):
        for category in dirs:
            category_path: str = os.path.join(root, category)
            for filename in os.listdir(category_path):
                file_path: str = os.path.join(category_path, filename)
                if os.path.isfile(file_path):
                    if file_count >= len(file_paths):
                        # Double the size of arrays when limit is reached
                        file_paths = np.resize(file_paths, file_count * 2)
                        file_labels = np.resize(file_labels, file_count * 2)

                    file_paths[file_count] = file_path
                    file_labels[file_count] = category
                    file_count += 1

    print(f"Initialized {file_count} files")
    # Trim the arrays to the actual length
    file_paths = file_paths[:file_count]
    file_labels = file_labels[:file_count]

    # Use concurrent.futures to read files in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results: List[str] = list(executor.map(read_file, file_paths))

    # Convert results to numpy array
    data: np.ndarray = np.array(results, dtype=object)
    labels: np.ndarray = np.array(file_labels, dtype=object)

    print(f"Loaded {file_count} files")
    print("Returning as (data: np.ndarray, labels: np.ndarray)")
    return data, labels


def clean_texts(texts: np.ndarray, lang: str = "english") -> np.ndarray:
    cleaned_texts: np.ndarray = np.empty_like(texts, dtype=object)

    for ix, text in texts:
        text = text.lower()
        # Remove punctuation
        translator: Dict[int, int | None] = str.maketrans("", "", punctuation)
        text = text.translate(translator)

        # Remove stopwords
        words: List[str] = nltk.word_tokenize(text)
        stop_words: Set[str] = set(nltk.corpus.stopwords.words(lang))
        filtered_words: Generator[str, None, None] = (word for word in words if word not in stop_words)

        cleaned_texts[ix] = " ".join(filtered_words)

    return cleaned_texts


if __name__ == "__main__":
    nltk.download("punkt")
    nltk.download("stopwords")

    # Load train and test datasets
    data: np.ndarray
    labels: np.ndarray
    data, labels = load_dir("./data/20news-bydate-train")

    # Clean the text in DataFrame
    data = clean_texts(data)

    # Create a DataFrame
    df: pd.DataFrame = pd.DataFrame(data={"text": data, "label": labels})
    print(df.head())
