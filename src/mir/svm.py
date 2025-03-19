from typing import Tuple, List
import os
import concurrent.futures
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

    # Trim the arrays to the actual length
    file_paths = file_paths[:file_count]
    file_labels = file_labels[:file_count]

    # Use concurrent.futures to read files in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results: List[str] = list(executor.map(read_file, file_paths))

    # Convert results to numpy array
    data: np.ndarray = np.array(results, dtype=object)
    labels: np.ndarray = np.array(file_labels, dtype=object)

    return data, labels


if __name__ == "__main__":
    # Load train and test datasets
    data: np.ndarray
    labels: np.ndarray
    data, labels = load_dir("./data/20news-bydate-train")
    print(f"Data Size: {len(data)}")
    print(f"Labels Size: {len(labels)}")

    # Create a DataFrame
    df: pd.DataFrame = pd.DataFrame(data={"text": data, "label": labels})
    print(df.head())
