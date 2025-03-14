import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Tuple


def read_file(file_path: str) -> Optional[str]:
    """Helper function to read a text file.

    Args:
        file_path (str): The path of the file to read.

    Returns:
        Optional[str]: The file content as a string, or None if an error occurs.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()  # Remove blank spaces
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None  # Ignore failed reads


def load_20news_data(
    directory: str, num_workers: int = 4
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Optimized dataset loader with parallel processing.

    Args:
        directory (str): Path to dataset folder (e.g., 'data/20news-bydate-train').
        num_workers (int): Number of parallel file readers (default: 4).

    Returns:
        Tuple[np.ndarray, np.ndarray, Dict[str, int]]: A tuple containing:
        - NumPy array of loaded document texts.
        - NumPy array of numeric category labels.
        - Dictionary mapping category names to numeric labels.
    """
    texts: List[str] = []  # List of document texts
    labels: List[int] = []  # List of category labels
    categories: Dict[str, int] = {}  # Dictionary mapping category → numeric ID
    category_id: int = 0

    category_folders: List[str] = sorted(
        os.listdir(directory)
    )  # Ensure label consistency

    for category in tqdm(
        category_folders, desc="Processing categories", unit="category"
    ):
        category_path: str = os.path.join(directory, category)

        if not os.path.isdir(category_path):
            continue  # Skip non-folder files

        # Assign category numeric ID
        categories[category] = category_id
        file_paths: List[str] = [
            os.path.join(category_path, f) for f in os.listdir(category_path)
        ]

        # Use ThreadPoolExecutor for parallel file reading
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results: List[Optional[str]] = list(
                tqdm(
                    executor.map(read_file, file_paths),
                    total=len(file_paths),
                    desc=f"Reading {category}",
                    unit="file",
                    leave=False,
                )
            )

        # Filter out None values (failed reads) and update lists
        valid_texts: List[str] = [text for text in results if text is not None]
        texts.extend(valid_texts)
        labels.extend(
            [category_id] * len(valid_texts)
        )  # Faster than appending in a loop

        category_id += 1  # Move to the next category

    # Convert lists to NumPy arrays for speed & memory efficiency
    text_array: np.ndarray = np.array(
        texts, dtype=object
    )  # Object type for variable-length strings
    label_array: np.ndarray = np.array(labels, dtype=np.int32)

    print(
        f"[INFO] Loaded {len(text_array)} documents from '{directory}' across {len(categories)} categories."
    )
    return text_array, label_array, categories
