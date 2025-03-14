import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def read_file(file_path):
    """Helper function to read a text file"""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()  # Remove blank spaces
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None  # Ignore failed reads


def load_20news_data(directory, num_workers=4):
    """
    Optimized version of dataset loader with parallel processing.

    Args:
        directory (str): Path to dataset folder (e.g., 'data/20news-bydate-train')
        num_workers (int): Number of parallel file readers.

    Returns:
        texts (NumPy array): Loaded documents.
        labels (NumPy array): Numeric labels.
        categories (dict): Category index mapping.
    """
    texts, labels = [], []
    categories = {}
    category_id = 0

    category_folders = sorted(os.listdir(directory))  # Ensure label consistency

    for category in tqdm(
        category_folders, desc="Processing categories", unit="category"
    ):
        category_path = os.path.join(directory, category)

        if not os.path.isdir(category_path):
            continue  # Skip non-folder files

        # Assign category numeric ID
        categories[category] = category_id
        file_paths = [os.path.join(category_path, f) for f in os.listdir(category_path)]

        # Use ThreadPoolExecutor for parallel file reading
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(
                tqdm(
                    executor.map(read_file, file_paths),
                    total=len(file_paths),
                    desc=f"Reading {category}",
                    unit="file",
                    leave=False,
                )
            )

        # Filter out None values (failed reads) and update lists
        valid_texts = [text for text in results if text is not None]
        texts.extend(valid_texts)
        labels.extend(
            [category_id] * len(valid_texts)
        )  # Faster than appending in a loop

        category_id += 1  # Move to the next category

    # Convert lists to NumPy arrays for speed & memory efficiency
    texts = np.array(texts, dtype=object)  # Object type for variable-length strings
    labels = np.array(labels, dtype=np.int32)

    print(
        f"[INFO] Loaded {len(texts)} documents from '{directory}' across {len(categories)} categories."
    )
    return texts, labels, categories
