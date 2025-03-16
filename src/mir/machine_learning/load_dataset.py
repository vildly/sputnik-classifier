import os
import json
import numpy as np
import math
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Tuple


def read_file(file_path: str) -> Optional[str]:
    """Reads a text file and returns its content."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def load_20news_data(
    directory: str, num_workers: int = 4
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], np.ndarray]:
    """
    Loads the 20 Newsgroups dataset efficiently using NumPy.

    Args:
        directory (str): Path to dataset folder (e.g., 'data/20news-bydate-train').
        num_workers (int): Number of parallel file readers (default: 4).

    Returns:
        Tuple[np.ndarray, np.ndarray, Dict[str, int], np.ndarray]:
        - NumPy array of document texts.
        - NumPy array of numeric category labels.
        - Dictionary mapping category names to numeric labels.
        - NumPy array of category names per document.
    """
    category_folders = np.array(sorted(os.listdir(directory)))
    category_mapping: Dict[str, int] = {
        category: i for i, category in enumerate(category_folders)
    }

    texts = np.array([], dtype=object)
    labels = np.array([], dtype=np.int32)
    categories = np.array([], dtype=object)

    for category, category_id in tqdm(
        category_mapping.items(), desc="Processing categories", unit="category"
    ):
        category_path = os.path.join(directory, category)
        if not os.path.isdir(category_path):
            continue  # Skip non-folder files

        # Read all files in parallel
        file_paths = np.array([
            os.path.join(category_path, f) for f in os.listdir(category_path)
        ])

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = np.array(list(executor.map(read_file, file_paths)), dtype=object)

        # Remove None values before concatenation
        valid_mask = np.array([
            x is not None for x in results
        ])  # Boolean mask for valid texts
        valid_texts = results[valid_mask]

        if len(valid_texts) > 0:
            texts = (
                np.concatenate((texts, valid_texts)) if texts.size > 0 else valid_texts
            )
            labels = np.concatenate((
                labels,
                np.full(len(valid_texts), category_id, dtype=np.int32),
            ))
            categories = np.concatenate((
                categories,
                np.full(len(valid_texts), category, dtype=object),
            ))

    print(
        f"[INFO] Loaded {len(texts)} documents from '{directory}' across {len(category_mapping)} categories."
    )

    return texts, labels, category_mapping, categories


def save_news_data_to_json(
    directory: str, output_file: str = "news_data.json", num_workers: int = 4
) -> None:
    """
    Loads news data, selects 100 random articles using the decimal digits of pi as a seed,
    structures them into JSON format, and saves.

    Args:
        directory (str): Path to the dataset folder.
        output_file (str): Path to save the JSON file.
        num_workers (int): Number of parallel file readers.
    """
    texts, _, category_mapping, categories = load_20news_data(
        directory, num_workers=num_workers
    )

    # Use the first 7 decimal places of pi as a reproducible seed
    pi_seed = int(str(math.pi)[2:9])
    random.seed(pi_seed)

    # Randomly sample up to 100 documents
    sample_size = min(100, len(texts))
    selected_indices = np.random.choice(len(texts), sample_size, replace=False)

    json_data = {
        "seed": pi_seed,
        "categories": category_mapping,  # Store category mappings
        "articles": [
            {
                "user_input": texts[i],
                "reference": categories[i],
                "agent_response": "",  # Placeholder for future responses
            }
            for i in selected_indices
        ],
    }

    # Save to JSON
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(json_data, json_file, indent=4, ensure_ascii=False)

    print(
        f"[INFO] JSON data saved to '{output_file}'. Total selected articles: {len(json_data['articles'])}"
    )
