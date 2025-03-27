import os
import math
import json
import random
import concurrent.futures
import numpy as np
from tqdm import tqdm
import nltk
from string import punctuation
from typing import Dict, List, Set, Generator, Tuple, Union


def clean_texts(texts: np.ndarray, lang: str = "english") -> np.ndarray:
    cleaned_texts: np.ndarray = np.empty_like(texts, dtype=object)
    for ix, text in tqdm(enumerate(texts), desc="Cleaning texts", total=len(texts), unit="texts"):
        # Lowercase
        text = text.lower()

        # Remove punctuation
        translator: Dict[int, int | None] = str.maketrans("", "", punctuation)
        text = text.translate(translator)

        # Remove stopwords
        words: List[str] = nltk.word_tokenize(text)
        stop_words: Set[str] = set(nltk.corpus.stopwords.words(lang))
        filtered_words: Generator[str, None, None] = (word for word in words if word not in stop_words)

        cleaned_texts[ix] = " ".join(filtered_words)
    print(f"{len(cleaned_texts)} texts cleaned")
    return cleaned_texts


def _read_file(file_path: str) -> str:
    ########################################################################
    # Top-level function for Windows pickling.
    ########################################################################
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        return file.read()


def load_20newsdata(paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads data from the provided paths. Files are expected in subdirectories (e.g., by category).
    Returns a tuple of all texts and their associated labels.
    """
    initial_size = 20
    file_paths = np.empty(initial_size, dtype=object)
    file_labels = np.empty(initial_size, dtype=object)
    file_count = 0

    for path in paths:
        for root, dirs, _ in os.walk(path):
            for category in dirs:
                category_path = os.path.join(root, category)
                for filename in os.listdir(category_path):
                    file_path = os.path.join(category_path, filename)
                    if os.path.isfile(file_path):
                        if file_count >= len(file_paths):
                            file_paths = np.resize(file_paths, file_count * 2)
                            file_labels = np.resize(file_labels, file_count * 2)
                        file_paths[file_count] = file_path
                        file_labels[file_count] = category
                        file_count += 1

    file_paths = file_paths[:file_count]
    file_labels = file_labels[:file_count]

    ########################################################################
    # Read file contents in parallel.
    ########################################################################
    with concurrent.futures.ProcessPoolExecutor() as executor:
        files_read = list(
            tqdm(
                executor.map(_read_file, file_paths),
                desc="Reading files",
                total=file_count,
                unit="files",
            )
        )

    data = np.array(files_read, dtype=object)
    labels = np.array(file_labels, dtype=object)

    ########################################################################
    # Informative printing about data distribution.
    ########################################################################
    print("\nTotal files loaded:", file_count)
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("Distribution of categories in loaded data:")
    print("\tLABEL: COUNT")
    for label, count in zip(unique_labels, counts):
        print(f"\t{label}: {count}")
    print()

    return data, labels


def sample_data(
    data: np.ndarray,
    labels: np.ndarray,
    size: Union[int, None] = None,
    seed: Union[int, None] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Samples data using a stratified approach from the given data arrays.

    :param data: A NumPy array of text data.
    :param labels: A NumPy array of corresponding labels.
    :param size: The number of documents to sample. If None or too large, all documents are used.
    :param seed: Random seed for reproducibility.
    :return: A tuple (sampled_data, sampled_labels) where sampled_data are texts.
    """
    total_docs = len(data)
    if size is None or size > total_docs:
        size = total_docs

    ########################################################################
    # Prepare stratified selection to preserve category ratios.
    ########################################################################
    category_to_indices = {}
    for i, lbl in enumerate(labels):
        category_to_indices.setdefault(lbl, []).append(i)

    # Use the provided seed, or fallback to a deterministic seed from π.
    if seed is not None:
        effective_seed = seed
    else:
        effective_seed = int(str(math.pi)[2:9])
    random.seed(effective_seed)

    all_indices = list(range(total_docs))
    random.shuffle(all_indices)

    final_indices = []
    allocated = 0
    for _, idx_list in category_to_indices.items():
        cat_ratio = len(idx_list) / total_docs
        cat_count = int(round(cat_ratio * size))
        random.shuffle(idx_list)
        final_indices.extend(idx_list[:cat_count])
        allocated += cat_count

    ########################################################################
    # Handle any rounding discrepancy.
    ########################################################################
    discrepancy = size - allocated
    if discrepancy != 0:
        leftover = list(set(all_indices) - set(final_indices))
        random.shuffle(leftover)
        if discrepancy > 0:
            final_indices.extend(leftover[:discrepancy])
        else:
            final_indices = final_indices[:discrepancy]

    random.shuffle(final_indices)

    sampled_data = data[final_indices]
    sampled_labels = labels[final_indices]

    ########################################################################
    # Informative printing about final sampled distribution.
    ########################################################################
    print("\nSample size requested:", size)
    print("Actual size of final sampled dataset:", len(final_indices))
    unique_sampled_labels, counts_sampled = np.unique(sampled_labels, return_counts=True)
    print("Distribution of categories in the sampled dataset:")
    print("\tLABEL: COUNT")
    for label, count in zip(unique_sampled_labels, counts_sampled):
        print(f"\t{label}: {count}")
    print()

    return sampled_data, sampled_labels


def save_data_to_json(data: np.ndarray, labels: np.ndarray, out: str = "sample.json") -> None:
    """
    Saves the provided tuple of texts and labels into a JSON file.
    The JSON structure is as follows:
      {
          "seed": <seed used>,
          "categories": { <category_name>: <id>, ... },
          "articles": [
              { "user_input": <text>, "reference": <label>, "agent_response": "" },
              ...
          ]
      }

    :param sampled_data: Tuple where the first element is a NumPy array of texts and the second is the labels.
    :param out: The file path where the JSON file will be saved.
    """
    # Build category mapping from the sampled labels.
    unique_labels = sorted(np.unique(labels))
    category_mapping = {cat: i for i, cat in enumerate(unique_labels)}

    articles = []
    for text, label in zip(data, labels):
        articles.append({"user_input": text, "reference": label, "agent_response": ""})

    json_data = {"categories": category_mapping, "articles": articles}

    with open(out, "w", encoding="utf-8") as o:
        json.dump(json_data, o, indent=4, ensure_ascii=False)
    print(f"Data saved to {out}")


if __name__ == "__main__":
    # Load training and test data separately.
    train_data: np.ndarray
    train_labels: np.ndarray
    train_data, train_labels = load_20newsdata(paths=["./data/20news-bydate-train"])
    test_data: np.ndarray
    test_labels: np.ndarray
    test_data, test_labels = load_20newsdata(paths=["./data/20news-bydate-test"])

    # Combine data and labels from train and test.
    combined_data: np.ndarray = np.concatenate((train_data, test_data), axis=0)
    combined_labels: np.ndarray = np.concatenate((train_labels, test_labels), axis=0)

    # Perform stratified sampling with a size of documents and a provided seed.
    sampled_data: np.ndarray
    sampled_labels: np.ndarray
    sampled_data, sampled_labels = sample_data(data=combined_data, labels=combined_labels, size=100, seed=None)

    cleaned_sampled_data: np.ndarray = clean_texts(texts=sampled_data, lang="english")

    # Save the sampled data to a JSON file.
    save_data_to_json(data=cleaned_sampled_data, labels=sampled_labels, out="sample.json")
