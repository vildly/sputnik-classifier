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

    for ix, text in tqdm(iterable=enumerate(texts), desc="Cleaning texts", total=len(texts), unit="texts"):
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
    # Top-level function for Windows pickling
    ########################################################################
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        return file.read()


def load_20newsdata(paths: List[str], max_docs: Union[int, None] = None) -> Tuple[np.ndarray, np.ndarray]:
    ########################################################################
    # Gather file paths from multiple directories
    ########################################################################
    initial_size = 20
    file_paths = np.empty(initial_size, dtype=object)
    file_labels = np.empty(initial_size, dtype=object)
    file_count = 0

    for path in paths:
        for root, dirs, _ in os.walk(path):
            for category in dirs:
                category_path = os.path.join(root, category)
                for filename in os.listdir(category_path):
                    if max_docs is not None and file_count >= max_docs:
                        break
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
    # Read file contents in parallel
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

    #######################################################################
    # Informative printing about data distribution
    #######################################################################
    print("\nTotal files loaded:", file_count)
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("Distribution of categories in loaded data:")
    print("\tLABEL: COUNT")
    for label, count in zip(unique_labels, counts):
        print(f"\t{label}: {count}")
    print()

    return data, labels


def sample_data(paths: List[str], out="sample.json", size: Union[int, None] = None) -> None:
    ########################################################################
    # Load data from directories
    ########################################################################
    texts, labels = load_20newsdata(paths=paths, max_docs=None)
    total_docs = len(texts)
    if size is None or size > total_docs:
        size = total_docs

    ########################################################################
    # Prepare stratified selection to preserve category ratios
    ########################################################################
    category_to_indices = {}
    for i, lbl in enumerate(labels):
        category_to_indices.setdefault(lbl, []).append(i)

    # Deterministic seed from pi
    pi_seed = int(str(math.pi)[2:9])
    random.seed(pi_seed)

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
    # Handle any rounding discrepancy
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

    ########################################################################
    # Clean only the subset of texts chosen for final sampling
    ########################################################################
    final_texts = texts[final_indices]
    cleaned_texts = clean_texts(final_texts, lang="english")

    ########################################################################
    # Build category mapping and JSON structure
    ########################################################################
    unique_categories = sorted(category_to_indices.keys())
    category_mapping = {cat: i for i, cat in enumerate(unique_categories)}

    json_data = {"seed": pi_seed, "categories": category_mapping, "articles": []}

    for i, idx in enumerate(final_indices):
        json_data["articles"].append({"user_input": cleaned_texts[i], "reference": labels[idx], "agent_response": ""})

    ########################################################################
    # Informative printing about final sampled distribution
    ########################################################################
    print("\nSample size requested:", size)
    print("Actual size of final sampled dataset:", len(final_indices))
    final_labels = labels[final_indices]
    unique_labels_sampled, counts_sampled = np.unique(final_labels, return_counts=True)
    print("Distribution of categories in the sampled dataset:")
    print("\tLABEL: COUNT")
    for label, count in zip(unique_labels_sampled, counts_sampled):
        print(f"\t{label}: {count}")
    print()

    ########################################################################
    # Write out the final JSON
    ########################################################################
    with open(out, "w", encoding="utf-8") as o:
        json.dump(json_data, o, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    sample_data(paths=["./data/20news-bydate-test", "./data/20news-bydate-train"], out="./data/sample.json", size=50)
