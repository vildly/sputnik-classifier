import re
import os
import concurrent.futures
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import Counter

print("PyTorch Version : {}".format(torch.__version__))


def word_tokenizer(text):
    return re.findall(r"\b\w+\b", text.lower())


def build_vocabulary(datasets, tokenizer, min_freq=1):
    token_counter = Counter()

    for text in datasets:
        tokens = tokenizer(text)
        token_counter.update(tokens)

    vocab = {token: idx + 1 for idx, (token, freq) in enumerate(token_counter.items()) if freq >= min_freq}
    vocab["<UNK>"] = 0  # Unknown token at index 0
    return vocab


def text_to_indices(text, tokenizer, vocab):
    tokens = tokenizer(text)
    indices = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    return indices


def read_file(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        return file.read()


def load_dir(path):
    initial_size = 20
    file_paths = np.empty(initial_size, dtype=object)
    file_labels = np.empty(initial_size, dtype=object)
    file_count = 0

    # root, dirs, files
    for root, dirs, _ in os.walk(path):
        for category in dirs:
            category_path = os.path.join(root, category)
            for filename in os.listdir(category_path):
                file_path = os.path.join(category_path, filename)
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
        results = list(executor.map(read_file, file_paths))

    # Convert results to numpy array
    data = np.array(results, dtype=object)
    labels = np.array(file_labels, dtype=object)

    return data, labels


# Load train and test datasets
train_data, train_labels = load_dir(path="./data/20news-bydate-train")
print(f"Train Data Size: {len(train_data)}")
print(f"Train Labels Size: {len(train_labels)}")

vocab = build_vocabulary(datasets=train_data, tokenizer=word_tokenizer, min_freq=5)

print(f"Vocabulary Size: {len(vocab)}")

sample_text = "This is a sample sentance waaaaaassup."
indices = text_to_indices(text=sample_text, tokenizer=word_tokenizer, vocab=vocab)
print(f"Sample Text: {sample_text}")
print(f"Indices: {indices}")
