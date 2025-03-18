import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
import os
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Tuple, Union
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter

# Load tokenizer once (instead of per document)
nltk.download("stopwords")
nltk.download("punkt")
TOKENIZER = nltk.tokenize.TreebankWordTokenizer()


def _save_vocab(vocab, path):
    with open(path, "w") as f:
        json.dump(vocab, f)
    print(f"[INFO] Vocabulary saved to {path}")


def load_vocab(path):
    with open(path, "r") as f:
        vocab = json.load(f)
    print(f"[INFO] Vocabulary loaded from {path}")
    return vocab


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 20,
    ):
        super(LSTMClassifier, self).__init__()
        self.vocab_size = vocab_size  # Store vocab size as an attribute
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(x_embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        output = self.fc(hidden)
        return output

    def train_model(
        self,
        train_loader: DataLoader,
        num_epochs: int = 5,
        lr: float = 0.003,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(num_epochs):
            self.train()
            total_loss = 0

            progress_bar = tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"
            )

            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Update tqdm with current loss
                progress_bar.set_postfix(loss=f"{total_loss / len(train_loader):.4f}")

            print(
                f"[INFO] Epoch {epoch + 1} Completed: Avg Loss = {total_loss / len(train_loader):.4f}"
            )

        path = "./model.pth"
        self._save(path=path)
        print(f"[INFO] Model saved to {path}")
        print("[INFO] Model Training Complete!")

    def _save(self, path: str) -> None:
        """Saves the model's state dictionary along with vocab size."""
        torch.save(
            {"model_state_dict": self.state_dict(), "vocab_size": self.vocab_size}, path
        )
        print(f"[INFO] Model and vocab size saved to {path}")

    @classmethod
    def from_pretrained(
        cls, path: str, device: torch.device = torch.device("cpu")
    ) -> "LSTMClassifier":
        # Load model data including state_dict and vocab_size
        checkpoint = torch.load(path, map_location=device)

        # Retrieve vocab_size from the checkpoint
        vocab_size = checkpoint["vocab_size"]

        # Instantiate the model with the retrieved vocab_size
        model = cls(vocab_size=vocab_size)

        # Load the state dictionary into the model
        model.load_state_dict(checkpoint["model_state_dict"])

        # Move model to appropriate device
        model.to(device)

        print(f"[INFO] Model loaded from {path} with vocab size: {vocab_size}")

        return model


class NewsDataset(Dataset):
    def __init__(self, encoded_texts: np.ndarray, labels: np.ndarray):
        self.data = torch.tensor(encoded_texts, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]  # Return (text, label) pairs


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
            results = np.fromiter(executor.map(read_file, file_paths), dtype=object)

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


def clean_text(texts: np.ndarray) -> np.ndarray:
    """
    Cleans an array of input text by removing headers, email addresses, ids,
    special characters, non-ASCII characters, and stop words, and normalizing whitespace.

    Args:
        texts (np.ndarray): A NumPy array of text strings.

    Returns:
        np.ndarray: A NumPy array of cleaned text strings.
    """
    if texts.dtype != object:
        raise ValueError("The NumPy array must contain string objects.")

    # Compile regex patterns
    header_pattern = re.compile(
        r"^(Subject|From|Distribution|Organization|NNTP-Posting-Host|Lines):.*$",
        re.MULTILINE,
    )
    email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b")
    id_pattern = re.compile(
        r"\b\d{5,}\b"
    )  # matches series of digits that look like IDs
    special_char_pattern = re.compile(r"[^a-zA-Z0-9\s]")
    non_ascii_pattern = re.compile(r"[^\x00-\x7F]+")  # matches non-ASCII characters
    multiple_spaces_pattern = re.compile(r"\s+")

    # Load English stop words
    stop_words = set(stopwords.words("english"))

    cleaned_texts = np.empty_like(texts, dtype=object)

    for i in tqdm(range(len(texts)), desc="Cleaning Text", unit="doc"):
        text = texts[i]

        # Remove headers
        text = header_pattern.sub("", text)

        # Remove email addresses
        text = email_pattern.sub("", text)

        # Remove IDs and normalize text
        text = id_pattern.sub("", text)
        text = special_char_pattern.sub(" ", text)

        # Remove non-ASCII characters
        text = non_ascii_pattern.sub(" ", text)

        # Normalize whitespace
        text = text.lower()
        text = multiple_spaces_pattern.sub(" ", text).strip()

        # Tokenize text and remove stopwords
        tokens = text.split()
        tokens = [word for word in tokens if word not in stop_words]

        # Reconstruct cleaned text
        text = " ".join(tokens)

        cleaned_texts[i] = text

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


def create_test_set(
    directory: str,
    num_workers: int = 4,
    seed: Optional[int] = None,
    sample_size: int = 100,
):
    """
    Preprocesses the news dataset, selects a specified number of preprocessed articles,
    and saves them to a JSON file.

    Args:
        directory (str): Path to the dataset folder.
        output_file (str): Path to save the JSON file.
        num_workers (int): Number of parallel file readers.
        seed (int, optional): Seed for the random number generator.
        sample_size (int): Number of articles to sample.
    """
    # Load the dataset
    texts, _, category_mapping, categories = load_20news_data(
        directory, num_workers=num_workers
    )

    # Clean the texts
    cleaned_texts = clean_text(texts)

    # Tokenize the cleaned texts
    tokenized_texts = tokenize_text(cleaned_texts, num_workers=num_workers)

    # Use a provided seed or a default value
    random_seed = seed if seed is not None else int(str(math.pi)[2:9])
    random.seed(random_seed)

    # Ensure compatibility with JSON structure by joining tokens back to a single string
    textual_texts = [" ".join(tokens) for tokens in tokenized_texts]

    # Determine the sample size (no more than the total number of texts)
    sample_size = min(sample_size, len(texts))

    # Randomly select articles based on the given seed and sample size
    selected_indices = np.random.choice(len(texts), sample_size, replace=False)

    # Structure the sampled data for JSON output
    json_data = {
        "seed": random_seed,
        "categories": category_mapping,
        "articles": [
            {
                "user_input": textual_texts[i],
                "reference": categories[i],
                "agent_response": "",  # Placeholder for future responses
            }
            for i in selected_indices
        ],
    }

    # Save the structured data into a JSON file
    output_file = "./test_set.json"
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(json_data, json_file, indent=4, ensure_ascii=False)

    print(
        f"[INFO] JSON data saved to '{output_file}'. Total selected articles: {len(json_data['articles'])}"
    )


def build_vocab(tokenized_texts: np.ndarray, min_freq: int = 2) -> Dict[str, int]:
    """
    Creates a vocabulary mapping each unique word to an index.

    Args:
        tokenized_texts (np.ndarray): A NumPy array of tokenized documents.
        min_freq (int): The minimum frequency a word must have to be included in the vocabulary.

    Returns:
        Dict[str, int]: A dictionary mapping words to unique index values.
    """
    word_counter: Counter = Counter()
    for tokens in tokenized_texts:
        word_counter.update(tokens)

    # Only keep words that appear `min_freq` times or more
    vocab: Dict[str, int] = {
        word: idx + 1
        for idx, (word, freq) in enumerate(word_counter.items())
        if freq >= min_freq
    }
    vocab["<PAD>"] = 0  # Special padding token for sequence alignment

    path = "./vocab.json"
    _save_vocab(vocab, path=path)
    print(f"[INFO] Saved vocabulary to {path}")
    print(f"[INFO] Vocabulary Size: {len(vocab)} words")
    return vocab


def encode_texts(
    tokenized_texts: np.ndarray, vocab: dict[str, int], max_length: int = 100
) -> np.ndarray:
    """
    Converts a NumPy array of tokenized documents into sequences of word indices.

    Args:
        tokenized_texts (np.ndarray): Tokenized texts as a NumPy array.
        vocab (dict[str, int]): The vocabulary mapping words to indices.
        max_length (int): The maximum sequence length (texts are padded/truncated to this length).

    Returns:
        np.ndarray: Encoded text sequences as a NumPy array.
    """
    encoded_texts = np.zeros((len(tokenized_texts), max_length), dtype=np.int32)

    for i, tokens in enumerate(tokenized_texts):
        encoded = [
            vocab.get(word, 0) for word in tokens[:max_length]
        ]  # Map words to indices
        for j in range(len(encoded)):
            if encoded[j] >= len(vocab):  # ✅ Ensure index is valid
                encoded[j] = 0  # Convert out-of-range indices to `<PAD>`
        encoded_texts[i, : len(encoded)] = encoded  # Insert sequence

    return encoded_texts
