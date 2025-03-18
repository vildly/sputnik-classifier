import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import os
import math
import json
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import accuracy_score
from typing import Dict, Optional, Tuple, Union

# Downloading NLTK datasets for tokenization and stopword filtering.
nltk.download("stopwords")
nltk.download("punkt")
TOKENIZER = nltk.tokenize.TreebankWordTokenizer()


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 20,
    ):
        """
        Initializes an LSTM-based classifier for sequence classification tasks.

        Args:
            vocab_size (int): Size of the vocabulary for embedding.
            embed_dim (int): Dimension of word embeddings.
            hidden_dim (int): Hidden state size of the LSTM layers.
            output_dim (int): Number of output classes.
        """
        super(LSTMClassifier, self).__init__()
        self.vocab_size = vocab_size  # Ensure vocab_size is an instance attribute
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute logits for input sequence batch.

        Args:
            x (torch.Tensor): A batch of input sequences (word indices).

        Returns:
            torch.Tensor: Logits for each class.
        """
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
        """
        Trains the model using the specified DataLoader and hyperparameters.

        Args:
            train_loader (DataLoader): DataLoader for training batches.
            num_epochs (int): Number of training epochs.
            lr (float): Learning rate for the optimizer.
        """
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
                progress_bar.set_postfix(loss=f"{total_loss / len(train_loader):.4f}")

            print(
                f"[INFO] Epoch {epoch + 1} Completed: Avg Loss = {total_loss / len(train_loader):.4f}"
            )

        self._save(
            path="./model.pth",
            embed_dim=self.embedding.embedding_dim,
            hidden_dim=self.lstm.hidden_size,
        )
        print("[INFO] Model Training Complete!")

    def _save(self, path: str, embed_dim: int, hidden_dim: int) -> None:
        """
        Saves the model state and hyperparameters to a file.

        Args:
            path (str): Destination path for saving model.
            embed_dim (int): Embedding dimension size.
            hidden_dim (int): Hidden state size of the LSTM layers.
        """
        save_dict = {
            "model_state_dict": self.state_dict(),
            "vocab_size": self.vocab_size,
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
        }
        torch.save(save_dict, path)
        print(f"[INFO] Model and vocab size saved to {path}")

    @classmethod
    def from_pretrained(
        cls, path: str, device: torch.device = torch.device("cpu")
    ) -> "LSTMClassifier":
        """
        Loads a pretrained model from a checkpoint.

        Args:
            path (str): Path to the saved model checkpoint.
            device (torch.device, optional): Device to map model (CPU/GPU).

        Returns:
            LSTMClassifier: An instance of LSTMClassifier with weights loaded.
        """
        checkpoint = torch.load(path, map_location=device)
        vocab_size = checkpoint["vocab_size"]
        embed_dim = checkpoint["embed_dim"]
        hidden_dim = checkpoint["hidden_dim"]
        model = cls(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        print(
            f"[INFO] Model loaded from {path} with vocab size: {vocab_size}, embed_dim: {embed_dim}, hidden_dim: {hidden_dim}"
        )
        return model


class SubsetDataset(Dataset):
    def __init__(self, full_data, full_labels, sample_size, seed):
        """
        Initializes a dataset consisting of a random sample from the full dataset.

        Args:
            full_data (np.ndarray): Full dataset of encoded texts.
            full_labels (np.ndarray): Corresponding labels.
            sample_size (int): Number of samples to include in the subset.
            seed (int): Seed for random number generator.
        """
        np.random.seed(seed)
        self.indices = np.random.choice(len(full_labels), sample_size, replace=False)
        self.data = torch.tensor(full_data[self.indices], dtype=torch.long)
        self.labels = torch.tensor(full_labels[self.indices], dtype=torch.long)

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieves the data and label at a given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Data tensor and corresponding label.
        """
        return self.data[idx], self.labels[idx]


def derive_pi_seed(num_subsets, offset=0):
    """
    Generates a list of pseudo-random seeds derived from the digits of pi.

    Args:
        num_subsets (int): Number of seeds to generate.
        offset (int, optional): Starting point in pi's digits.

    Returns:
        List[int]: A list of integer seeds.
    """
    pi_str = str(math.pi).replace(".", "")
    seeds = []
    for i in range(num_subsets):
        seed_str = pi_str[offset + i : offset + i + 2]
        if len(seed_str) < 2:
            break
        seeds.append(int(seed_str))
    return seeds


def load_20news_data(
    directory: str, num_workers: int = 4
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], np.ndarray]:
    """
    Loads and processes 20 Newsgroups data from the specified directory.

    Args:
        directory (str): Directory containing data organized by category.
        num_workers (int): Number of workers for concurrent file reading.

    Returns:
        Tuple: Contains arrays of texts, labels, a category-to-index mapping, and categories.
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
            continue

        file_paths = np.array([
            os.path.join(category_path, f) for f in os.listdir(category_path)
        ])
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = np.fromiter(executor.map(read_file, file_paths), dtype=object)

        valid_mask = np.array([x is not None for x in results])
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


def read_file(file_path: str) -> Optional[str]:
    """
    Reads the content of a file, returning None if unsuccessful.

    Args:
        file_path (str): Path to the file.

    Returns:
        Optional[str]: The file's content or None if an error occurred.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def clean_text(texts: np.ndarray) -> np.ndarray:
    """
    Cleans and preprocesses text data by removing noise and unwanted patterns.

    Args:
        texts (np.ndarray): Array of raw text data.

    Returns:
        np.ndarray: Cleaned text array with stopwords removed.
    """
    if texts.dtype != object:
        raise ValueError("The NumPy array must contain string objects.")

    header_pattern = re.compile(
        r"^(Subject|From|Distribution|Organization|NNTP-Posting-Host|Lines):.*$",
        re.MULTILINE,
    )
    email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b")
    id_pattern = re.compile(r"\b\d{5,}\b")
    special_char_pattern = re.compile(r"[^a-zA-Z0-9\s]")
    non_ascii_pattern = re.compile(r"[^\x00-\x7F]+")
    multiple_spaces_pattern = re.compile(r"\s+")

    stop_words = set(stopwords.words("english"))
    cleaned_texts = np.empty_like(texts, dtype=object)

    for i in tqdm(range(len(texts)), desc="Cleaning Text", unit="doc"):
        text = texts[i]
        text = header_pattern.sub("", text)
        text = email_pattern.sub("", text)
        text = id_pattern.sub("", text)
        text = special_char_pattern.sub(" ", text)
        text = non_ascii_pattern.sub(" ", text)
        text = text.lower()
        text = multiple_spaces_pattern.sub(" ", text).strip()
        tokens = text.split()
        tokens = [word for word in tokens if word not in stop_words]
        text = " ".join(tokens)
        cleaned_texts[i] = text

    return cleaned_texts


def tokenize_text(texts: Union[np.ndarray, list], num_workers: int = 8) -> np.ndarray:
    """
    Tokenizes text sequences using a specified tokenizer, parallelized across a given number of workers.

    Args:
        texts (Union[np.ndarray, list]): Texts to tokenize.
        num_workers (int, optional): Number of threads for parallelization.

    Returns:
        np.ndarray: Tokenized text arrays.
    """
    if isinstance(texts, list):
        texts = np.array(texts, dtype=object)

    if texts.dtype != object:
        raise ValueError("The NumPy array must contain string objects.")

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


def build_vocab(tokenized_texts: np.ndarray, min_freq: int = 2) -> Dict[str, int]:
    """
    Builds a vocabulary from tokenized texts, filtered by a minimum frequency.

    Args:
        tokenized_texts (np.ndarray): Array of token lists.
        min_freq (int): Minimum frequency for a word to be included.

    Returns:
        Dict[str, int]: Vocabulary dictionary mapping tokens to indices.
    """
    word_counter: Counter = Counter()
    for tokens in tokenized_texts:
        word_counter.update(tokens)

    vocab: Dict[str, int] = {
        word: idx + 1
        for idx, (word, freq) in enumerate(word_counter.items())
        if freq >= min_freq
    }
    vocab["<PAD>"] = 0

    path = "./vocab.json"
    with open(path, "w") as f:
        json.dump(vocab, f)
    print(f"[INFO] Vocabulary saved to {path}")
    print(f"[INFO] Vocabulary Size: {len(vocab)} words")
    return vocab


def encode_texts(
    tokenized_texts: np.ndarray, vocab: dict[str, int], max_length: int = 100
) -> np.ndarray:
    """
    Encodes tokenized texts into sequences of indices according to vocabulary mappings.

    Args:
        tokenized_texts (np.ndarray): Array of lists of tokens.
        vocab (dict[str, int]): Vocabulary mapping from token to index.
        max_length (int, optional): Maximum sequence length to pad or truncate to.

    Returns:
        np.ndarray: Array of encoded, padded sequences.
    """
    encoded_texts = np.zeros((len(tokenized_texts), max_length), dtype=np.int32)

    for i, tokens in enumerate(tokenized_texts):
        encoded = [vocab.get(word, 0) for word in tokens[:max_length]]
        encoded_texts[i, : len(encoded)] = encoded

    return encoded_texts


def train_pipeline(
    train_dir,
    num_epochs=5,
    lr=0.003,
    batch_size=32,
    subset_size=1000,
    num_subsets=10,
    embed_dim=128,
    hidden_dim=256,
    max_length=200,
):
    """
    Orchestrates the full training process: data loading, processing, and model training.

    Args:
        train_dir (str): Directory containing training data.
        num_epochs (int, optional): Number of training epochs.
        lr (float, optional): Learning rate.
        batch_size (int, optional): Batch size.
        subset_size (int, optional): Size of data subsets for training.
        num_subsets (int, optional): Number of data subsets to generate for training.
        embed_dim (int, optional): Embedding dimension.
        hidden_dim (int, optional): Hidden dimension for LSTM.
        max_length (int, optional): Maximum length for input sequences.
    """
    train_texts, train_labels, _, _ = load_20news_data(
        directory=train_dir, num_workers=8
    )
    train_tokens = tokenize_text(texts=clean_text(train_texts), num_workers=8)
    vocab = build_vocab(tokenized_texts=train_tokens, min_freq=1)
    train_sequences = encode_texts(
        tokenized_texts=train_tokens, vocab=vocab, max_length=max_length
    )

    model = LSTMClassifier(
        vocab_size=len(vocab), embed_dim=embed_dim, hidden_dim=hidden_dim
    )

    pi_seeds = derive_pi_seed(num_subsets)

    all_indices = np.array([], dtype=int)
    for seed in pi_seeds:
        np.random.seed(seed)
        indices = np.random.choice(len(train_labels), subset_size, replace=False)
        all_indices = np.concatenate((all_indices, indices), axis=None)

    subset_data = train_sequences[all_indices]
    subset_labels = train_labels[all_indices]

    train_dataset = NewsDataset(encoded_texts=subset_data, labels=subset_labels)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    model.train_model(train_loader=train_loader, num_epochs=num_epochs, lr=lr)


def evaluate_pipeline(
    test_dir,
    vocab_path="./vocab.json",
    model_path="./model.pth",
    max_length=200,
    batch_size=32,
):
    """
    Evaluates the model using a test dataset in terms of accuracy.

    Args:
        test_dir (str): Directory containing test data.
        vocab_path (str, optional): Path to vocabulary file.
        model_path (str, optional): Path to saved model file.
        max_length (int, optional): Maximum sequence length.
        batch_size (int, optional): Batch size for evaluation.
    """
    test_texts, test_labels, _, _ = load_20news_data(directory=test_dir, num_workers=8)

    vocab = load_vocab(vocab_path)
    test_tokens = tokenize_text(texts=clean_text(test_texts))
    test_sequences = encode_texts(
        tokenized_texts=test_tokens, vocab=vocab, max_length=max_length
    )

    test_dataset = NewsDataset(encoded_texts=test_sequences, labels=test_labels)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier.from_pretrained(path=model_path)
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating", unit="batch")
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"[INFO] Test Accuracy: {accuracy:.4f}")


def load_vocab(path):
    """
    Loads a vocabulary mapping from a JSON file.

    Args:
        path (str): Path to the vocabulary JSON file.

    Returns:
        dict: The vocabulary mapping from tokens to indices.
    """
    with open(path, "r") as f:
        vocab = json.load(f)
    print(f"[INFO] Vocabulary loaded from {path}")
    return vocab


class NewsDataset(Dataset):
    """
    Dataset class for encoded texts and corresponding labels.
    Used to supply data in batches to DataLoader during model training and evaluation.
    """

    def __init__(self, encoded_texts: np.ndarray, labels: np.ndarray):
        self.data = torch.tensor(encoded_texts, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int):
        """
        Retrieves the data and label at a given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Data tensor and corresponding label.
        """
        return self.data[idx], self.labels[idx]


# Main executable section setting hyperparameters and initiating training and evaluation.
if __name__ == "__main__":
    train_dir = "./data/20news-bydate-train/"
    test_dir = "./data/20news-bydate-test/"

    embed_dim = 256
    hidden_dim = 512
    max_length = 150

    train_pipeline(
        train_dir=train_dir,
        num_epochs=1,
        lr=0.001,
        batch_size=64,
        subset_size=1000,
        num_subsets=1,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        max_length=max_length,
    )

    evaluate_pipeline(
        test_dir=test_dir,
        vocab_path="./vocab.json",
        model_path="./model.pth",
        max_length=max_length,
        batch_size=32,
    )
