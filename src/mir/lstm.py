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
from typing import Dict, Optional, Tuple, Union, List

# Downloading NLTK datasets for tokenization and stopword filtering.
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
TOKENIZER = nltk.tokenize.TreebankWordTokenizer()


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 32, hidden_dim: int = 64, output_dim: int = 20) -> None:
        super(LSTMClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(hidden)

    def train_model(self, train_loader: DataLoader, num_epochs: int, lr: float, device: Optional[torch.device] = None) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(num_epochs):
            self.train()
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                progress_bar.set_postfix(loss=f"{total_loss / len(train_loader):.4f}")
            print(f"[INFO] Epoch {epoch + 1} Completed: Avg Loss = {total_loss / len(train_loader):.4f}")

    def _save(self, path: str, embed_dim: int, hidden_dim: int) -> None:
        save_dict = {
            "model_state_dict": self.state_dict(),
            "vocab_size": self.vocab_size,
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
        }
        torch.save(save_dict, path)
        print(f"[INFO] Model and vocab size saved to {path}")

    @classmethod
    def from_pretrained(cls, path: str, device: torch.device = torch.device("cpu")) -> "LSTMClassifier":
        checkpoint = torch.load(path, map_location=device)
        model = cls(vocab_size=checkpoint["vocab_size"], embed_dim=checkpoint["embed_dim"], hidden_dim=checkpoint["hidden_dim"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return model


class SubsetDataset(Dataset):
    def __init__(self, full_data: np.ndarray, full_labels: np.ndarray, sample_size: int, seed: int) -> None:
        np.random.seed(seed)
        self.indices = np.random.choice(len(full_labels), sample_size, replace=False)
        self.data = torch.tensor(full_data[self.indices], dtype=torch.long)
        self.labels = torch.tensor(full_labels[self.indices], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


def derive_pi_seed(num_subsets: int, offset: int = 0) -> List[int]:
    pi_str = str(math.pi).replace(".", "")
    seeds = []
    for i in range(num_subsets):
        seed_str = pi_str[offset + i : offset + i + 2]
        if len(seed_str) < 2:
            break
        seeds.append(int(seed_str))
    return seeds


def load_20news_data(
    directory: str, num_workers: int = 2, max_workers: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], np.ndarray]:
    category_folders = np.array(sorted(os.listdir(directory)))
    category_mapping: Dict[str, int] = {category: i for i, category in enumerate(category_folders)}

    texts = np.array([], dtype=object)
    labels = np.array([], dtype=np.int32)
    categories = np.array([], dtype=object)

    for category, category_id in tqdm(category_mapping.items(), desc="Processing categories", unit="category"):
        category_path = os.path.join(directory, category)
        if not os.path.isdir(category_path):
            continue

        file_paths = np.array([os.path.join(category_path, f) for f in os.listdir(category_path)])
        with ThreadPoolExecutor(max_workers=(max_workers or num_workers)) as executor:
            results = np.fromiter(executor.map(read_file, file_paths), dtype=object)

        valid_mask = np.array([x is not None for x in results])
        valid_texts = results[valid_mask]

        if len(valid_texts) > 0:
            texts = np.concatenate((texts, valid_texts)) if texts.size > 0 else valid_texts
            labels = np.concatenate((labels, np.full(len(valid_texts), category_id, dtype=np.int32)))
            categories = np.concatenate((categories, np.full(len(valid_texts), category, dtype=object)))

    print(f"[INFO] Loaded {len(texts)} documents from '{directory}' across {len(category_mapping)} categories.")
    return texts, labels, category_mapping, categories


def read_file(file_path: str) -> Optional[str]:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def clean_text(texts: np.ndarray, stop_words: Optional[set] = None) -> np.ndarray:
    if texts.dtype != object:
        raise ValueError("The NumPy array must contain string objects.")

    header_pattern = re.compile(r"^(Subject|From|Distribution|Organization|NNTP-Posting-Host|Lines):.*$", re.MULTILINE)
    email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b")
    id_pattern = re.compile(r"\b\d{5,}\b")
    special_char_pattern = re.compile(r"[^a-zA-Z0-9\s]")
    non_ascii_pattern = re.compile(r"[^\x00-\x7F]+")
    multiple_spaces_pattern = re.compile(r"\s+")

    if stop_words is None:
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


def tokenize_text(texts: Union[np.ndarray, list], tokenizer=None, num_workers: int = 2, max_workers: Optional[int] = None) -> np.ndarray:
    if isinstance(texts, list):
        texts = np.array(texts, dtype=object)

    if texts.dtype != object:
        raise ValueError("The NumPy array must contain string objects.")

    if tokenizer is None:
        tokenizer = TOKENIZER

    with ThreadPoolExecutor(max_workers=(max_workers or num_workers)) as executor:
        tokenized_texts = list(
            tqdm(
                executor.map(tokenizer.tokenize, texts),
                total=len(texts),
                desc="Tokenizing Texts",
                unit="doc",
            )
        )

    return np.array(tokenized_texts, dtype=object)


def build_vocab(tokenized_texts: np.ndarray, min_freq: int = 2, save_path: Optional[str] = "./vocab.json") -> Dict[str, int]:
    word_counter: Counter = Counter()
    for tokens in tokenized_texts:
        word_counter.update(tokens)

    vocab: Dict[str, int] = {word: idx + 1 for idx, (word, freq) in enumerate(word_counter.items()) if freq >= min_freq}
    vocab["<PAD>"] = 0

    if save_path:
        with open(save_path, "w") as f:
            json.dump(vocab, f)
        print(f"[INFO] Vocabulary saved to {save_path}")

    print(f"[INFO] Vocabulary Size: {len(vocab)} words")
    return vocab


def encode_texts(tokenized_texts: np.ndarray, vocab: Dict[str, int], max_length: int = 50) -> np.ndarray:
    encoded_texts = np.zeros((len(tokenized_texts), max_length), dtype=np.int32)

    for i, tokens in enumerate(tokenized_texts):
        encoded = [vocab.get(word, 0) for word in tokens[:max_length]]
        encoded_texts[i, : len(encoded)] = encoded

    return encoded_texts


def train_pipeline(
    train_dir: str,
    num_epochs: int = 1,
    lr: float = 0.001,
    batch_size: int = 16,
    subset_size: int = 50,
    num_subsets: int = 1,
    embed_dim: int = 32,
    hidden_dim: int = 64,
    max_length: int = 50,
    num_workers: int = 2,
    vocab_save_path: str = "./vocab.json",
    stop_words: Optional[set] = None,
    device: Optional[torch.device] = None,
) -> None:
    train_texts, train_labels, _, _ = load_20news_data(directory=train_dir, num_workers=num_workers)
    train_tokens = tokenize_text(texts=clean_text(train_texts, stop_words=stop_words), num_workers=num_workers)
    vocab = build_vocab(tokenized_texts=train_tokens, min_freq=1, save_path=vocab_save_path)
    train_sequences = encode_texts(tokenized_texts=train_tokens, vocab=vocab, max_length=max_length)

    model = LSTMClassifier(vocab_size=len(vocab), embed_dim=embed_dim, hidden_dim=hidden_dim)

    pi_seeds = derive_pi_seed(num_subsets)

    all_indices = np.array([], dtype=int)
    for seed in pi_seeds:
        np.random.seed(seed)
        indices = np.random.choice(len(train_labels), subset_size, replace=False)
        all_indices = np.concatenate((all_indices, indices), axis=None)

    subset_data = train_sequences[all_indices]
    subset_labels = train_labels[all_indices]

    train_dataset = SubsetDataset(
        full_data=subset_data,
        full_labels=subset_labels,
        sample_size=subset_size,
        seed=np.random.randint(1000),
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    model.train_model(train_loader=train_loader, num_epochs=num_epochs, lr=lr, device=device)


def evaluate_pipeline(
    test_dir: str,
    vocab_path: str,
    model_path: str,
    max_length: int = 50,
    batch_size: int = 16,
    num_workers: int = 2,
    device: Optional[torch.device] = None,
) -> None:
    test_texts, test_labels, _, _ = load_20news_data(directory=test_dir, num_workers=num_workers)

    vocab = load_vocab(vocab_path)
    test_tokens = tokenize_text(texts=clean_text(test_texts))
    test_sequences = encode_texts(tokenized_texts=test_tokens, vocab=vocab, max_length=max_length)

    test_dataset = NewsDataset(encoded_texts=test_sequences, labels=test_labels)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier.from_pretrained(path=model_path, device=device)
    model.eval()

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


def load_vocab(path: str) -> Dict[str, int]:
    with open(path, "r") as f:
        vocab = json.load(f)
    print(f"[INFO] Vocabulary loaded from {path}")
    return vocab


class NewsDataset(Dataset):
    def __init__(self, encoded_texts: np.ndarray, labels: np.ndarray) -> None:
        self.data = torch.tensor(encoded_texts, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


if __name__ == "__main__":
    train_dir = "./data/20news-bydate-train/"
    test_dir = "./data/20news-bydate-test/"

    train_pipeline(
        train_dir=train_dir,
        num_epochs=1,
        lr=0.001,
        batch_size=16,
        subset_size=50,
        num_subsets=1,
        embed_dim=32,
        hidden_dim=64,
        max_length=50,
    )
    evaluate_pipeline(test_dir=test_dir, vocab_path="./vocab.json", model_path="./model.pth", max_length=50, batch_size=16)
