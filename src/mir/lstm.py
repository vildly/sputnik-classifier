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
from typing import Dict, Optional, Tuple, List

# Setup NLTK for tokenization and stopword filtering
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
TOKENIZER = nltk.tokenize.TreebankWordTokenizer()


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 32, hidden_dim: int = 64, output_dim: int = 20) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(hidden)

    def train_model(self, train_loader: DataLoader, num_epochs: int, lr: float, device: torch.device) -> None:
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
                progress_bar.set_postfix(loss=total_loss / len(train_loader))
            print(f"[INFO] Epoch {epoch + 1} Completed: Avg Loss = {total_loss / len(train_loader):.4f}")

    def save(self, model_path: str) -> None:
        torch.save(self.state_dict(), model_path)
        print(f"[INFO] Model saved to {model_path}")

    @classmethod
    def from_pretrained(cls, vocab_size: int, embed_dim: int, hidden_dim: int, model_path: str, device: torch.device):
        model = cls(vocab_size, embed_dim, hidden_dim)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        return model


class TextDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray) -> None:
        self.data = torch.tensor(data, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


def derive_pi_seeds(num_subsets: int, offset: int = 0) -> List[int]:
    pi_str = str(math.pi).replace(".", "")
    seeds = [int(pi_str[i : i + 2]) for i in range(offset, offset + 2 * num_subsets, 2) if len(pi_str[i : i + 2]) == 2]
    return seeds


def load_data(directory: str, num_workers: int = 2) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    category_map = {cat: idx for idx, cat in enumerate(sorted(os.listdir(directory)))}
    texts, labels = [], []

    # Collect texts and labels
    for category, cat_id in tqdm(category_map.items(), desc="Loading data", unit="category"):
        category_path = os.path.join(directory, category)
        if os.path.isdir(category_path):
            file_paths = (os.path.join(category_path, f) for f in os.listdir(category_path))
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                results = list(executor.map(read_file, file_paths))
            valid_texts = [r for r in results if r is not None]
            texts.extend(valid_texts)
            labels.extend([cat_id] * len(valid_texts))

    # Use numpy directly
    texts_array = np.array(texts, dtype=object)
    labels_array = np.array(labels, dtype=np.int32)

    print(f"[INFO] Loaded {len(texts)} documents from {len(category_map)} categories.")
    return texts_array, labels_array, category_map


def read_file(file_path: str) -> Optional[str]:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def clean_texts(texts: np.ndarray) -> np.ndarray:
    stop_words = set(stopwords.words("english"))

    # Define the patterns to clean the text
    patterns = [
        re.compile(r"^(Subject|From|Distribution|Organization|Lines):.*$", re.MULTILINE),
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"),
        re.compile(r"\b\d{5,}\b"),
        re.compile(r"[^a-zA-Z0-9\s]"),
        re.compile(r"[^\x00-\x7F]+"),
        re.compile(r"\s+"),
    ]

    cleaned_texts = np.empty_like(texts, dtype=object)

    for i, text in enumerate(tqdm(texts, desc="Cleaning texts", unit="doc")):
        for pattern in patterns:
            text = pattern.sub(" ", text)

        # Lowercase and remove stop words
        text = text.lower().strip()
        text = " ".join(word for word in text.split() if word not in stop_words)

        cleaned_texts[i] = text

    return cleaned_texts


def tokenize_texts(texts: np.ndarray, num_workers: int = 2) -> np.ndarray:
    # Direct computation with numpy, avoiding unnecessary list wrapping
    def tokenize_single(text):
        return TOKENIZER.tokenize(text)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        tokenized_texts = list(tqdm(executor.map(tokenize_single, texts), total=len(texts), desc="Tokenizing texts", unit="doc"))

    return np.array(tokenized_texts, dtype=object)


def build_vocab(tokenized_texts: np.ndarray, vocab_path: str, min_freq: int = 2) -> Dict[str, int]:
    counter = Counter(word for tokens in tokenized_texts for word in tokens)
    vocab = {word: idx + 1 for idx, (word, freq) in enumerate(counter.items()) if freq >= min_freq}
    vocab["<PAD>"] = 0

    with open(vocab_path, "w") as f:
        json.dump(vocab, f)
    print(f"[INFO] Vocabulary saved to {vocab_path}, size: {len(vocab)}")
    return vocab


def encode_texts(tokenized_texts: np.ndarray, vocab: Dict[str, int], max_length: int = 50) -> np.ndarray:
    return np.array([[vocab.get(word, 0) for word in tokens[:max_length]] + [0] * (max_length - len(tokens)) for tokens in tokenized_texts])


def train_pipeline(config: Dict) -> None:
    train_texts, train_labels, _ = load_data(config["train_dir"], config["num_workers"])
    train_texts = clean_texts(train_texts)
    train_tokens = tokenize_texts(train_texts, config["num_workers"])
    vocab = build_vocab(train_tokens, min_freq=1, vocab_path=config["vocab_path"])
    train_sequences = encode_texts(train_tokens, vocab, max_length=config["max_length"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(len(vocab), config["embed_dim"], config["hidden_dim"])
    seeds = derive_pi_seeds(config["num_subsets"])

    selected_indices = []
    for seed in seeds:
        np.random.seed(seed)
        selected_indices.extend(np.random.choice(len(train_labels), config["subset_size"], replace=False))

    subset_data = train_sequences[selected_indices]
    subset_labels = train_labels[selected_indices]

    train_loader = DataLoader(TextDataset(subset_data, subset_labels), batch_size=config["batch_size"], shuffle=True)

    model.train_model(train_loader, config["num_epochs"], config["lr"], device)
    model.save(config["model_path"])


def evaluate_pipeline(config: Dict) -> None:
    test_texts, test_labels, _ = load_data(config["test_dir"], config["num_workers"])
    test_texts = clean_texts(test_texts)
    test_tokens = tokenize_texts(test_texts, config["num_workers"])
    vocab = load_vocab(config["vocab_path"])
    test_sequences = encode_texts(test_tokens, vocab, max_length=config["max_length"])

    if config.get("subset_size") is not None:
        selected_indices = np.random.choice(len(test_labels), config["subset_size"], replace=False)
        test_sequences, test_labels = test_sequences[selected_indices], test_labels[selected_indices]

    test_loader = DataLoader(TextDataset(test_sequences, test_labels), batch_size=config["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier.from_pretrained(len(vocab), config["embed_dim"], config["hidden_dim"], config["model_path"], device)

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating", unit="batch"):
            inputs = inputs.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"[INFO] Test Accuracy: {accuracy:.4f}")


def load_vocab(vocab_path: str) -> Dict[str, int]:
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    print(f"[INFO] Vocabulary loaded from {vocab_path}")
    return vocab


if __name__ == "__main__":
    config = {
        "train_dir": "./data/20news-bydate-train/",
        "test_dir": "./data/20news-bydate-test/",
        "vocab_path": "./vocab.json",
        "model_path": "./model.pth",
        "num_epochs": 1,
        "lr": 0.001,
        "batch_size": 64,
        "subset_size": 50,
        "num_subsets": 1,
        "embed_dim": 32,
        "hidden_dim": 64,
        "max_length": 50,
        "num_workers": 8,
    }

    train_pipeline(config)
    evaluate_pipeline(config)
