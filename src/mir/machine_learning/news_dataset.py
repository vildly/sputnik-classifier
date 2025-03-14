import numpy as np
import torch
from torch.utils.data import Dataset


class NewsDataset(Dataset):
    def __init__(self, encoded_texts: np.ndarray, labels: np.ndarray):
        self.data = torch.tensor(encoded_texts, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]  # Return (text, label) pairs
