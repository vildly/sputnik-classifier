import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 20,
    ):
        super(LSTMClassifier, self).__init__()
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

        print("[INFO] Model Training Complete!")


def save_model(model: nn.Module, path: str) -> None:
    """Saves the model's state dictionary to the specified path."""
    torch.save(model.state_dict(), path)
    print(f"[INFO] Model saved to {path}")


def load_model(
    model: nn.Module, path: str, device: torch.device = torch.device("cpu")
) -> nn.Module:
    """Loads model parameters from the specified file into the provided model instance."""
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"[INFO] Model loaded from {path}")
    return model
