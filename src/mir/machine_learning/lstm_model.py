import torch
import torch.nn as nn
import torch.optim as optim
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

    def forward(self, x):
        x_embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(x_embedded)
        hidden = torch.cat(
            (hidden[-2], hidden[-1]), dim=1
        )  # Merge bidirectional outputs
        output = self.fc(hidden)
        return output


def train_model(
    model: LSTMClassifier,
    vocab: dict[str, int],
    train_loader: torch.utils.data.DataLoader,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(len(vocab)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    # 📌 Training Loop with tqdm
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # ✅ Wrap `train_loader` with `tqdm` for real-time batch progress
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"
        )

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # ✅ Update tqdm with current loss
            progress_bar.set_postfix(loss=f"{total_loss / len(train_loader):.4f}")

        print(
            f"[INFO] Epoch {epoch + 1} Completed: Avg Loss = {total_loss / len(train_loader):.4f}"
        )

    print("[INFO] Model Training Complete!")
    return model
