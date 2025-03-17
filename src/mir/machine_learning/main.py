import json
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader
from lstm_model import LSTMClassifier
from load_dataset import load_20news_data
from preprocess import clean_text, tokenize_text
from encode import build_vocab, encode_texts
from news_dataset import NewsDataset

train_dir = "./../data/20news-bydate-train/"
test_dir = "./../data/20news-bydate-test/"


def save_vocab(vocab, path):
    with open(path, "w") as f:
        json.dump(vocab, f)
    print(f"[INFO] Vocabulary saved to {path}")


def load_vocab(path):
    with open(path, "r") as f:
        vocab = json.load(f)
    print(f"[INFO] Vocabulary loaded from {path}")
    return vocab


def train_and_save_model():
    # Load and preprocess training data
    train_texts, train_labels, _, _ = load_20news_data(
        directory=train_dir, num_workers=8
    )

    # Preprocess the training data
    train_tokens = tokenize_text(texts=clean_text(train_texts), num_workers=8)
    # min_freq=1 to include all words
    vocab = build_vocab(tokenized_texts=train_tokens, min_freq=1)
    # Save the vocabulary
    save_vocab(vocab=vocab, path="./vocab.json")

    # Create Dataset and DataLoader
    train_sequences = encode_texts(
        tokenized_texts=train_tokens, vocab=vocab, max_length=200
    )
    train_dataset = NewsDataset(encoded_texts=train_sequences, labels=train_labels)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

    # Instantiate and train the model
    model = LSTMClassifier(vocab_size=len(vocab))
    model.train_model(train_loader=train_loader)

    # Save model
    model.save(path="./model.pth")


def evaluate_model():
    # Load test dataset
    test_texts, test_labels, _, _ = load_20news_data(directory=test_dir, num_workers=8)

    # Load the trained model
    model = LSTMClassifier.from_pretrained(path="./model.pth")

    # Preprocess the test data with the loaded training vocabulary
    # Load the same vocabulary used during training
    vocab = load_vocab(path="./vocab.json")
    test_tokens = tokenize_text(texts=clean_text(test_texts))
    test_sequences = encode_texts(
        tokenized_texts=test_tokens, vocab=vocab, max_length=200
    )

    # Create Dataset and DataLoader for test data
    test_dataset = NewsDataset(encoded_texts=test_sequences, labels=test_labels)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

    # Set model to eval mode
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Collect predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"[INFO] Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    train_and_save_model()  # First, we train and save the model
    evaluate_model()  # Then, we evaluate it using the test set
