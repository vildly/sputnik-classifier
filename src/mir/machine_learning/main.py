import math
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from lstm import (
    create_test_set,
    load_vocab,
    load_20news_data,
    clean_text,
    tokenize_text,
    build_vocab,
    encode_texts,
    NewsDataset,
    LSTMClassifier,
)

train_dir = "./../data/20news-bydate-train/"
test_dir = "./../data/20news-bydate-test/"


def train_model():
    # Load and preprocess training data
    train_texts, train_labels, _, _ = load_20news_data(
        directory=train_dir, num_workers=8
    )

    # Preprocess the training data
    train_tokens = tokenize_text(texts=clean_text(train_texts), num_workers=8)
    # min_freq=1 to include all words
    vocab = build_vocab(tokenized_texts=train_tokens, min_freq=1)

    # Create Dataset and DataLoader
    train_sequences = encode_texts(
        tokenized_texts=train_tokens, vocab=vocab, max_length=200
    )
    train_dataset = NewsDataset(encoded_texts=train_sequences, labels=train_labels)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

    # Instantiate and train the model
    model = LSTMClassifier(vocab_size=len(vocab))
    model.train_model(train_loader=train_loader)


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
    # For the LLMs and then later the LSTM
    # create_test_set(
    #     directory=test_dir,
    #     num_workers=8,
    #     seed=int(str(math.pi)[2:9]),
    #     sample_size=100,
    # )

    train_model()
    evaluate_model()
