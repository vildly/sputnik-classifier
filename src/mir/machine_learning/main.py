from load_dataset import load_20news_data, save_news_data_to_json
from preprocess import clean_text, tokenize_text
from encode import build_vocab, encode_texts
from news_dataset import NewsDataset
from torch.utils.data import DataLoader
from lstm_model import LSTMClassifier, train_model

# ✅ Loaded the dataset
# ✅ Cleaned & tokenized the text
# ✅ Built a vocabulary & encoded text into sequences
# ✅ Created PyTorch Dataset & DataLoader objects
# ✅ Defined an LSTM Model & Training Loop
#
# 🚀 The next step is to:
# Evaluate the model on the test set

train_dir = "./../data/20news-bydate-train/"
test_dir = "./../data/20news-bydate-test/"

# Save a 100 random test documents to JSON file (pi based selection)
# save_news_data_to_json(
#     directory=test_dir, output_file="./../data/datav3_100.json", num_workers=8
# )

# Load training dataset
train_texts, train_labels, categories, _ = load_20news_data(train_dir, num_workers=8)

# Load test dataset independently
test_texts, test_labels, _, _ = load_20news_data(test_dir, num_workers=8)

print(f"[INFO] Category Mapping: {categories}")

# Clean and tokenize the training data
train_tokens = tokenize_text(clean_text(train_texts))
test_tokens = tokenize_text(clean_text(test_texts))

print("[INFO] Sample cleaned and tokenized document:", train_tokens[0][:10], "...")

# Apply Vocabulary + Encoding
vocab = build_vocab(train_tokens)
train_sequences = encode_texts(train_tokens, vocab)
test_sequences = encode_texts(test_tokens, vocab)

print("[INFO] Sample Encoded Text:", train_sequences[0][:10], "...")

# Create Dataset Objects
train_dataset = NewsDataset(train_sequences, train_labels)
test_dataset = NewsDataset(test_sequences, test_labels)

# Create DataLoaders for batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("[INFO] DataLoader Created! Ready for Model Training")

model = LSTMClassifier(vocab_size=len(vocab))
trained_model = train_model(model, vocab, train_loader)
