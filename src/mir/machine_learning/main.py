from load_dataset import load_20news_data
from preprocess import clean_text, tokenize_text
from encode import build_vocab, encode_texts
from news_dataset import NewsDataset
from torch.utils.data import DataLoader
from lstm_model import LSTMClassifier, train_model

# ✅ Loaded the dataset
# ✅ Cleaned & tokenized the text
# ✅ Built a vocabulary & encoded text into sequences
# ✅ Created PyTorch Dataset & DataLoader objects
#
# 🚀 The next step is to:
# Define & Train an LSTM Model

# Load train and test data with optimizations
train_dir = "./../data/20news-bydate-train"
test_dir = "./../data/20news-bydate-test"

train_texts, train_labels, categories = load_20news_data(train_dir, num_workers=8)
test_texts, test_labels, _ = load_20news_data(test_dir, num_workers=8)

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
