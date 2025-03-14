from load_dataset import load_20news_data
from preprocess import clean_text, tokenize_text

# Load train and test data with optimizations
train_dir = "./data/20news-bydate-train"
test_dir = "./data/20news-bydate-test"

train_texts, train_labels, categories = load_20news_data(train_dir, num_workers=8)
test_texts, test_labels, _ = load_20news_data(test_dir, num_workers=8)

print(f"[INFO] Category Mapping: {categories}")


# Clean and tokenize the training data
train_tokens = tokenize_text(train_texts)
test_tokens = tokenize_text(test_texts)

print("[INFO] Sample cleaned and tokenized document:", train_tokens[0])
