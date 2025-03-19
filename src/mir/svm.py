# svm.py
#
# Partially used guide:
# https://developer.ibm.com/tutorials/awb-classifying-data-svm-algorithm-python/
from typing import Dict, Generator, Set, Tuple, List
from string import punctuation
import os
import concurrent.futures
import nltk
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


def read_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        return file.read()


def load_dir(path: str) -> Tuple[np.ndarray, np.ndarray]:
    print(f"Loading files from {path}")
    print("This may take a while...")
    initial_size: int = 20
    file_paths: np.ndarray = np.empty(initial_size, dtype=object)
    file_labels: np.ndarray = np.empty(initial_size, dtype=object)
    file_count: int = 0

    for root, dirs, _ in os.walk(path):
        for category in dirs:
            category_path: str = os.path.join(root, category)
            for filename in os.listdir(category_path):
                file_path: str = os.path.join(category_path, filename)
                if os.path.isfile(file_path):
                    if file_count >= len(file_paths):
                        # Double the size of arrays when limit is reached
                        file_paths = np.resize(file_paths, file_count * 2)
                        file_labels = np.resize(file_labels, file_count * 2)

                    file_paths[file_count] = file_path
                    file_labels[file_count] = category
                    file_count += 1

    print(f"Initialized {file_count} files")
    # Trim the arrays to the actual length
    file_paths = file_paths[:file_count]
    file_labels = file_labels[:file_count]

    # Use concurrent.futures to read files in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results: List[str] = list(executor.map(read_file, file_paths))

    # Convert results to numpy array
    data: np.ndarray = np.array(results, dtype=object)
    labels: np.ndarray = np.array(file_labels, dtype=object)

    print(f"Loaded {file_count} files")
    print("Returning as (data: np.ndarray, labels: np.ndarray)")
    return data, labels


def clean_texts(texts: np.ndarray, lang: str = "english") -> np.ndarray:
    print("Cleaning texts")
    print("This may take a while...")
    cleaned_texts: np.ndarray = np.empty_like(texts, dtype=object)

    for ix, text in enumerate(texts):
        text = text.lower()
        # Remove punctuation
        translator: Dict[int, int | None] = str.maketrans("", "", punctuation)
        text = text.translate(translator)

        # Remove stopwords
        words: List[str] = nltk.word_tokenize(text)
        stop_words: Set[str] = set(nltk.corpus.stopwords.words(lang))
        filtered_words: Generator[str, None, None] = (word for word in words if word not in stop_words)

        cleaned_texts[ix] = " ".join(filtered_words)

    print("Texts cleaned")
    return cleaned_texts


if __name__ == "__main__":
    nltk.download("punkt")
    nltk.download("stopwords")

    # Load train and test datasets
    train_data: np.ndarray
    train_labels: np.ndarray
    train_data, train_labels = load_dir("./data/20news-bydate-train")

    test_data: np.ndarray
    test_labels: np.ndarray
    test_data, test_labels = load_dir("./data/20news-bydate-test")

    # Clean the text in DataFrame
    cleaned_train_data = clean_texts(train_data)
    cleaned_test_data = clean_texts(test_data)

    # Create a DataFrame
    train_df: pd.DataFrame = pd.DataFrame(data={"text": cleaned_train_data, "label": train_labels})
    print(train_df.head())
    test_df: pd.DataFrame = pd.DataFrame(data={"text": cleaned_test_data, "label": test_labels})
    print(test_df.head())

    # Vectorize text data
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_tfidf = vectorizer.fit_transform(cleaned_train_data)
    X_test_tfidf = vectorizer.transform(cleaned_test_data)

    # Initialize and train the SVM classifier
    clf_svm = SVC(kernel="linear", random_state=42)
    clf_svm.fit(X_train_tfidf, train_labels)

    # Predict and evaluate the model on test data
    train_predictions = clf_svm.predict(X_train_tfidf)
    test_predictions = clf_svm.predict(X_test_tfidf)

    # Print training and test set performance
    print("Training Accuracy:", accuracy_score(train_labels, train_predictions))
    print("Test Accuracy:", accuracy_score(test_labels, test_predictions))

    # Detailed report
    print("Classification Report on Test Data:")
    print(classification_report(test_labels, test_predictions))

    # Confusion Matrix and Seaborn Heatmap
    conf_matrix = confusion_matrix(test_labels, test_predictions)
    plt.figure(figsize=(10, 8))

    # Convert labels to a list of strings
    unique_labels = [str(label) for label in np.unique(test_labels)]

    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
