# svm.py
#
# Partially used guide:
# https://developer.ibm.com/tutorials/awb-classifying-data-svm-algorithm-python/
from typing import Dict, Generator, Set, Tuple, List
from string import punctuation
import os
import concurrent.futures
import nltk
from mpmath import mp
import numpy as np
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
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

    print("Initializing files...")
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
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results: List[str] = list(tqdm(iterable=executor.map(read_file, file_paths), desc="Reading files", total=file_count, unit="files"))

    # Convert results to numpy array
    data: np.ndarray = np.array(results, dtype=object)
    labels: np.ndarray = np.array(file_labels, dtype=object)

    print(f"Loaded {file_count} files")
    return data, labels


def clean_texts(texts: np.ndarray, lang: str = "english") -> np.ndarray:
    print("Cleaning texts")
    print("This may take a while...")
    cleaned_texts: np.ndarray = np.empty_like(texts, dtype=object)

    for ix, text in tqdm(iterable=enumerate(texts), desc="Cleaning texts", total=len(texts), unit="texts"):
        text = text.lower()
        # Remove punctuation
        translator: Dict[int, int | None] = str.maketrans("", "", punctuation)
        text = text.translate(translator)

        # Remove stopwords
        words: List[str] = nltk.word_tokenize(text)
        stop_words: Set[str] = set(nltk.corpus.stopwords.words(lang))
        filtered_words: Generator[str, None, None] = (word for word in words if word not in stop_words)

        cleaned_texts[ix] = " ".join(filtered_words)

    print(f"{len(cleaned_texts)} texts cleaned")
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

    # Combine the datasets
    print("Combining datasets")
    combined_data: np.ndarray = np.concatenate((train_data, test_data), axis=0)
    combined_labels: np.ndarray = np.concatenate((train_labels, test_labels), axis=0)

    # Clean the text in DataFrame
    clean_combined_data = clean_texts(combined_data)

    # Set the precision for pi (decimal places)
    mp.dps = 50
    seed = float(str(mp.pi)[2:5])

    # Split data using the seed
    print("Splitting data")
    X_train, X_test, y_train, y_test = train_test_split(clean_combined_data, combined_labels, test_size=0.2, random_state=seed)

    # Use CountVectorizer to find the vocabulary size
    print("Calculating vocabulary size to max_features")
    count_vectorizer = CountVectorizer()
    count_vectorizer.fit(clean_combined_data)
    vocabulary_size = len(count_vectorizer.vocabulary_)

    # Calculate max_features based on the square root of the vocabulary size
    max_features = int(np.sqrt(vocabulary_size))

    # Vectorize text data
    print("Vectorizing text data")
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Initialize and train the SVM classifier
    clf_svm = SVC(kernel="linear", random_state=seed, probability=True)
    clf_svm.fit(X_train_tfidf, y_train)

    # Predict and evaluate the model on test data
    print("Evaluating model")
    print("This may take a while...")
    train_pred = clf_svm.predict(X_train_tfidf)
    test_pred = clf_svm.predict(X_test_tfidf)

    # Print training and test set performance
    train_accuracy = accuracy_score(y_train, train_pred)
    print("Training Accuracy:", train_accuracy)
    test_accuracy = accuracy_score(y_test, test_pred)
    print("Test Accuracy:", test_accuracy)

    # Detailed report
    print("Classification report on test data")
    print(classification_report(y_test, test_pred))

    # Confusion Matrix and Seaborn Heatmap
    conf_matrix = confusion_matrix(y_test, test_pred)
    plt.figure(figsize=(10, 8))

    # Convert labels to a list of strings
    unique_labels = [str(label) for label in np.unique(test_labels)]

    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
