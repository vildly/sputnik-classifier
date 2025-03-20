# svm.py
#
# Partially used guide:
# https://codemax.app/snippet/introduction-to-text-classification-with-support-vector-machines-svm/
from typing import Dict, Generator, Set, Tuple, List, Union
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import save_npz, load_npz
import joblib


def _read_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        return file.read()


def load_20newsdata(path: str, max: Union[int, None] = None) -> Tuple[np.ndarray, np.ndarray]:
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
                if max is not None and file_count >= max:
                    break
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
        results: List[str] = list(tqdm(iterable=executor.map(_read_file, file_paths), desc="Reading files", total=file_count, unit="files"))

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


def train_pipeline(
    model_path: str, X_train_path: str, X_test_path: str, y_train_path: str, y_test_path: str, ignore_existing: bool = False
) -> None:
    # Check if file extensions are valid
    if not all(map(lambda path: path.endswith(".npz"), [X_train_path, X_test_path])):
        raise ValueError("Invalid file extensions. X_train and X_test should have .npz extensions.")

    if not all(map(lambda path: path.endswith(".npy"), [y_train_path, y_test_path])):
        raise ValueError("Invalid file extensions. y_train and y_test should have .npy extensions.")

    if not model_path.endswith(".pkl"):
        raise ValueError("Invalid file extension. Model path should have a .pkl extension.")

    # Check if existing files should be ignored
    files_exist = os.path.exists(model_path) and all(
        map(os.path.exists, [X_train_path, X_test_path, y_train_path, y_test_path, model_path])
    )

    if files_exist and not ignore_existing:
        print("All files exist and are not being ignored. Skipping all.")
        return

    if (
        os.path.exists(X_train_path)
        and os.path.exists(X_test_path)
        and os.path.exists(y_train_path)
        and os.path.exists(y_test_path)
        and not ignore_existing
    ):
        print("Data files exist and ignore_existing is False. Skipping data generation.")
        return
    else:
        nltk.download(info_or_id="punkt", quiet=True)
        nltk.download(info_or_id="stopwords", quiet=True)

        # Load train and test datasets
        train_data: np.ndarray
        train_labels: np.ndarray
        train_data, train_labels = load_20newsdata("./data/20news-bydate-train")

        test_data: np.ndarray
        test_labels: np.ndarray
        test_data, test_labels = load_20newsdata("./data/20news-bydate-test")

        # Combine the datasets
        print("Combining datasets")
        combined_data: np.ndarray = np.concatenate((train_data, test_data), axis=0)
        combined_labels: np.ndarray = np.concatenate((train_labels, test_labels), axis=0)

        # Clean the text in DataFrame
        clean_combined_data = clean_texts(combined_data)

        # Vectorize the text data
        # max_features (maximum number of features to keep)
        max_features = int(np.sqrt(len(clean_combined_data) * (len(clean_combined_data[0]) * 0.2)))
        # max_df (ignore terms that appear in more than 80% of the documents)
        max_df = 0.8
        print(f"Vectorizing text data (max_features: {max_features}, max_df: {max_df})")
        print("This may take a while...")
        vectorizer = TfidfVectorizer(max_features=max_features, max_df=max_df)
        vector_data = vectorizer.fit_transform(clean_combined_data)

        # Set the precision for pi (decimal places)
        mp.dps = 50
        seed = int(str(mp.pi)[2:5])
        print("Splitting data")
        print("This may take a while...")
        X_train, X_test, y_train, y_test = train_test_split(vector_data, combined_labels, test_size=0.2, random_state=seed)

        print("Saving data")
        save_npz(file=X_train_path, matrix=X_train)
        save_npz(file=X_test_path, matrix=X_test)
        np.save(file=y_train_path, arr=y_train)
        np.save(file=y_test_path, arr=y_test)

        if os.path.exists(model_path) and not ignore_existing:
            print("Model already exists and ignore_existing is False. Skipping training.")

        print("Training model")
        print("This may take a very long time (depending on the amount of data)...")
        clf_svm = SVC(kernel="linear", random_state=seed, probability=True)
        clf_svm.fit(X_train, y_train)

        print("Saving model")
        joblib.dump(value=clf_svm, filename=model_path)


if __name__ == "__main__":
    model_path = "svm_model.pkl"
    X_train_path = "X_train.npz"
    X_test_path = "X_test.npz"
    y_train_path = "y_train.npy"
    y_test_path = "y_test.npy"

    train_pipeline(
        model_path=model_path,
        X_train_path=X_train_path,
        X_test_path=X_test_path,
        y_train_path=y_train_path,
        y_test_path=y_test_path,
        ignore_existing=False,
    )

    clf_svm = joblib.load(filename=model_path)
    X_train = load_npz(file="X_train.npz")
    X_test = load_npz(file="X_test.npz")
    y_train = np.load(file="y_train.npy", allow_pickle=True)
    y_test = np.load(file="y_test.npy", allow_pickle=True)

    # Predict and evaluate the model on test data
    print("Evaluating model")
    print("This may take a while...")
    train_pred = clf_svm.predict(X_train)
    test_pred = clf_svm.predict(X_test)

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
    unique_labels = [str(label) for label in np.unique(y_test)]

    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
