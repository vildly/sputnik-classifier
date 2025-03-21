# svm.py
#
# Partially used guide:
# https://codemax.app/snippet/introduction-to-text-classification-with-support-vector-machines-svm/
from typing import Dict, Generator, Set, Tuple, List, Union, Any
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


class CLF_svm:
    def __init__(self, seed: int) -> None:
        self.clf = SVC(kernel="linear", random_state=seed, probability=True)
        self.vectorizer = TfidfVectorizer()

    def fit(self, X: Any, y: Any) -> None:
        """
        Vectorizes and fits the model of the data.

        Parameters
        ----------
        X : Any -- Data to fit the model to. Accepted inputs are lists, numpy arrays, scipy-sparse matrices, or pandas dataframes.
        y : Any -- Labels for the data. Accepted inputs are lists, numpy arrays, scipy-sparse matrices, or pandas dataframes.
        """
        # Vectorize the text data
        # max_features (maximum number of features to keep)
        max_features = int(np.sqrt(len(X) * (len(X[0]) * 0.2)))
        # max_df (ignore terms that appear in more than 80% of the documents)
        max_df = 0.8
        self.vectorizer = TfidfVectorizer(max_features=max_features, max_df=max_df)
        self.clf.fit(self.vectorizer.fit_transform(X), y)

    def predict(self, X: Any) -> Any:
        """
        Vectorizes and predicts the labels of the data.

        Parameters
        ----------
        data : Any -- Data to predict labels for. Accepted inputs are lists, numpy arrays, scipy-sparse matrices, or pandas dataframes.

        Returns
        -------
        Any -- Predicted labels.
        """
        # Transform the data with the already fitted vectorizer and predict the labels
        return self.clf.predict(self.vectorizer.transform(X))


def _read_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        return file.read()


def load_20newsdata(path: str, max: Union[int, None] = None) -> Tuple[np.ndarray, np.ndarray]:
    print(f"Loading files from {path}")
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


if __name__ == "__main__":
    nltk.download("punkt")
    nltk.download("stopwords")

    print("Loading 20 news data")
    train_data: np.ndarray
    train_labels: np.ndarray
    train_data, train_labels = load_20newsdata("./data/20news-bydate-train", max=500)
    test_data: np.ndarray
    test_labels: np.ndarray
    test_data, test_labels = load_20newsdata("./data/20news-bydate-test", max=500)

    print("Combining datasets")
    combined_data: np.ndarray = np.concatenate((train_data, test_data), axis=0)
    combined_labels: np.ndarray = np.concatenate((train_labels, test_labels), axis=0)

    print("Cleaning combined data")
    clean_combined_data = clean_texts(combined_data)

    # TODO: 10 or more iterations here for cross-validation
    # Set the precision for pi (decimal places)
    mp.dps = 50  # TODO: Change this to dynamic value
    seed = int(str(mp.pi)[2:5])

    print("Splitting data")
    print("This may take a moment depending on the amount of data...")
    X_train, X_test, y_train, y_test = train_test_split(clean_combined_data, combined_labels, test_size=0.2, random_state=seed)

    print("Training model")
    print("This may take a very long time (depending on the amount of data)...")
    clf_svm = CLF_svm(seed=seed)
    clf_svm.fit(X_train, y_train)

    print("Evaluating model")
    print("This may take a while...")
    test_pred = clf_svm.predict(X_test)

    # Print training and test set performance
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
