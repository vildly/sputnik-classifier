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
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


class CLF_svm:
    def __init__(self) -> None:
        # Adding random_state and probability=True ensures reproducibility
        # but slows down the training process.
        self.clf = SVC(kernel="linear")
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
    train_data, train_labels = load_20newsdata("./data/20news-bydate-train", max=5000)
    test_data: np.ndarray
    test_labels: np.ndarray
    test_data, test_labels = load_20newsdata("./data/20news-bydate-test", max=5000)

    print("Combining datasets")
    combined_data: np.ndarray = np.concatenate((train_data, test_data), axis=0)
    combined_labels: np.ndarray = np.concatenate((train_labels, test_labels), axis=0)

    print("Cleaning combined data")
    clean_combined_data = clean_texts(combined_data)

    k = 10
    # Ensures that pi has enough decimal places for the seed
    mp.dps = 3 * k + 2
    # Get the required sequence of decimals
    pi_str = str(mp.pi)[2 : 3 * k + 2]
    accuracies = np.zeros(k)
    clf_reports = np.empty(k, dtype=object)
    conf_matrices = np.empty(k, dtype=object)
    precision_scores = np.zeros((k, len(np.unique(combined_labels))))
    recall_scores = np.zeros((k, len(np.unique(combined_labels))))
    f1_scores = np.zeros((k, len(np.unique(combined_labels))))
    support_scores = np.zeros((k, len(np.unique(combined_labels))))

    for i in range(k):
        seed = int(pi_str[i * 3 : i * 3 + 3])  # Gets the next 3 decimal places of pi
        print(f"[k{i}] Seed: {seed}")

        print(f"[k{i}] Splitting data")
        print("This may take a moment depending on the amount of data...")
        X_train, X_test, y_train, y_test = train_test_split(clean_combined_data, combined_labels, test_size=0.2, random_state=seed)

        print(f"[k{i}] Training model")
        print("This may take a very long time (depending on the amount of data)...")
        clf_svm = CLF_svm()
        clf_svm.fit(X_train, y_train)

        print(f"[k{i}] Evaluating model")
        print("This may take a while...")
        test_pred = clf_svm.predict(X_test)

        print(f"[k{i}] Adding accuracy to list")
        accuracies[i] = accuracy_score(y_test, test_pred)

        print(f"[k{i}] Generating classification report")
        clf_reports[i] = classification_report(y_test, test_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_test, test_pred, average=None)
        precision_scores[i] = precision
        recall_scores[i] = recall
        f1_scores[i] = f1
        support_scores[i] = support

        print(f"[k{i}] Generating confusion matrix")
        conf_matrices[i] = confusion_matrix(y_test, test_pred)

    # Final evaluation
    # Aggregated classification report
    aggregated_precision = np.mean(precision_scores, axis=0)
    aggregated_recall = np.mean(recall_scores, axis=0)
    aggregated_f1 = np.mean(f1_scores, axis=0)
    aggregated_support = np.sum(support_scores, axis=0)

    print("Aggregate classification report:")
    for label_idx, label in enumerate(np.unique(combined_labels)):
        print(
            f"Label {label} - Precision: {aggregated_precision[label_idx]:.4f}\n"
            f"Recall: {aggregated_recall[label_idx]:.4f}\n"
            f"F1 Score: {aggregated_f1[label_idx]:.4f}\n"
            f"Support: {aggregated_support[label_idx]}"
        )

    # Combine all confusion matrices
    combined_conf_matrix = np.sum(np.array(conf_matrices), axis=0)

    plt.figure(figsize=(10, 8))

    # WARNING:
    # Using the combined labels here may not always work as expected
    # instead aim to save the y_test from each iteration to an array
    # and then extract each unique label from that.
    unique_labels = [str(label) for label in np.unique(combined_labels)]

    sns.heatmap(combined_conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title("Combined Confusion Matrix Across All Folds")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
