# svm.py
#
# Partially used guide:
# https://codemax.app/snippet/introduction-to-text-classification-with-support-vector-machines-svm/
from typing import Dict, Union
from sampler import load_20newsdata, clean_texts, sample_data
from typing import Any
import os
import nltk
from mpmath import mp
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from db import add_one, connect_to_database, get_collection
from dotenv import load_dotenv


class CLF_svm:
    def __init__(self) -> None:
        # Adding random_state and probability=True ensures reproducibility
        # but slows down the training process.
        self.clf = SVC(kernel="linear", class_weight="balanced")
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


if __name__ == "__main__":
    load_dotenv()
    connect_to_database(connection_string=os.getenv("MONGODB_URI"))

    nltk.download(info_or_id="punkt", quiet=True)
    nltk.download(info_or_id="punkt_tab", quiet=True)
    nltk.download(info_or_id="stopwords", quiet=True)

    print("Loading 20 news data")
    train_data: np.ndarray
    train_labels: np.ndarray
    train_data, train_labels = load_20newsdata(paths=["./data/20news-bydate-train"])
    test_data: np.ndarray
    test_labels: np.ndarray
    test_data, test_labels = load_20newsdata(paths=["./data/20news-bydate-test"])

    print("Combining datasets")
    combined_data: np.ndarray = np.concatenate((train_data, test_data), axis=0)
    combined_labels: np.ndarray = np.concatenate((train_labels, test_labels), axis=0)

    print("Sample dataset")
    sampled_data: np.ndarray
    sampled_labels: np.ndarray
    # Setting size to None ensures that the entire dataset is sampled
    sampled_data, sampled_labels = sample_data(data=combined_data, labels=combined_labels, size=None)

    print("Cleaning combined data")
    cleaned_sampled_data = clean_texts(texts=sampled_data, lang="english")

    k = 10
    # Ensures that pi has enough decimal places for the seed
    mp.dps = 3 * k + 2
    # Get the required sequence of decimals
    pi_str = str(mp.pi)[2 : 3 * k + 2]
    accuracies = np.zeros(k)
    clf_reports = np.empty(k, dtype=object)
    conf_matrices = np.empty(k, dtype=object)
    # Matrix with k rows and num_classes as columns
    # This is because we store the scores for each class
    used_labels = np.unique(sampled_labels)
    labels_len = used_labels.size
    precisions = np.zeros((k, labels_len))
    recalls = np.zeros((k, labels_len))
    f1s = np.zeros((k, labels_len))
    supports = np.zeros((k, labels_len))

    # Define a type alias for clarity (optional but helpful)
    # This represents the structure of your 'records' dictionary plus the other string values
    SvmIterationData = Dict[str, Union[str, float, int, dict]]  # Adjust types as needed within records
    SvmModelData = Dict[str, Union[str, SvmIterationData]]  # Allows model_id (str) or iteration records (dict)

    # Initialize with type hints telling Pyright what to expect
    data_to_add: Dict[str, Dict[str, SvmModelData]] = {
        "models": {
            "svm": {
                "model_id": "svm"
                # Classification reports (SvmIterationData) will be added here
            }
        }
    }

    print("Starting cross-validation")
    print("This may take a very long time (depending on the amount of data)...")
    for i in tqdm(iterable=range(k), desc="Cross-validation", total=k, unit="folds"):
        # Gets the next 3 decimal places of pi as the seed
        seed = int(pi_str[i * 3 : i * 3 + 3])

        # NOTE:
        # Using stratify here ensures that the labels are distributed evenly
        # between the training and test sets
        # This is important for very small datasets!
        X_train, X_test, y_train, y_test = train_test_split(
            cleaned_sampled_data, sampled_labels, test_size=0.05, random_state=seed, stratify=sampled_labels
        )

        # NOTE:
        # Commented this out so that the database is not filled with duplicates
        # if i == 0:
        #     # Add the test data to the database
        #     data_col = get_collection(db="data", collection="v1")
        #     # --- Create the formatted list of articles ---
        #     # Use zip to pair articles from X_test with their labels from y_test
        #     formatted_articles = [
        #         {
        #             "user_input": article,  # The text from X_test
        #             "reference": label,  # The corresponding label from y_test
        #             "agent_response": "",  # The required empty string
        #         }
        #         for article, label in zip(X_test, y_test)  # Iterate through pairs
        #     ]
        #     db.add_one(
        #         col=data_col,
        #         data={
        #             "seed": seed,
        #             "categories:": used_labels.tolist(),
        #             "articles": formatted_articles,
        #         },
        #     )

        clf_svm = CLF_svm()
        clf_svm.fit(X_train, y_train)

        y_pred = clf_svm.predict(X=X_test)

        accuracies[i] = accuracy_score(y_true=y_test, y_pred=y_pred)
        clf_reports[i] = classification_report(y_true=y_test, y_pred=y_pred)
        precisions[i], recalls[i], f1s[i], supports[i] = precision_recall_fscore_support(
            y_true=y_test, y_pred=y_pred, average=None, labels=used_labels, zero_division=0
        )

        # Setting labels here ensures that the confusion matrix is always the same
        # dimensions across all iterations
        conf_matrices[i] = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=used_labels)

        # Add results to database
        records = (
            pd.DataFrame(
                {
                    "Label": used_labels,
                    "precision": precisions[i],
                    "recall": recalls[i],
                    "f1-score": f1s[i],
                    "support": supports[i],
                }
            )
            .set_index("Label")
            .to_dict("index")
        )
        records["seed"] = seed
        records["accuracy"] = accuracies[i]
        records["macro_avg"] = {
            "precision": np.mean(precisions[i]),
            "recall": np.mean(recalls[i]),
            "f1-score": np.mean(f1s[i]),
            "support": np.sum(supports[i]),
        }
        records["weighted_avg"] = {
            "precision": np.sum(precisions[i] * supports[i]) / np.sum(supports[i]),
            "recall": np.sum(recalls[i] * supports[i]) / np.sum(supports[i]),
            "f1-score": np.sum(f1s[i] * supports[i]) / np.sum(supports[i]),
            "support": np.sum(supports[i]),
        }
        # Add the 'records' for the current iteration to the 'svm' dictionary
        # using a unique key for the iteration
        data_to_add["models"]["svm"][f"classification_report_iter_{i}"] = records

    # Add the data to the database
    res_col = get_collection(db="results", collection="v1")
    add_one(col=res_col, data=data_to_add)
