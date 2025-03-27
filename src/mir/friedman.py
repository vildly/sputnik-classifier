import os
from pymongo import MongoClient
from dotenv import load_dotenv
from scipy.stats import friedmanchisquare

# Load environment variables
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")

def fetch_and_prepare_data():
    """Fetch relevant data from MongoDB and prepare it for statistical testing."""
    try:
        # Connect to MongoDB
        client = MongoClient(MONGODB_URI)
        db = client["results"]  # Access the `results` database
        results_collection = db["v1"]  # Access the `v1` collection

        # Data structure to store F1 Scores for each model and category
        data = {}  # Format: {model_name: {category: [f1_scores]}}

        # Fetch each document in the 'v1' collection
        documents = results_collection.find({})
        for doc in documents:
            models = doc.get("models", {})
            for model_name, model_data in models.items():
                print(f"Processing model: {model_name}")

                classification_report = model_data.get("classification_report", {})
                if not isinstance(classification_report, dict):
                    print(f"WARNING: No classification_report found for model: {model_name}")
                    continue

                # Initialize the model in the data dictionary if not present
                if model_name not in data:
                    data[model_name] = {}

                for category, metrics in classification_report.items():
                    # Check if metrics are structured correctly
                    if not isinstance(metrics, dict):
                        print(f"WARNING: Skipping category: {category} in model: {model_name}, type: {type(metrics)}")
                        continue

                    # Normalize keys to lowercase to handle inconsistent capitalization
                    metrics = {k.lower(): v for k, v in metrics.items()}

                    f1_score = metrics.get("f1-score")  # Lowercase "f1 score"
                    if f1_score is not None:
                        if category not in data[model_name]:
                            data[model_name][category] = []
                        data[model_name][category].append(f1_score)
                        print(f"  Category: {category}, F1 Score: {f1_score}")
                    else:
                        print(f"  WARNING: No F1 Score found for category: {category}, model: {model_name}")

        # Close the connection
        client.close()

        return data

    except Exception as e:
        print(f"An error occurred: {e}")
        return {}

def run_friedman_test(data):
    """Run Friedman's test on the prepared data."""
    print("\nRunning Friedman's test...")

    # Prepare data for each category
    friedman_results = {}
    for category in data.get(next(iter(data)), {}):  # Iterate over categories from the first model
        print(f"\nProcessing category: {category}")

        # Collect F1 Scores across all models for this category
        scores = []
        for model, categories in data.items():
            if category in categories:
                scores.append(categories[category])  # Collect all scores for the category
            else:
                print(f"WARNING: Missing data for category {category} in model {model}")
        
        if len(scores) >= 2:
            # Check if scores length matches across models
            try:
                stat, p = friedmanchisquare(*scores)
                friedman_results[category] = {"statistic": stat, "p-value": p}
                print(f"  Friedman statistic: {stat}, p-value: {p}")
            except ValueError as e:
                print(f"ERROR: Inconsistent score lengths for category {category}: {e}")
        else:
            print(f"  WARNING: Not enough models to run Friedman's test for {category}")

    return friedman_results

def main():
    # Fetch the data
    data = fetch_and_prepare_data()

    if data:
        print("\nStructured Data for Models and Categories:")
        for model, categories in data.items():
            print(f"\nModel: {model}")
            for category, f1_scores in categories.items():
                print(f"  {category}: {f1_scores}")

        # Run Friedman's test
        friedman_results = run_friedman_test(data)
        
        # Display test results
        print("\nFriedman's Test Results:")
        for category, result in friedman_results.items():
            print(f"Category: {category}, Statistic: {result['statistic']}, p-value: {result['p-value']}")

    else:
        print("No data fetched or an error occurred.")

if __name__ == "__main__":
    main()
