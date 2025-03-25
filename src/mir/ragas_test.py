import pandas as pd
from ragas import evaluate, EvaluationDataset
from ragas.metrics import FactualCorrectness, SemanticSimilarity
from ragas.metrics import NonLLMStringSimilarity, BleuScore, RougeScore, ExactMatch, StringPresence
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from dotenv import load_dotenv
from pathlib import Path
import datetime
from pylo import get_logger

load_dotenv()
logger = get_logger()
# Environment variables for other things (if needed)
RAGAS_APP_TOKEN = os.getenv("RAGAS_APP_TOKEN")

# Initialize LLM and Embeddings wrappers for evaluation
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())


def evaluate_data(mongo_data):
    """
    Evaluate test cases using the responses from OpenRouter.

    Each document in mongo_data is expected to have an "items" key which
    is a list of dictionaries structured as follows:

        {
          "feature": "work-order description",
          "target": "ground truth category",
          "response": "the answer from OpenRouter"
        }

    Args:
        mongo_data (list): List of documents (dictionaries) from MongoDB.

    Returns:
        pd.DataFrame: DataFrame containing the original data and evaluation metrics.
    """

    results = []
    for document in mongo_data:
        # Extract the list of test cases; adjust key names as needed.

        query = document.get("user_input")
        ground_truth = document.get("reference")
        response = document.get("agent_response")

        if query is None:
            logger.warning("Test case missing 'feature' key. Skipping entry.")
            continue

        results.append(
            {
                "user_input": query,  # The work-order description
                "reference": ground_truth,  # Ground truth category
                "agent_response": response,  # Response from OpenRouter
            }
        )

    results_df = pd.DataFrame(results)

    # Prepare DataFrame for RAGAS Evaluation:
    # RAGAS expects columns for the question, reference, and response.
    ragas_data = results_df.copy()
    ragas_data = ragas_data.rename(columns={"user_input": "question"})
    ragas_data["response"] = results_df["agent_response"]

    # Ensure that 'reference' exists
    if "reference" not in ragas_data.columns:
        logger.warning("'reference' column not found. Adding a placeholder.")
        ragas_data["reference"] = None

    # Create the evaluation dataset from the DataFrame
    eval_dataset = EvaluationDataset.from_pandas(ragas_data)

    # Define the evaluation metrics
    metrics = [
        FactualCorrectness(llm=evaluator_llm),
        SemanticSimilarity(embeddings=evaluator_embeddings),
        NonLLMStringSimilarity(),
        BleuScore(),
        RougeScore(rouge_type="rouge1"),  # Example for unigrams
        ExactMatch(),
        StringPresence(),
    ]

    # Perform the evaluation using RAGAS
    ragas_results = evaluate(eval_dataset, metrics, llm=evaluator_llm)

    # Optionally upload the results if needed for your RAGAS setup
    ragas_results.upload()
    logger.debug(ragas_results)
    # Add the RAGAS metrics to the results DataFrame
    for metric_name, scores in ragas_results.to_pandas().items():
        if metric_name != "hash":
            results_df[metric_name] = scores

    # Optionally, save results in a timestamped output directory
    cwd = Path(__file__).parent.resolve()
    output_dir = cwd.joinpath("output")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_dir = output_dir.joinpath(f"ragas_{timestamp}")
    timestamped_dir.mkdir(exist_ok=True)

    results_csv_path = timestamped_dir.joinpath("test_results.csv")
    metrics_csv_path = timestamped_dir.joinpath("metrics_summary.csv")
    results_df.to_csv(results_csv_path, index=False)

    metrics_df = ragas_results.to_pandas()
    metrics_df.to_csv(metrics_csv_path, index=False)

    logger.info(f"Test results saved to: {results_csv_path}")
    logger.info(f"Metrics summary saved to: {metrics_csv_path}")
    logger.debug("\nRAGAS Results:")
    logger.debug(ragas_results)

    return str(ragas_results), ragas_results.to_pandas().to_json(orient="records")


if __name__ == "__main__":
    # This block is for standalone testing of this module.
    logger.info("ragas_module loaded. Call evaluate_data(mongo_data) from your main module.")
