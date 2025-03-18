# process_results.py
import json
import re
import os
import asyncio
from dotenv import load_dotenv

from ragas_test import evaluate_data

from db import (
    add_one,
    connect_to_database,
    find_by_id,
    get_collection,
)

load_dotenv()

def strip_markdown_fences(text):
    """Remove Markdown code fences from text."""
    text = text.strip()
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    return text

async def process_and_store_results(model_keys: list):
    """Process all job entries for specified models and store the results."""
    jobs_col = get_collection(db="jobs", collection="v1")
    results_col = get_collection(db="results", collection="v1")
    data_col = get_collection(db="data", collection="v1")

    # Fetch all job documents
    job_documents = jobs_col.find({})  # Fetch all jobs that include the specified models

    for job in job_documents:
        job_id = job['_id']
        data_id = job.get('data_id')

        print(f"Processing job ID: {job_id} for models: {model_keys}")

        data_doc = find_by_id(col=data_col, doc_id=data_id)
        if not data_doc:
            print(f"Data document not found for data_id: {data_id}")
            continue

        articles = data_doc.get("articles", [])
        results_structure = {"models": {}}

        for model_key in model_keys:
            print(f"Processing model: {model_key} for job ID: {job_id}")

            # Instead of next(), collect all entries that match this model_key
            model_results_list = [
                entry.get(model_key) for entry in job.get("model", []) if model_key in entry
            ]

            if not model_results_list:
                print(f"No results found for model_key: {model_key} in job ID: {job_id}")
                continue

            all_results = []
            # Process each batch for the same model and merge the results
            for model_result in model_results_list:
                content = model_result.get("choices", [{}])[0].get("message", {}).get("content", "")
                try:
                    parsed_content = json.loads(strip_markdown_fences(content))
                    print(f"Parsed content for job ID {job_id}, model {model_key}: {parsed_content}")
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON content for job ID {job_id}, model {model_key}: {e}")
                    continue

                for key, category in parsed_content.items():
                    try:
                        index = int(key)
                    except ValueError:
                        print(f"Non-integer key encountered in job ID {job_id} for model {model_key}: {key}")
                        continue

                    if index < len(articles):
                        user_input = articles[index]["user_input"]
                        reference = articles[index]["reference"]
                        agent_response = category
                        
                        all_results.append({
                            "user_input": user_input,
                            "reference": reference,
                            "agent_response": agent_response
                        })
                    else:
                        print(f"Index {index} out of range for articles in job ID {job_id}")
            
            ragas_avg, ragas_metrics = evaluate_data(all_results)

            if all_results:
                # If there is already data for this model_id, merge with existing results
                if model_key in results_structure["models"]:
                    results_structure["models"][model_key]["results"].extend(all_results)
                else:
                    results_structure["models"][model_key] = {
                        "model_id": model_key,
                        "results": all_results,
                        "ragas_avg": ragas_avg,
                        "ragas_metrics": ragas_metrics
                    }

        if results_structure["models"]:
            add_result = add_one(col=results_col, data=results_structure)
            print(f"Stored results for job ID {job_id}: {add_result.inserted_id}")

async def main():
    connect_to_database(connection_string=os.getenv("MONGODB_URI"))

    # Specify the model keys you want to process
    model_keys = [
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemini-2.0-pro-exp-02-05:free"
    ]
    print(f"Starting processing for models: {model_keys}")
    await process_and_store_results(model_keys)

if __name__ == "__main__":
    asyncio.run(main())
