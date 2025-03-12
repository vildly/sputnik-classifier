import csv
import json

# Read CSV and convert to JSON format
csv_file = "./../../data/data.csv"
json_file = "./../../data/data.json"

data = {"items": []}  # Initialize the JSON structure

with open(csv_file, encoding="utf-8") as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip header row
    for row in csv_reader:
        feature, target = row
        data["items"].append({"feature": feature, "target": target})

# Save to a JSON file
with open(json_file, "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print(f"Converted {csv_file} to {json_file} successfully!")
