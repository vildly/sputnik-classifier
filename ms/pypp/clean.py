from typing import Dict, List

import json

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def _load_json(file_path) -> dict:
    """
    Loads the JSON data from the given file path and returns it as a dictionary.
    """
    # Read the JSON data from the input file
    with open(file=file_path, mode="r", encoding="utf-8") as infile:
        return json.load(infile)


def _write_json(data, file_path) -> None:
    """
    Writes the data to the given file path. The data is written as a JSON.
    """
    # Write the cleaned data to the output file
    with open(file=file_path, mode="w", encoding="utf-8") as outfile:
        json.dump(data, outfile, indent=2, ensure_ascii=False)
    print(f"Wrote to {file_path}")


def _remove_stop_words(text: str, spacer: str = "", language: str = "english") -> str:
    """
    Removes stop words from the given text. The text is tokenized and the
    stop words are removed. The text is then joined back together with the
    given spacer.
    """
    # Maybe move the stop_words as a global, but for now this works.
    #
    # Provides the stop words for the given language
    stop_words = set(stopwords.words(language))
    # Tokenizes the text
    tokens = word_tokenize(text=text, language=language)
    # Returns the text without the stop words
    return spacer.join([w for w in tokens if w.lower() not in stop_words])


def filter_data(in_file_path: str, out_file_path: str, keys: List[str], language: str = "english") -> None:
    """
    Use this script to clean (pre-process) data from a JSON file.
    The output file will contain a list of objects with only the keys provided.
    """
    if not len(in_file_path):
        raise Exception("Input file path missing")
    if not len(out_file_path):
        raise Exception("Output file path missing")
    if not len(keys):
        raise Exception("No keys provided")
    data = _load_json(in_file_path)

    # Create a new list with only the title and description for each element
    # Get the set of English stop words
    filt_data: List[str] = []

    for data_obj in data:
        temp: List[str] = []
        for key in keys:
            # Using .get() so that if the key is missing, it defaults to None
            cur_obj = data_obj.get(key) or None
            if cur_obj is None:
                continue
            # Check if the current item is empty
            if not len(cur_obj):
                continue

            cur_obj = cur_obj.lower()

            cur_obj = _remove_stop_words(
                text=cur_obj,
                spacer=" ",
                language=language
            )

            # TODO: Concatenate all keys to one key to reduce tokens?
            # TODO: Filter out empty strings?
            
            # If the pre-processing has removed all content then
            # simply continue and skip adding the object to the list
            if not len(cur_obj):
                continue

            # Assign the new object the filtered key
            temp.append(cur_obj)

        # If we have at least one key with valid content, combine the parts.
        if temp:
            # Join the parts with a space and append to the new list.
            combined = " ".join(temp)
            filt_data.append(combined)

    _write_json(filt_data, out_file_path)
