import json

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def load_json(file_path) -> dict:
    """
    Loads the JSON data from the given file path and returns it as a dictionary.
    """
    # Read the JSON data from the input file
    with open(file=file_path, mode="r", encoding="utf-8") as infile:
        return json.load(infile)


def write_json(data, file_path) -> None:
    """
    Writes the data to the given file path. The data is written as a JSON.
    """
    # Write the cleaned data to the output file
    with open(file=file_path, mode="w", encoding="utf-8") as outfile:
        json.dump(data, outfile, indent=2, ensure_ascii=False)
    print(f"Wrote to {file_path}")


def remove_stop_words(text: str, spacer: str = "", language: str = "english") -> str:
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


def filter_data(in_file_path: str, out_file_path: str, language: str = "english") -> None:
    """
    Use this script to clean data from a JSON file. The input file should
    contain a list of objects, each with a "title" and "description" key.
    The output file will contain a list of objects with only the "title" and
    "description" keys.
    """
    data = load_json(in_file_path)

    # Create a new list with only the title and description for each element
    # Get the set of English stop words
    filtered = []

    for item in data:
        # Using .get() so that if the key is missing, it defaults to an empty
        # string
        title = item.get("title") or ""
        description = item.get("description") or ""

        # Convert to lowercase
        title = title.lower()
        description = description.lower()

        # Remove stop words
        title = remove_stop_words(
            text=title,
            spacer=" ",
            language=language
        )
        description = remove_stop_words(
            text=description,
            spacer=" ",
            language=language
        )

        # TODO: Concatenate all keys to one key to reduce tokens?
        # TODO: Filter out empty strings?

        filtered.append({
            "title": title,
            "description": description
        })

    write_json(filtered, out_file_path)
