import argparse
import os
import nltk

from clean import filter_data


def get_dependencies() -> None:
    # Important as NLTK searches in these directories:
    # - $HOME/nltk_data
    # - $HOME/<path-to-project>/.venv/nltk_data
    # - $HOME/<path-to-project>/.venv/share/nltk_data
    # - $HOME/<path-to-project>/.venv/lib/nltk_data
    # - /usr/share/nltk_data
    # - /usr/local/share/nltk_data
    # - /usr/lib/nltk_data
    # - /usr/local/lib/nltk_data
    download_dir = "./.venv/lib/nltk_data"
    nltk.download("stopwords", download_dir=download_dir)
    nltk.download("punkt_tab", download_dir=download_dir)
    nltk.download("punkt", download_dir=download_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("-o", type=str, required=True, help="Path to the output JSON file.")
    parser.add_argument("-keys", type=str, nargs="+", required=True, help="List of keys to keep (example: -keys: title description richText)")
    args = parser.parse_args()
    
    get_dependencies() 

    input_file = args.i 
    output_file = args.o
    keys = args.keys

    # Ensures that the output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filter_data(
        in_file_path=input_file,
        out_file_path=output_file,
        keys=keys,
        language="swedish"
    )
