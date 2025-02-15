from clean import filter_data

import os
import nltk

# Important as NLTK searches in these directories:
# - $HOME/nltk_data
# - $HOME/<path-to-pro>/.venv/nltk_data
# - $HOME/<path-to-pro>/.venv/share/nltk_data
# - $HOME/<path-to-pro>/.venv/lib/nltk_data
# - /usr/share/nltk_data
# - /usr/local/share/nltk_data
# - /usr/lib/nltk_data
# - /usr/local/lib/nltk_data
download_dir = "./.venv/lib/nltk_data"
nltk.download("stopwords", download_dir=download_dir)
nltk.download("punkt_tab", download_dir=download_dir)
nltk.download("punkt", download_dir=download_dir)


if __name__ == "__main__":
    # Specify your input and output file names
    input_file = ".data/wo_data.json"
    output_file = ".data/cleaned_wo_data.json"

    # Ensures that the output directory exists
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filter_data(
        in_file_path=input_file,
        out_file_path=output_file,
        language="swedish"
    )
