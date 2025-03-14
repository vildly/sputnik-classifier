from collections import Counter
import numpy as np
from typing import Dict


def build_vocab(tokenized_texts: np.ndarray, min_freq: int = 2) -> Dict[str, int]:
    """
    Creates a vocabulary mapping each unique word to an index.

    Args:
        tokenized_texts (np.ndarray): A NumPy array of tokenized documents.
        min_freq (int): The minimum frequency a word must have to be included in the vocabulary.

    Returns:
        Dict[str, int]: A dictionary mapping words to unique index values.
    """
    word_counter: Counter = Counter()
    for tokens in tokenized_texts:
        word_counter.update(tokens)

    # Only keep words that appear `min_freq` times or more
    vocab: Dict[str, int] = {
        word: idx + 1
        for idx, (word, freq) in enumerate(word_counter.items())
        if freq >= min_freq
    }
    vocab["<PAD>"] = 0  # Special padding token for sequence alignment

    print(f"[INFO] Vocabulary Size: {len(vocab)} words")
    return vocab


def encode_texts(
    tokenized_texts: np.ndarray, vocab: Dict[str, int], max_length: int = 100
) -> np.ndarray:
    """
    Converts a NumPy array of tokenized documents into sequences of word indices.

    Args:
        tokenized_texts (np.ndarray): Tokenized texts as a NumPy array.
        vocab (Dict[str, int]): The vocabulary mapping words to indices.
        max_length (int): The maximum sequence length (texts are padded/truncated to this length).

    Returns:
        np.ndarray: A NumPy array of shape (num_samples, max_length) containing encoded word indices.
    """
    encoded_texts = np.zeros((len(tokenized_texts), max_length), dtype=np.int32)

    for i, tokens in enumerate(tokenized_texts):
        encoded = [
            vocab.get(word, 0) for word in tokens[:max_length]
        ]  # Map words to indices
        encoded_texts[i, : len(encoded)] = (
            encoded  # Insert sequence without looping over each padding
        )

    return encoded_texts
