import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Initialize vector with zeros
    bow = np.zeros(len(vocab), dtype=int)
    
    # Create a mapping from word → index
    vocab_index = {word: i for i, word in enumerate(vocab)}
    
    # Count occurrences
    for token in tokens:
        if token in vocab_index:
            bow[vocab_index[token]] += 1
    
    return bow