import numpy as np

def bag_of_words_vector(tokens, vocab):
    bow = np.zeros(len(vocab), dtype=int)

    for token in tokens:
        if token in vocab:
            bow[vocab.index(token)] += 1

    return bow