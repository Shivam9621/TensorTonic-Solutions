import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """

    # Tokenize documents
    tokenized_docs = [doc.split() for doc in documents]

    # Build vocabulary (sorted)
    vocabulary = sorted(set(word
                            for doc in tokenized_docs
                            for word in doc))

    N = len(documents)

    # Document Frequency
    df = {}
    for word in vocabulary:
        df[word] = sum(1 for doc in tokenized_docs if word in doc)

    # IDF
    idf = {}
    for word in vocabulary:
        idf[word] = math.log(N / df[word])

    # TF-IDF Matrix
    tfidf_matrix = np.zeros((N, len(vocabulary)))

    for i, doc in enumerate(tokenized_docs):
        counts = Counter(doc)
        total_terms = len(doc)

        for j, word in enumerate(vocabulary):
            tf = counts[word] / total_terms if total_terms > 0 else 0
            tfidf_matrix[i][j] = tf * idf[word]

    return tfidf_matrix, vocabulary