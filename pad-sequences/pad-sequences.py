import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
        N = len(seqs)
        L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """

    if len(seqs) == 0:
        return np.empty((0, 0), dtype=int)

    if max_len is None:
        max_len = max(len(seq) for seq in seqs)

    result = []

    for seq in seqs:
        if len(seq) > max_len:
            result.append(seq[:max_len])  # truncate
        else:
            result.append(seq + [pad_value] * (max_len - len(seq)))  # pad

    return np.array(result, dtype=int)