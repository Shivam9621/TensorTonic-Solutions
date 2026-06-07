import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    positions = np.arange(seq_len).reshape(-1, 1)

    even_idx = np.arange(0, d_model, 2)
    div_term = np.power(base, even_idx / d_model)

    angles = positions / div_term

    pe = np.zeros((seq_len, d_model))

    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles[:, :d_model // 2])

    return pe