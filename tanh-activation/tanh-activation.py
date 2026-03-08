import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    x = np.array(x)  # convert input to numpy array
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))