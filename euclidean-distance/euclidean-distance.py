import numpy as np

def euclidean_distance(x, y):
    return float(np.sqrt(np.sum((np.array(x) - np.array(y))**2)))