import numpy as np

def geometric_pmf_mean(k, p):
    """
    Compute Geometric PMF and Mean.
    """

    k = np.array(k)

    # PMF: P(X = k) = (1-p)^(k-1) * p
    pmf = ((1 - p) ** (k - 1)) * p

    # Mean: E[X] = 1/p
    mean = 1 / p

    return pmf, mean