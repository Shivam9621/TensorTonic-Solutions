import numpy as np

def focal_loss(p, y, gamma=2.0):
    p = np.array(p, dtype=float)
    y = np.array(y, dtype=float)

    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)

    loss = -((1 - p) ** gamma) * y * np.log(p) \
           - (p ** gamma) * (1 - y) * np.log(1 - p)

    return np.mean(loss)