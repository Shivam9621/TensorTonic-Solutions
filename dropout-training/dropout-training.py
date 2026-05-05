import numpy as np

def dropout(x, p=0.5, rng=None):
    x = np.array(x)

    if rng is None:
        rand = np.random.random(x.shape)
    else:
        rand = rng.random(x.shape)

    mask = (rand >= p).astype(float)

    # scale factor
    scale = 1 / (1 - p)

    output = x * mask * scale
    dropout_pattern = mask * scale   # 🔥 important fix

    return output, dropout_pattern