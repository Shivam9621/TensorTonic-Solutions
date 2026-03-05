import numpy as np

def dropout(x, p, rng=None):
    x = np.array(x, dtype=float)
    
    if rng is not None:
        rand = rng.random(x.shape)
    else:
        rand = np.random.random(x.shape)
    
    keep_prob = 1 - p
    
    # Create dropout pattern
    dropout_pattern = (rand < keep_prob).astype(float) * (1 / keep_prob)
    
    # Apply dropout
    output = x * dropout_pattern
    
    return output, dropout_pattern