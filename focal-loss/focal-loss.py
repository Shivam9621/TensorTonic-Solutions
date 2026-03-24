import numpy as np

def focal_loss(p, y, gamma=2.0):
    # Convert to numpy arrays (IMPORTANT FIX)
    p = np.array(p)
    y = np.array(y)
    
    # Avoid log(0)
    eps = 1e-9
    p = np.clip(p, eps, 1 - eps)
    
    # Compute focal loss
    loss = -(1 - p) ** gamma * y * np.log(p) \
           - (p ** gamma) * (1 - y) * np.log(1 - p)
    
    return np.mean(loss)