import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    # Compute hinge loss for each sample
    losses = np.maximum(0, margin - y_true * y_score)
    
    # Apply reduction
    if reduction == "mean":
        return np.mean(losses)
    elif reduction == "sum":
        return np.sum(losses)
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")