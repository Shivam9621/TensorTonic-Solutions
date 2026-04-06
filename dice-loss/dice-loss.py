import numpy as np

def dice_loss(p, y, eps=1e-8):
    p = np.array(p)
    y = np.array(y)
    
    intersection = np.sum(p * y)
    dice = (2 * intersection + eps) / (np.sum(p) + np.sum(y) + eps)
    
    return 1 - dice