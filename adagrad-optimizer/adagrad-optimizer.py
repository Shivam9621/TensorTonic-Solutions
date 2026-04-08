import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    """
    # Convert to numpy arrays (IMPORTANT FIX)
    w = np.array(w)
    g = np.array(g)
    G = np.array(G)

    # Step 1: accumulate squared gradients
    G = G + g**2
    
    # Step 2: update parameters
    w = w - (lr / (np.sqrt(G + eps))) * g
    
    return w.tolist(), G.tolist()