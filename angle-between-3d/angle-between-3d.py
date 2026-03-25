import numpy as np

def angle_between_3d(v, w):
    v = np.array(v, dtype=float)
    w = np.array(w, dtype=float)

    norm_v = np.linalg.norm(v)
    norm_w = np.linalg.norm(w)

    # Fix here 👇
    if norm_v == 0 or norm_w == 0:
        return np.nan

    dot_product = np.dot(v, w)
    cos_theta = dot_product / (norm_v * norm_w)

    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    return np.arccos(cos_theta)