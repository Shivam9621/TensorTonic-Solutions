import numpy as np

def r2_score(y_true, y_pred) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    # Constant target edge case
    if ss_tot == 0:
        return 1.0 if np.array_equal(y_true, y_pred) else 0.0

    return 1.0 - (ss_res / ss_tot)