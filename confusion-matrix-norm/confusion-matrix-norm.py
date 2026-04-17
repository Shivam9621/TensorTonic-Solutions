import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    """
    Compute confusion matrix with optional normalization.
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Infer number of classes if not given
    if num_classes is None:
        num_classes = max(max(y_true), max(y_pred)) + 1

    # Initialize confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=float)

    # Fill confusion matrix
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    # Apply normalization
    if normalize == 'true':  # row-wise
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        cm = cm / row_sums

    elif normalize == 'pred':  # column-wise
        col_sums = cm.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1
        cm = cm / col_sums

    elif normalize == 'all':  # total normalization
        total = cm.sum()
        if total != 0:
            cm = cm / total

    # 'none' → no change

    return cm