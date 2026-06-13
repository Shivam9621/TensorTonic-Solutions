import numpy as np

def knn_distance(X_train, X_test, k):
    # Convert to numpy arrays
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    # Handle 1D inputs
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # Pairwise Euclidean distances using broadcasting
    distances = np.sqrt(
        np.sum((X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]) ** 2, axis=2)
    )

    # Sort training indices by distance
    sorted_idx = np.argsort(distances, axis=1)

    if k <= n_train:
        return sorted_idx[:, :k].astype(int)

    # Pad with -1 if k > n_train
    result = np.full((n_test, k), -1, dtype=int)
    result[:, :n_train] = sorted_idx
    return result