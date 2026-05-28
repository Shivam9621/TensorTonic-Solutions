import numpy as np

def stratified_split(X, y, test_size=0.2, rng=None):
    """
    Split features X and labels y into train/test while preserving class proportions.
    """

    X = np.array(X)
    y = np.array(y)

    X_train, X_test = [], []
    y_train, y_test = [], []

    classes, counts = np.unique(y, return_counts=True)

    for cls, count in zip(classes, counts):

        idx = np.where(y == cls)[0]

        if rng is not None:
            rng.shuffle(idx)
        else:
            np.random.shuffle(idx)

        test_count = int(round(count * test_size))

        # keep at least one sample in train if possible
        if count > 1 and test_count >= count:
            test_count = count - 1

        test_idx = np.sort(idx[:test_count])
        train_idx = np.sort(idx[test_count:])

        X_test.extend(X[test_idx])
        y_test.extend(y[test_idx])

        X_train.extend(X[train_idx])
        y_train.extend(y[train_idx])

    return X_train, X_test, y_train, y_test