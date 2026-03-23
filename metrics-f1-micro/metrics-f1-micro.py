def f1_micro(y_true, y_pred) -> float:
    from collections import Counter

    tp = 0
    fp = 0
    fn = 0

    labels = set(y_true) | set(y_pred)

    for label in labels:
        for t, p in zip(y_true, y_pred):
            if p == label and t == label:
                tp += 1
            elif p == label and t != label:
                fp += 1
            elif p != label and t == label:
                fn += 1

    if tp == 0:
        return 0.0

    return (2 * tp) / (2 * tp + fp + fn)