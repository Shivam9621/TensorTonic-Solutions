def average_pooling_2d(X, pool_size):
    """
    Apply 2D average pooling with non-overlapping windows.
    """
    rows = len(X)
    cols = len(X[0])

    out_rows = rows // pool_size
    out_cols = cols // pool_size

    result = []

    for i in range(out_rows):
        row = []
        for j in range(out_cols):
            total = 0

            for r in range(i * pool_size, (i + 1) * pool_size):
                for c in range(j * pool_size, (j + 1) * pool_size):
                    total += X[r][c]

            row.append(total / (pool_size * pool_size))

        result.append(row)

    return result