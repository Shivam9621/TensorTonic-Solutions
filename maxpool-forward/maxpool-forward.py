def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    H = len(X)
    W = len(X[0])

    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1

    output = []

    for i in range(H_out):
        row = []
        for j in range(W_out):
            max_val = float('-inf')

            for r in range(pool_size):
                for c in range(pool_size):
                    val = X[i * stride + r][j * stride + c]
                    if val > max_val:
                        max_val = val

            row.append(max_val)

        output.append(row)

    return output