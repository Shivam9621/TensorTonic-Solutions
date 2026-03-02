def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    # Write code here
    n = len(X)          # number of samples
    d_out = len(W[0])   # number of output neurons
    
    Y = []
    
    for i in range(n):
        row = []
        for j in range(d_out):
            total = 0
            for k in range(len(W)):   # d_in
                total += X[i][k] * W[k][j]
            total += b[j]             # add bias
            row.append(float(total))
        Y.append(row)
    
    return Y