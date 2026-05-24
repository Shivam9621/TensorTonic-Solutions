def min_max_scaling(data):
    """
    Scale each column of the data matrix to the [0, 1] range.
    """

    rows = len(data)
    cols = len(data[0])

    mn = [float('inf')] * cols
    mx = [float('-inf')] * cols

    for i in range(rows):
        for j in range(cols):
            mn[j] = min(mn[j], data[i][j])
            mx[j] = max(mx[j], data[i][j])

    ans = []

    for i in range(rows):
        row = []

        for j in range(cols):
            if mx[j] == mn[j]:
                row.append(0.0)
            else:
                row.append((data[i][j] - mn[j]) / (mx[j] - mn[j]))

        ans.append(row)

    return ans