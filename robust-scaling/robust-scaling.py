def robust_scaling(values):
    if not values:
        return []

    sorted_vals = sorted(values)
    n = len(sorted_vals)

    # Median
    if n % 2 == 0:
        median = (sorted_vals[n//2 - 1] + sorted_vals[n//2]) / 2
        lower_half = sorted_vals[:n//2]
        upper_half = sorted_vals[n//2:]
    else:
        median = sorted_vals[n//2]
        lower_half = sorted_vals[:n//2]
        upper_half = sorted_vals[n//2 + 1:]

    # Helper median function
    def find_median(arr):
        if not arr:   # 🔥 important fix
            return 0
        m = len(arr)
        if m % 2 == 0:
            return (arr[m//2 - 1] + arr[m//2]) / 2
        return arr[m//2]

    Q1 = find_median(lower_half)
    Q3 = find_median(upper_half)

    IQR = Q3 - Q1

    # Scaling
    if IQR == 0:
        return [x - median for x in values]

    return [(x - median) / IQR for x in values]