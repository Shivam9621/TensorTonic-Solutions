def linear_interpolation(values):
    result = values[:]
    n = len(result)

    i = 0
    while i < n:
        if result[i] is None:
            left = i - 1
            j = i

            while j < n and result[j] is None:
                j += 1

            right = j

            left_val = result[left]
            right_val = result[right]

            gap = right - left

            for k in range(left + 1, right):
                fraction = (k - left) / gap
                result[k] = left_val + fraction * (right_val - left_val)

            i = right
        else:
            i += 1

    return result