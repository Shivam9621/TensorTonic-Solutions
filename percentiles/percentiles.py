import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    
    x = np.sort(np.array(x, dtype=float))
    q = np.array(q, dtype=float)

    n = len(x)
    result = []

    for p in q:
        if p == 0:
            result.append(x[0])
            continue
            
        if p == 100:
            result.append(x[-1])
            continue

        pos = (p / 100) * (n - 1)

        lower = int(np.floor(pos))
        upper = int(np.ceil(pos))

        if lower == upper:
            result.append(x[lower])
        else:
            weight = pos - lower
            value = x[lower] + weight * (x[upper] - x[lower])
            result.append(value)

    return np.array(result)