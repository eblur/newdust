import numpy as np
import newdust

def percent_diff(a, b):
    # Return the absolute value of the percent difference between two values
    assert np.size(a) == np.size(b)
    result = np.abs(1.0 - (a/b))
    if np.size(a) > 1:
        if any(a == 0.0) or any(b == 0.0):
            return a - b
    elif a == 0.0 or b == 0:
        return a - b
    return result
