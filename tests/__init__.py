import numpy as np
import newdust

def percent_diff(a, b):
    # Return the absolute value of the percent difference between two values
    if any(a == 0.0) or any(b == 0.0):
        return a - b
    else:
        return np.abs(1.0 - (a/b))
