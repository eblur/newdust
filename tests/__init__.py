import numpy as np
import newdust

def percent_diff(a, b):
    # Return the absolute value of the percent difference between two values
    return np.abs(1.0 - (a/b))
