import numpy as np
from newdust import constants as c

__all__ = ['Sphere']

class Sphere(object):
    """
    | **ATTRIBUTES**
    | shape : A string describing the shape ('sphere')
    |
    | *functions*
    | vol(a)  : A function that returns the grain's volume (4/3 pi a^3)
    | cgeo(a) : A function that returns the grain's geometric cross-section (pi a^2)
    |
    | Takes grain radius (a) in units of microns only
    """
    def __init__(self):
        self.shape = 'sphere'

    def vol(self, a):
        return (4.0/3.0) * np.pi * np.power(a * c.micron2cm, 3)  # cm^3

    def cgeo(self, a):
        return np.pi * np.power(a * c.micron2cm, 2)  # cm^2
