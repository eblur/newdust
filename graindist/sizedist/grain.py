"""
Single grain size distribution
"""

import numpy as np
from newdust.graindist import shape

__all__ = ['Grain']

# Some default values
AMICRON = 1.0  # um
RHO     = 3.0     # g cm^-3 (average grain material density)

#-------------------------------

class Grain(object):
    """
    | **ATTRIBUTES**
    | a   : scalar [micron]
    |
    | *functions*
    | ndens(md, rho=3.0, shape=shape.Sphere()) : returns number density of dust grains
    |   md = dust mass column [g cm^-2]
    |   rho = dust grain material density [g cm^-3]
    |   shape = dust grain shape (default spherical)
    """
    def __init__(self, rad=AMICRON):
        assert np.size(rad) == 1
        self.a   = np.array([rad])

    def ndens(self, md, rho=RHO, shape=shape.Sphere()):
        """
        Calculate number density of dust grains
            |
            | **INPUTS**
            | md : dust mass density [e.g. g cm^-2]
            | rho : grain material density [g cm^-3]
        """
        gvol = shape.vol(self.a)
        return md / (gvol * rho)
