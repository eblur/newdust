"""
Single grain size distribution
"""

import numpy as np
from ... import constants as c

__all__ = ['Grain']

# Some default values
AMICRON = 1.0  # um

#-------------------------------

class Grain(object):
    """
    | **ATTRIBUTES**
    | a   : scalar [micron]
    |
    | **FUNCTIONS**
    | ndens ( md : mass density [g cm^-2 or g cm^-3] )
    |    *returns* scalar number density [cm^-3]
    """
    def __init__(self, rad=AMICRON):
        assert np.size(rad) == 1
        self.a   = np.array([rad])

    def ndens(self, md, rho):
        """
        Calculate number density of dust grains
            |
            | **INPUTS**
            | md : dust mass density [e.g. g cm^-2]
            | rho : grain material density [g cm^-3]
        """
        gvol = 4. / 3. * np.pi * np.power(self.a*c.micron2cm, 3)
        return md / (gvol * rho)
