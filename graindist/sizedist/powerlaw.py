"""
Power law grain size distribution
"""

import numpy as np
from scipy.integrate import trapz
from ... import constants as c

__all__ = ['Powerlaw']

# Some default values
RHO      = 3.0     # g cm^-3 (average grain material density)

NA       = 100     # default number for grain size dist resolution
PDIST    = 3.5     # default slope for power law distribution

# min and max grain radii for MRN distribution
AMIN     = 0.005   # micron
AMAX     = 0.3     # micron

#------------------------------------

class Powerlaw(object):
    """
    | **ATTRIBUTES**
    | amin : minimum grain size [microns]
    | amax : maximum grain size [microns]
    | p   : scalar for power law dn/da \propto a^-p
    | NA  : int : number of a values to use
    | log : boolean : False (default), True = use log-spaced a values
    """
    def __init__(self, amin=AMIN, amax=AMAX, p=PDIST, na=NA, log=False):
        self.amin = amin
        self.amax = amax
        if log:
            self.a = np.logspace(amin, amax, na)
        else:
            self.a = np.linspace(amin, amax, na)
        self.p    = p

    def ndens(self, md, rho=RHO):
        """
        Calculate number density of dust grains as a function of grain size
            | **RETURNS** numpy.ndarray of dn/da values [number density per micron]
            |
            | **INPUTS**
            | md : dust mass density [e.g. g cm^-2]
            | rho : grain material density [g cm^-3]
        """
        adep  = np.power(self.a, -self.p)   # um^-p
        gdens = (4. / 3.) * np.pi * rho
        dmda  = adep * gdens * np.power(self.a * c.micron2cm, 3)  # g um^-p
        const = md / trapz(dmda, self.a)  # cm^-? um^p-1
        return const * adep  # cm^-? um^-1
