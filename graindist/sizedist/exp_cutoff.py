"""
Power law grain size distribution with an exponential cut-off at the large end
"""

import numpy as np
from scipy.integrate import trapz
from newdust import constants as c

__all__ = ['ExpCutoff']

# Some default values
RHO      = 3.0     # g cm^-3 (average grain material density)

NA       = 100     # default number for grain size dist resolution
PDIST    = 3.5     # default slope for power law distribution

# min and max grain radii for MRN distribution
AMIN     = 0.005   # micron
ACUT     = 0.3     # micron
NFOLD    = 5       # Number of e-foldings (a/amax) to cover past the amax point

#------------------------------------

class ExpCutoff(object):
    """
    | amin : minimum grain size [microns]
    | acut : maximum grain size [microns], after which exponential function will cause a turn over in grain size
    | p   : scalar for power law dn/da \propto a^-p
    | NA  : int : number of a values to use
    | log : boolean : False (default), True = use log-spaced a values
    | nfold : number of e-foldings to go beyond acut
    |
    | **ATTRIBUTES**
    |   acut, p, a
    |
    | **FUNCTIONS**
    | ndens(md, rho=3.0) : returns number density (dn/da) [cm^-2 um^-1]
    |   md = dust mass column [g cm^-2]
    |   rho = dust grain material density [g cm^-3]
    |
    | plot(ax, md, rho=3.0, *kwargs*) : plots (dn/da) a^4 [cm^-2 um^3]
    """
    def __init__(self, amin=AMIN, acut=ACUT, p=PDIST, na=NA, log=False, nfold=NFOLD):
        self.acut = acut
        if log:
            self.a = np.logspace(amin, acut * nfold, na)
        else:
            self.a = np.linspace(amin, acut * nfold, na)
        self.p    = p

    def ndens(self, md, rho=RHO):
        adep  = np.power(self.a, -self.p) * np.exp(-self.a/self.acut)   # um^-p
        gdens = (4. / 3.) * np.pi * rho
        dmda  = adep * gdens * np.power(self.a * c.micron2cm, 3)  # g um^-p
        const = md / trapz(dmda, self.a)  # cm^-? um^p-1
        return const * adep  # cm^-? um^-1

    def plot(self, ax, md, rho=RHO, **kwargs):
        ax.plot(self.a, self.ndens(md, rho) * np.power(self.a, 4), **kwargs)
        ax.set_xlabel("Radius (um)")
        ax.set_ylabel("$(dn/da) a^4$ (cm$^{-2}$ um$^{3}$)")
        ax.set_xscale('log')
        ax.set_yscale('log')
