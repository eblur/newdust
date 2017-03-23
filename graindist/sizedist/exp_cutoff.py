"""
Power law grain size distribution with an exponential cut-off at the large end
"""

import numpy as np
from scipy.integrate import trapz
from newdust.graindist import shape

__all__ = ['ExpCutoff']

# Some default values
RHO      = 3.0     # g cm^-3 (average grain material density)

NA       = 100     # default number for grain size dist resolution
PDIST    = 3.5     # default slope for power law distribution

# min and max grain radii for MRN distribution
AMIN     = 0.005   # micron
ACUT     = 0.3     # micron
NFOLD    = 5       # Number of e-foldings (a/amax) to cover past the amax point

SHAPE    = shape.Sphere()

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
    | *functions*
    | ndens(md, rho=3.0, shape=shape.Sphere()) : returns number density (dn/da) [cm^-2 um^-1]
    | mdens(md, rho=3.0, shape=shape.Sphere()) : returns mass density (dm/da) [g cm^-2 um^-1]
    |   md    = total dust mass column [g cm^-2]
    |   rho   = dust grain material density [g cm^-3]
    |   shape = dust grain shape (default spherical)
    |
    | plot(ax, md, rho=3.0, *kwargs*) : plots (dn/da) a^4 [cm^-2 um^3]
    """
    def __init__(self, amin=AMIN, acut=ACUT, p=PDIST, na=NA, log=False, nfold=NFOLD):
        self.acut = acut
        if log:
            self.a = np.logspace(np.log10(amin), np.log10(acut * nfold), na)
        else:
            self.a = np.linspace(amin, acut * nfold, na)
        self.p    = p

    def ndens(self, md, rho=RHO, shape=SHAPE):
        adep  = np.power(self.a, -self.p) * np.exp(-self.a/self.acut)   # um^-p
        mgra  = shape.vol(self.a) * rho  # g (mass of each grain)
        dmda  = adep * mgra
        const = md / trapz(dmda, self.a)  # cm^-? um^p-1
        return const * adep  # cm^-? um^-1

    def mdens(self, md, rho=RHO, shape=SHAPE):
        nd = self.ndens(md, rho, shape)  # dn/da [cm^-2 um^-1]
        mg = shape.vol(self.a) * rho     # grain mass for each radius [g]
        return nd * mg  # g cm^-2 um^-1

    def plot(self, ax, md, rho=RHO, shape=SHAPE, **kwargs):
        ax.plot(self.a, self.ndens(md, rho, shape) * np.power(self.a, 4), **kwargs)
        ax.set_xlabel("Radius (um)")
        ax.set_ylabel("$(dn/da) a^4$ (cm$^{-2}$ um$^{3}$)")
        ax.set_xscale('log')
        ax.set_yscale('log')
