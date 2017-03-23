"""
Power law grain size distribution
"""

import numpy as np
from scipy.integrate import trapz
from newdust.graindist import shape

__all__ = ['Powerlaw']

# Some default values
RHO      = 3.0     # g cm^-3 (average grain material density)

NA       = 100     # default number for grain size dist resolution
PDIST    = 3.5     # default slope for power law distribution

# min and max grain radii for MRN distribution
AMIN     = 0.005   # micron
AMAX     = 0.3     # micron

SHAPE    = shape.Sphere()

#------------------------------------

class Powerlaw(object):
    """
    | **INPUTS**
    | amin : minimum grain size [microns]
    | amax : maximum grain size [microns]
    | p   : scalar for power law dn/da \propto a^-p
    | NA  : int : number of a values to use
    | log : boolean : False (default), True = use log-spaced a values
    |
    | **ATTRIBUTES**
    |   amin, amax, p, a
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
    def __init__(self, amin=AMIN, amax=AMAX, p=PDIST, na=NA, log=False):
        if log:
            self.a = np.logspace(np.log10(amin), np.log10(amax), na)
        else:
            self.a = np.linspace(amin, amax, na)
        self.p    = p

    def ndens(self, md, rho=RHO, shape=SHAPE):
        adep  = np.power(self.a, -self.p)   # um^-p
        mgra  = shape.vol(self.a) * rho     # g (mass of each grain)
        dmda  = adep * mgra
        const = md / trapz(dmda, self.a)  # cm^-2 um^p-1
        return const * adep  # cm^-2 um^-1

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
