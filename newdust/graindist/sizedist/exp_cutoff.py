import numpy as np
import astropy.units as u
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
    Power law grain size distribution with an exponential cut-off at the large end
    """
    def __init__(self, amin=AMIN, acut=ACUT, p=PDIST, na=NA, log=False, nfold=NFOLD):
        """
        Inputs
        ------
        
        amin : astropy.units.Quantity -or- float :  minimum grain radius; if a float, micron units assumed
        
        acut : astropy.units.Quantity -or- float : maximum grain radius, 
        after which exponential function will cause a turn over in grain size;
        if a float, micron units assumed
        
        p   : float : slope for power law dn/da \propto a^-p
        
        NA  : int : number of a values to use
        
        log : boolean : False (default), True = use log-spaced a values
        
        nfold : number of e-foldings to go beyond `acut`
        """
        self.dtype = 'ExpCutoff'

        # Put amin and acut into units of micron
        if isinstance(amin, u.Quantity):
            amin_um = amin.to('micron').value
        else:
            amin_um = amin
        if isinstance(acut, u.Quantity):
            acut_um = acut.to('micron').value
        else:
            acut_um = acut

        # Set up the grid of grain sizes
        if log:
            self.a = np.logspace(np.log10(amin_um), np.log10(acut_um * nfold), na) * u.micron
        else:
            self.a = np.linspace(amin_um, acut_um * nfold, na) * u.micron

        # Log the relevant params
        self.p    = p
        self.acut = acut_um * u.micron

    def ndens(self, md, rho=RHO, shape=SHAPE):
        """
        Calculate number density of dust grains, given a dust mass column

        Inputs
        ------
        
        md : float : mass column density [g cm^-2]

        rho : float : grain material density [g cm^-3]

        shape : newdust.graindist.shape object (default is a Sphere)

        Returns
        -------
        
        Column density of grains in [cm^-2]
        """
        a_um = self.a.to('micron').value
        acut_um = self.acut.to('micron').value

        # power law slope component
        adep  = np.power(a_um, -self.p) * np.exp(-a_um/acut_um)   # um^-p

        # get the mass dependence, units of g um^-p
        mgra  = shape.vol(self.a) * rho  # g (mass of each grain)
        dmda  = adep * mgra              # g um^-p

        # integrate to get the correct scaling constant
        const = md / trapz(dmda, a_um)  # cm^-2 um^p-1

        # Final units are number column density per grain size unit (default:micron)
        return const * adep  # cm^-2 um^-1

    def mdens(self, md, rho=RHO, shape=SHAPE):
        """
        Calculate mass density function for the dust grains, given a total dust mass column

        Inputs
        ------
        
        md : float : mass column density [g cm^-2]

        rho : float : grain material density [g cm^-3]

        shape : newdust.graindist.shape object (default is a Sphere)

        Returns
        -------
        
        Mass column distribution of grains in [cg m^-2 um^-1]
        """
        nd = self.ndens(md, rho, shape)  # dn/da [cm^-2 um^-1]
        mg = shape.vol(self.a) * rho     # grain mass for each radius [g]
        return nd * mg  # g cm^-2 um^-1
