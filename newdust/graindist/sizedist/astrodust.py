import numpy as np
import astropy.units as u
from scipy.integrate import trapezoid as trapz
from newdust.graindist import shape

__all__ = ['Astrodust']

# Some default values
RHO      = 3.0     # g cm^-3 (average grain material density)

NA       = 100     # default number for grain size dist resolution

# min and max grain radii
AMIN     = 4.5e-4   # micron (equivalent to 4.5 angstrom)
AMAX     = 1.    # micron

SHAPE    = shape.Sphere()

#------------------------------------

class Astrodust(object):
    """
    The Astrodust grain size distribution accroding to Hensley & Draine 2022
    """
    def __init__(self, amin=AMIN, amax=AMAX, na=NA, log=False):
        """
        Inputs
        ------
        amin : astropy.units.Quantity -or- float :  minimum grain radius; if a float, micron units assumed

        amax : astropy.units.Quantity -or- float : maximum grain radius; if a float, micron units assumed

        NA  : int : number of a values to use in grid of grain radii

        log : boolean (False): if True, use log-spaced grid of grain radii
        """
        # Set the name of this size disribution
        self.dtype = 'Astrodust'

        # Put amin and amax into units of micron
        if isinstance(amin, u.Quantity):
            amin_um = amin.to('micron').value
        else:
            amin_um = amin
        if isinstance(amax, u.Quantity):
            amax_um = amax.to('micron').value
        else:
            amax_um = amax
        
        # Set up the grid of grain sizes
        if log:
            self.a = np.logspace(np.log10(amin_um), np.log10(amax_um), na) * u.micron
        else:
            self.a = np.linspace(amin_um, amax_um, na) * u.micron
        
        # Set up the constants according to Hensley & Draine 2022
        self.B = 3.31e-10 # H^-1
        self.a0 = 63.8 *u.angstrom 
        self.sigma = 0.353
        self.A0 = 2.97e-5 # H^-1
        self.A1 = -3.40
        self.A2 = -0.807
        self.A3 = 0.157
        self.A4 = 7.96e-3
        self.A5 = -1.68e-3


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
        a0_um = self.a0.to('micron').value

        # astro dust distribution
        adep  = self.B/a_um*np.exp(-(np.log((self.a/self.a0).to('').value)**2)/(2*self.sigma**2))\
                + self.A0/a_um*np.exp(self.A1*((np.log((self.a/u.angstrom).to('').value)))\
                + self.A2*((np.log((self.a/u.angstrom).to('').value))**2)\
                + self.A3*((np.log((self.a/u.angstrom).to('').value))**3)\
                + self.A4*((np.log((self.a/u.angstrom).to('').value))**4)\
                + self.A5*((np.log((self.a/u.angstrom).to('').value))**5))  # um^-1
        
        # get the mass dependence, units of g um^-1
        mgra  = shape.vol(self.a) * rho     # g (mass of each grain)
        dmda  = adep * mgra                 # g um^-1
        
        # Integrate over dmda and use that with total mass to get the 
        # correct constant for the entire function
        const = md / trapz(dmda, a_um)  # cm^-2
        
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
