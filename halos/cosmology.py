import numpy as np

from halo import *
from ..grainpop import *
from .. import constants as c

NZ = 500

__all__ = ['Cosmology']

#-----------------------------------------------------

class Cosmology(object):
    """
    | Cosmology object stores relevant cosmological parameters
    |
    | **ATTRIBUTES**
    | h0 : [km/s/Mpc] : Hubble's constant
    | m  : cosmic mass density in units of the critical density
    | l  : lambda mass density in units of the critical density
    | d  : cosmic dust density in units of the critical density
    """
    def __init__(self, h0=c.h0, m=c.omega_m, l=c.omega_l, d=c.omega_d):
        self.h0 = h0
        self.m  = m
        self.l  = l
        self.d  = d

    @property
    def cosmdens(self):
        """
        Returns the co-moving number density of dust grains
        for a given **Cosmology** object
        """
        return cosm.d * c.rho_crit * np.power(cosm.h0/c.h0, 2)  # g cm^-3

    def dchi(self, z, zp=0.0, nz=NZ):
        """
        | Calculates co-moving radial distance [Gpc] from zp to z using dx = cdt/a
        | **INPUTS**
        | z    : float : redshift
        | zp   ; float (0) : starting redshift
        | nz   : int (100) : number of z-values to use in calculation
        """
        zvals     = np.linspace(zp, z, nz)
        integrand = c.cperh0 * (x.h0/self.h0) / np.sqrt(self.m * (1.0+zvals)**3 + self.l)
        return c.intz(zvals, integrand) / (1e9 * c.pc2cm)  # Gpc, in comoving coordinates

    def da(self, theta, z, nz=NZ):
        """
        | Calculates the diameter distance [Gpc] for an object of angular size
        | theta and redshift z using DA = theta(radians) * dchi / (1+z)
        |
        | **INPUTS**
        | theta : float : angular size [arcsec]
        | z     : float : redshift of object
        | nz    : int (100) : number of z-values to use in dchi calculation
        """
        D_gpc = self.dchi(z, nz=nz)
        return theta * c.arcs2rad * D_gpc / (1.0+z)

    def md(self, z):
        D_cm = self.dchi(z) * (1.e9 * c.pc2cm)  # distance to object, in cm
        return self.cosmdens * D_cm  # g cm^-2

'''    # This needs to be rewritten to avoid energy for loop
    def cosm_taux(self, z, gpop, lam=np.array([1.0]), unit='kev', nz=NZ):
        """
        | Calculates the optical depth from dust distributed uniformly in the IGM
        |
        | **INPUTS**
        | z : redshift of source
        | gpop  : grainpop.SingleGrainPop
        | E     : scalar or np.array [keV]
        | nz    : number of z values to use in integral
        |
        | **RETURNS**
        | optical depth to X-ray scattering
        | = kappa * cosmdens * (1+z)^2 c dz / hfac
        """
        ## NEEDS TO BE UPDATED  -- May 8, 2017
        zvals = np.linspace(0.0, z, nz)
        assert unit in ALLOWED_UNITS
        if np.size(lam) > 1:
            if unit == 'kev':
                lam_z = lam
        lam   = np.linspace(halo.lam[0], halo.lam[-1], nz)

        for ener in E:
            Evals     = ener * (1.0 + zvals)
            gpop.calculate_ext(Evals, unit='kev')
            kappa     = ss.KappaScat(E=Evals, scatm=scatm, dist=dist).kappa
            hfac      = np.sqrt( cosm.m * np.power(1+zvals, 3) + cosm.l)
            integrand = kappa * md * np.power(1+zvals, 2) * \
                c.cperh0 * (c.h0/cosm.h0) / hfac
            result    = np.append(result, c.intz( zvals, integrand ))
        return result'''


#-----------------------------------------------------



def cosm_taux_screen( zg, E=1.0, dist=distlib.MRN_dist(md=cosmdens(Cosmology())), scatm=ss.ScatModel()):
    """
    | Calculates the optical depth from a screen of dust in the IGM
    |
    | **INPUTS**
    | zg : redshift of screen
    | E  : scalar or np.array [keV]
    | dist  : distlib.Powerlaw or distlib.Grain
    | scatm : ss.ScatModel
    |
    | **RETURNS**
    | tauX : np.array [optical depth to X-ray scattering] for the screen
    |      : = kappa(Eg) * M_d
    """
    Eg  = E * (1+zg)
    kappa = ss.KappaScat(E=Eg, scatm=scatm, dist=dist).kappa
    return dist.md * kappa
