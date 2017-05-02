import numpy as np
from scipy.interpolate import interp1d

from halo import *
from cosmology import *
from ..grainpop import *
from .. import constants as c

ALLOWED_UNITS = ['kev','angs']

class CosmHalo(object):
    """
    | *An htype class for storing halo properties*
    |
    | **ATTRIBUTES**
    | zs      : float : redshift of X-ray source
    | zg      : float : redshift of an IGM screen
    | cosm    : cosmo.Cosmology object
    | igmtype : labels the type of IGM scattering calculation : 'Uniform' or 'Screen'
    """
    def __init__(self, zs, zg, cosm, igmtype):
        self.zs      = zs
        self.zg      = zg
        self.cosm    = cosm
        self.igmtype = igmtype

#----------

def screenIGM(halo, gpop, zs, zg, cosm=Cosmology()):
    """
    | Calculates the intensity of a scattering halo from intergalactic
    | dust that is situated in an infinitesimally thin screen somewhere
    | along the line of sight.
    |
    | **MODIFIES**
    | halo.htype, halo.norm_int, halo.taux
    |
    | **INPUTS**
    | halo : Halo object
    | gpop : SingleGrainPop object
    | zs   : float : redshift of source
    | zg   : float : redshift of scr\een
    | cosm : cosmology.Cosmology
    """
    if zg >= zs:
        print("%% STOP: zg must be < zs")
        return

    # Store information about this halo calculation
    halo.htype = CosmHalo(zs=zs, zg=zg, cosm=cosm, igmtype='Screen')

    # Light was bluer when it hit the dust scattering screen
    if halo.lam_unit == 'kev':
        lam_g = halo.lam * (1.0 + zg)
    if halo.lam_unit == 'angs':
        lam_g = halo.lam / (1.0 + zg)

    X      = cosmo.dchi(zs, zp=zg, cosm=cosm) / cosmo.dchi(zs, cosm=cosm)  # Single value
    thscat = halo.theta / X                     # Scattering angle required

    gpop.calculate_ext(lam_g, unit=halo.lam_unit, theta=thscat)
    dsig = gpop.diff  # NE x NA x NTH, [cm^2 arsec^-2]

    ndmesh = np.repeat(
        np.repeat(gpop.ndens.reshape(1, NA, 1), NE, axis=0),
        NTH, axis=2)

    itemp     = np.power(X, -2.0) * dsig * ndmesh  # NE x NA x NTH, [um^-1 arcsec^-2]
    intensity = trapz(itemp, gpop.a, axis=1)  # NE x NTH, [arcsec^-2]
    halo.norm_int = intensity

    halo.taux = gpop.tau_sca
