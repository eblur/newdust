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
    if np.size(gpop.a) == 1:
        intensity = np.sum(itemp, gpop.a, axis=1)
    else:
        intensity = trapz(itemp, gpop.a, axis=1)
    halo.norm_int = intensity  # NE x NTH, [arcsec^-2]

    halo.taux = gpop.tau_sca

def uniformIGM(halo, gpop, zs, cosm=Cosmology(), nz=500):
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
    | cosm : cosmology.Cosmology
    | nz   : int : number of z-values to use in integration
    """
    # Store information about this halo calculation
    halo.htype = CosmHalo(zs=zs, zg=None, cosm=cosm, igmtype='Uniform')
    cosm_md    = cosm.cosmdens(zs)  # g cm^-3
    print("Adjusting grain population to md = %.2e [g cm^-3]" % cosm_md)
    gpop.md    = cosm_md  # Give the grain population the correct amount of dust, note different units than normal

    Dtot   = cosmo.dchi(zs, cosm=cosm, nz=nz)
    zpvals = np.linspace(0.0, zs, nz)

    c_H0_cm = c.cperh0 * (c.h0 / cosm.h0)  # cm
    hfac    = np.sqrt(cosm.m * np.power(1+zpvals, 3) + cosm.l)

    lam = c._make_array(halo.lam)
    taux_result = []
    for ll in lam:
        taux_result.append(_taux_integral(ll))
    halo.taux = np.array(taux_result)

    '''# Start calculating normalized intensity
    DP = np.array([])
    for zp in zpvals:
        DP = np.append(DP, cosmo.dchi(zs, zp=zp, cosm=cosm))
    X      = DP/Dtot

    intensity = np.array([])
    for al in halo.theta:
        thscat = al / X  # np.size(thscat) = nz
        gpop.calculate_ext(lam_g, unit=halo.lam_unit, theta=thscat)
        dsig   = gpop.int_diff  # NE x NTH, [cm^2 arsec^-2]

        dsig_xgrid = np.repeat(dsig.reshape(NE, NTH, 1), nz, axis=2)

        itemp     = c_H0_cm/hfac * np.power((1+zpvals)/X, 2) * dsig_xgrid
        intensity = np.append(intensity, trapz(itemp, zpvals, axis=2))'''

    def _taux_integral(lam):
        assert lam_unit in ALLOWED_UNITS
        if halo.lam_unit == 'kev':
            lz = lam * (1.0 + zpvals)
        if halo.lam_unit == 'angs':
            lz = lam / (1.0 + zpvals)

        gpop.calculate_ext(lz, unit=lam_unit)
        dsig = gpop.tau_sca  # length = NE, units are [cm^-1] due to input

        integrand = (1+zpvals)**2 * dsig * c_H0_cm / hfac  # unitless
        return trapz(integrand, zpvals)
