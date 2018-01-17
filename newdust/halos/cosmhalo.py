import numpy as np
from scipy.integrate import trapz

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
    | cosm    : Cosmology object
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
    assert zs >= 0.0
    assert zg >= 0.0

    # Store information about this halo calculation
    halo.htype = CosmHalo(zs=zs, zg=zg, cosm=cosm, igmtype='Screen')

    # Light was bluer when it hit the dust scattering screen
    if halo.lam_unit == 'kev':
        lam_g = halo.lam * (1.0 + zg)
    if halo.lam_unit == 'angs':
        lam_g = halo.lam / (1.0 + zg)

    X      = cosm.dchi(zs, zp=zg) / cosm.dchi(zs)  # Single value
    thscat = halo.theta / X                     # Scattering angle required

    gpop.calculate_ext(lam_g, unit=halo.lam_unit, theta=thscat)
    dsig = gpop.diff  # NE x NA x NTH, [cm^2 arsec^-2]

    NE, NTH, NA = np.size(halo.lam), np.size(halo.theta), np.size(gpop.a)
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
    assert zs >= 0.0

    # internal function for dealing with redshift
    def _apply_redshifts(l0, lam_unit, zvals):
        assert lam_unit in ALLOWED_UNITS
        if np.size(l0) > 1:
            assert np.shape(l0) == np.shape(zvals)
        if halo.lam_unit == 'kev':
            lz = l0 * (1.0 + zvals)
        if halo.lam_unit == 'angs':
            lz = l0 / (1.0 + zvals)
        return lz

    # Store information about this halo calculation
    halo.htype = CosmHalo(zs=zs, zg=None, cosm=cosm, igmtype='Uniform')
    cosm_md    = cosm.cosmdens  # g cm^-3
    print("Adjusting grain population to md = %.2e [g cm^-3]" % cosm_md)
    gpop.md    = cosm_md  # Give the grain population the correct amount of dust, note different units than normal

    # Set up for cosmological integrals
    NE, NTH = np.size(halo.lam), np.size(halo.theta)
    zpvals  = np.linspace(0.0, zs-zs/nz, nz)
    c_H0_cm = c.cperh0 * (c.h0 / cosm.h0)  # scalar, [cm]
    hfac    = np.sqrt(cosm.m * np.power(1+zpvals, 3) + cosm.l)  # length nz

    # internal function for calculating total scattering optical depth
    def _taux_integral(lam, lam_unit, zvals):
        NE, NZ = len(lam), len(zvals)
        lam_2d = np.repeat(lam.reshape(NE, 1), NZ, axis=1)     # NE x nz
        zp_2d  = np.repeat(zpvals.reshape(1, NZ), NE, axis=0)  # NE x nz
        lz_2d  = _apply_redshifts(lam_2d, lam_unit, zp_2d)     # NE x nz
        gpop.calculate_ext(lz_2d.flatten(), unit=lam_unit)     # new NE = NE times nz
        dtau = gpop.tau_sca.reshape(NE, nz)  # reshape to NE x nz, units are [cm^-1] due to input

        hfac_2d   = np.repeat(hfac.reshape(1, nz), NE, axis=0)  # NE x nz
        integrand = (1+zp_2d)**2 * dtau * c_H0_cm / hfac_2d  # unitless
        result    = trapz(integrand, zpvals, axis=1)  # integrate over z
        return result  # np.shape(result) = (NE,)

    # Calculate the total scattering optical depth
    lam = c._make_array(halo.lam)
    taux_result = _taux_integral(lam, halo.lam_unit, zpvals)
    halo.taux = taux_result

    # Calculate normalized intensity
    Dtot   = cosm.dchi(zs, nz=nz)
    DP     = np.array([])
    for zp in zpvals:
        DP = np.append(DP, cosm.dchi(zs, zp=zp))
    X      = DP/Dtot

    def _halo_integral(l0, lam_unit, zvals, thscat):
        assert np.size(l0) == 1
        assert np.size(thscat) == np.size(zvals)

        lz = _apply_redshifts(l0, lam_unit, zvals)
        gpop.calculate_ext(lz, unit=lam_unit, theta=thscat)
        dsig = gpop.int_diff  # nz x nz, [cm^-1 arcsec^-2, based on number density units above]

        dsig_diag = dsig.diagonal()  # gets diagonal of the matrix, length nz
        integrand = c_H0_cm/hfac * np.power((1+zvals)/X, 2) * dsig_diag  # [arcsec^-2]
        return trapz(integrand, zvals)

    # Integrate scattering halo for each obsrvation angle
    # This is the long, slow way of calculating with for-loops
    # Could switch to interpolation methods instead
    # But interpolation could also be a problem for accuracy of Mie scattering halos
    halo.norm_int = np.zeros(shape=(NE,NTH))
    ic = 0
    for al in halo.theta:
        thscat = al / X  # length nz
        result = []
        for l0 in lam:
            result.append(_halo_integral(l0, halo.lam_unit, zpvals, thscat))
        assert len(result) == NE
        halo.norm_int[:,ic] = np.array(result)
        ic += 1
