import numpy as np
from scipy.integrate import trapz

from halo import *
from ..grainpop import *
from .. import constants as c

__all__ = ['UniformGalHalo','ScreenGalHalo','path_diff','uniformISM','screenISM']

ANGLES = np.logspace(0.0, 3.5, np.int(3.5/0.05))

class UniformGalHalo(object):
    """
    | *An htype class for storing halo properties (see halo.py)*
    |
    | **ATTRIBUTES**
    | description = 'Uniform'
    | md : Dust mass column  [g cm^-2]
    """
    def __init__(self, md):
        self.description = 'Uniform'
        self.md = md

class ScreenGalHalo(object):
    """
    | *An htype class for storing halo properties (see halo.py)*
    |
    | **ATTRIBUTES**
    | description = 'Screen'
    | md  : Dust mass column [g cm^-2]
    | x   : float[0-1] : position of dust screen
    | x = 0 is the position of the source
    | x = 1 is the position of the observer
    """
    def __init__(self, md, x):
        self.description = 'Screen'
        self.md  = md
        self.x   = x

#--------------- Galactic Halos --------------------

def _is_small_angle(radians):
    if np.max(np.abs(radians)) > 0.2:
        return False
    else:
        return True


def path_diff(alpha, x):
    """
    | Calculates path difference associated with a particular alpha and x : alpha^2*(1-x)/(2x), units of D (distance to X-ray source)
    | ASSUMES SMALL ANGLES
    |
    | **INPUTS**
    | alpha  : scalar : observation angle [arcsec]
    | x      : scalar or np.array : position of dust patch (source is at x=0, observer at x=1)
    """
    assert (np.max(x) < 1.0) & (np.min(x) > 0)
    if (np.size(alpha) > 1) & (np.size(x) > 1):
        assert len(alpha) == len(x)
    alpha_rad = alpha * c.arcs2rad
    if not _is_small_angle(alpha_rad):
        print("WARNING: astrodust.halos.galhalo functions assume small angle scattering and the largest angle is > 0.01 rad")
    return alpha_rad**2 * (1-x) / (2*x)

def time_delay(alpha, x, dkpc):
    """
    | Returns time delay [seconds] associated with a particular alpha and x, given distance to X-ray source.
    | ASSUMES SMALL ANGLES
    |
    | **INPUTS**
    | alpha : observation angle [arcsec]
    | x     : position of a dust patch (source is at x=0, observer at x=1)
    | D     : distance to the X-ray source [kpc]
    """
    delta_x = path_diff(alpha, x)
    d_cm    = dkpc * 1.e3 * c.pc2cm   # cm
    return delta_x * d_cm / c.clight  # seconds

def uniformISM(halo, gpop, nx=500):
    """
    | Calculate the X-ray scattering intensity for dust distributed
    | uniformly along the line of sight
    |
    | **MODIFIES**
    | halo.htype, halo.taux, halo.norm_int
    |
    | **INPUTS**
    | halo : Halo object
    | gpop : SingleGrainPop object
    """
    assert isinstance(halo, Halo)
    assert isinstance(gpop, SingleGrainPop)

    NE, NA     = np.size(halo.lam), np.size(gpop.a)
    halo.htype = UniformGalHalo(md=gpop.mdens)
    halo.norm_int = np.zeros(shape=(NE, np.size(halo.theta)))

    xgrid      = np.linspace(1.0/nx, 1.0, nx)
    xmesh      = np.repeat(
        np.repeat(xgrid.reshape(1, 1, nx), NE, axis=0),
        NA, axis=1)
    ndmesh     = np.repeat(
        np.repeat(gpop.ndens.reshape(1, NA, 1), NE, axis=0),
        nx, axis=2)
    assert np.shape(xmesh) == (NE, NA, nx)
    assert np.shape(ndmesh) == (NE, NA, nx)
    i_th = 0
    for al in halo.theta:
        thscat = al / xgrid  # nx, goes from small to large angle
        gpop.calculate_ext(halo.lam, unit=halo.lam_unit, theta=thscat)
        dsig   = gpop.diff  # NE x NA x nx, [cm^2 arcsec^-2]
        itemp  = dsig * ndmesh / xmesh**2  # NE x NA x nx, [um^-1 arcsec^-2]

        intx      = trapz(itemp, xgrid, axis=2)  # NE x NA, [um^-1 arcsec^-2]
        intensity = trapz(intx, gpop.a, axis=1)  # NE, [arcsec^-2]
        halo.norm_int[:,i_th] = intensity
        i_th += 1

    halo.taux  = gpop.tau_sca

    #thsca_grid = np.min(halo.theta) / xgrid[::-1]  # scattering angles go from min obs angle -> inf [arcsec]
    #agrid      = np.repeat(gpop.a.reshape(1, NA, 1), NE, axis=0)  # NE x NA
    #gpop.calculate_ext(halo.lam, unit=halo.lam_unit, theta=thsca_grid)
    #dscat      = interp  ## Need to figure out multi-dimensional interpolation


def screenISM(halo, gpop, x=0.5):
    """
    | Calculate the X-ray scattering intensity for dust in an
    | infinitesimally thin wall somewhere on the line of sight.
    |
    | **MODIFIES**
    | halo.htype, halo.taux, halo.norm_int
    |
    | **INPUTS**
    | halo : Halo object
    | xg   : float : distance FROM source / distance between source and observer
    | NH   : float : column density [cm^-2]
    | d2g  : float : dust-to-gass mass ratio
    """
    assert isinstance(halo, Halo)
    assert isinstance(gpop, SingleGrainPop)
    assert x != 0.0

    NE, NA, NTH = np.size(halo.lam), np.size(gpop.a), np.size(halo.theta)
    halo.htype = ScreenGalHalo(md=gpop.mdens, x=x)

    thscat = halo.theta / x
    gpop.calculate_ext(halo.lam, unit=halo.lam_unit, theta=thscat)
    dsig   = gpop.diff  # NE x NA x NTH, [cm^2 arsec^-2]

    ndmesh = np.repeat(
        np.repeat(gpop.ndens.reshape(1, NA, 1), NE, axis=0),
        NTH, axis=2)

    itemp  = np.power(x, -2.0) * dsig * ndmesh  # NE x NA x NTH, [um^-1 arcsec^-2]
    intensity = trapz(itemp, gpop.a, axis=1)  # NE x NTH, [arcsec^-2]

    halo.norm_int = intensity
    halo.taux     = gpop.tau_sca
