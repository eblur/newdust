import numpy as np
from scipy.integrate import trapz
import astropy.units as u

from .halo import Halo
from ..grainpop import *
from .. import constants as c

__all__ = ['UniformGalHalo','ScreenGalHalo','path_diff','time_delay']

ANGLES = np.logspace(0.0, 3.5, np.int(3.5/0.05))

class UniformGalHalo(Halo):
    def __init__(self, *args, **kwargs):
        Halo.__init__(self, *args, **kwargs)
        self.description = 'Uniform'
        self.md = None

    def calculate(self, gpop, nx=500):
        """
        Calculate the X-ray scattering intensity for dust distributed
        uniformly along the line of sight

        Parameters
        ----------
        gpop : newdust.grainpop.SingleGrainPop

        nx : int
            Number of x-values to use for calculation (Default: 500)

        Returns
        -------
        None. Updates the md, norm_int, and taux attributes.
        """
        assert isinstance(gpop, SingleGrainPop)
        self.md    = gpop.mdens

        NE, NA     = np.size(self.lam), np.size(gpop.a)
        self.norm_int = np.zeros(shape=(NE, np.size(self.theta)))

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
        for al in self.theta:
            thscat = al / xgrid  # nx, goes from small to large angle
            gpop.calculate_ext(self.lam, unit=self.lam_unit, theta=thscat)
            dsig   = gpop.diff  # NE x NA x nx, [cm^2 arcsec^-2]
            itemp  = dsig * ndmesh / xmesh**2  # NE x NA x nx, [um^-1 arcsec^-2]

            intx      = trapz(itemp, xgrid, axis=2)  # NE x NA, [um^-1 arcsec^-2]
            intensity = trapz(intx, gpop.a, axis=1)  # NE, [arcsec^-2]
            self.norm_int[:,i_th] = intensity
            i_th += 1

        self.taux  = gpop.tau_sca

class ScreenGalHalo(Halo):
    def __init__(self, *args, **kwargs):
        Halo.__init__(self, *args, **kwargs)
        self.description = 'Screen'
        self.md   = None
        self.x    = None

    def calculate(self, gpop, x=0.5):
        """
        Calculate the X-ray scattering intensity for dust in an
        infinitesimally thin wall somewhere on the line of sight.

        Parameters
        ----------
        gpop : newdust.grainpop.SingleGrainPop

        x : float (0.0, 1.0]
            1.0 - (distance to screen / distance to X-ray source)

        Returns
        -------
        None. Updates the md, x, norm_int, and taux attributes.
        """
        assert isinstance(gpop, SingleGrainPop)
        assert (x > 0.0) & (x <= 1.0)
        self.md   = gpop.mdens
        self.x    = x

        NE, NA, NTH = np.size(self.lam), np.size(gpop.a), np.size(self.theta)

        thscat = self.theta / x
        gpop.calculate_ext(self.lam, unit=self.lam_unit, theta=thscat)
        dsig   = gpop.diff  # NE x NA x NTH, [cm^2 arsec^-2]

        ndmesh = np.repeat(
            np.repeat(gpop.ndens.reshape(1, NA, 1), NE, axis=0),
            NTH, axis=2)

        itemp  = np.power(x, -2.0) * dsig * ndmesh  # NE x NA x NTH, [um^-1 arcsec^-2]
        intensity = trapz(itemp, gpop.a, axis=1)  # NE x NTH, [arcsec^-2]

        self.norm_int = intensity
        self.taux     = gpop.tau_sca

    #------- Deal with variable scattering halo images ----#
    def variable_profile(self, time, lc, dist=8.0, tnow=None):
        """
        Given a light curve, calculate the energy-dependent intensity of
        the scattering halo at some time afterwards.

        Parameters
        ----------
        time : numpy.ndarray (days)
            Time values for light curve

        lc : numpy.ndarray (unitless)
            Light curve in units of source flux
            (i.e. will be multiplied by self.fabs)

        dist : float (kpc)
            Distance to the object in kpc

        tnow : float (days)
            Time for calculating the halo image
            (Default: Last time value in light curve)

        Returns
        -------
        numpy.ndarray (NE x NTH) [fabs units / arcsec**2]
            Scattering halo intensity as a function of
            energy and observation angle.
        """
        assert self.fabs is not None, "Must run calculate_intensity before" \
                                "calculating a variable profile"

        tzero = time[0]
        # echo observation time
        if tnow is None:
            tnow = time[-1]

        assert tnow > tzero, "Invalid value for tnow"

        # cross section data in arcsec, convert to radian
        theta_rad = self.theta/3600./180.*np.pi

        ne, ntheta = len(self.lam), len(self.theta)
        inten  = np.zeros(shape=(ne, ntheta))
        lctm   = (time-tzero)

        for i in range(len(self.lam)):
            deltat = time_delay(self.theta, self.x, dist) * u.second.to(u.day)
            t      = tnow - deltat
            for j in range(ntheta):
                inten[i,j] += np.interp(t[j], lctm, lc * self.norm_int[i,j] * self.fabs[i])

        return inten

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
