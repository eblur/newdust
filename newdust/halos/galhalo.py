import numpy as np
from scipy.integrate import trapz
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline

import astropy.units as u
from astropy.io import fits

from .halo import Halo
from ..grainpop import *

__all__ = ['UniformGalHalo','ScreenGalHalo','path_diff','time_delay']

ANGLES = np.logspace(0.0, 3.5, int(3.5/0.05))

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

        # `al` (alpha) is the observed angular distance of the 
        # scattering halo image from the point source center
        i_th = 0
        for al in self.theta:
            thscat = al / xgrid  # nx, goes from small to large angle
            gpop.calculate_ext(self.lam, theta=thscat)
            dsig  = gpop.diff.to('cm^2 arcsec^-2').value # NE x NA x nx, [cm^2 arcec^-2]
            itemp  = dsig * ndmesh / xmesh**2  # NE x NA x nx, [um^-1 arcsec^-2]

            intx      = trapz(itemp, xgrid, axis=2)  # NE x NA, [um^-1 arcsec^-2]
            intensity = trapz(intx, gpop.a.to('micron').value, axis=1)  # NE, [arcsec^-2]
            self.norm_int[:,i_th] = intensity
            i_th += 1
        # attach the units from the above calculation
        self.norm_int *= u.Unit('arcsec^-2')

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
        gpop.calculate_ext(self.lam, theta=thscat)
        dsig   = gpop.diff.to('cm^2 arcsec^-2') # NE x NA x NTH, [cm^2 arcsec^-2]

        ndmesh = np.repeat(
            np.repeat(gpop.ndens.reshape(1, NA, 1), NE, axis=0),
            NTH, axis=2) * u.Unit('cm^-2')
        # dust column density, size distribution per micron (hidden unit)

        itemp  = np.power(x, -2.0) * dsig * ndmesh  # NE x NA x NTH, [um^-1 arcsec^-2]
        intensity = trapz(itemp, gpop.a.to('micron').value, axis=1)  # NE x NTH, [arcsec^-2]
        print(intensity.unit)

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
                inten[i,j] += np.interp(t[j], time, lc * self.norm_int[i,j] * self.fabs[i])

        return inten

    def fake_variable_image(self, time, lc, arf,
                            exposure=10.e3, tnow=None, dist=8.0,
                            pix_scale=0.5, num_pix=[2400,2400],
                            lmin=None, lmax=None, save_file=None):
        """
        Make a fake image of a variable scattering halo
        using a telescope ARF as input.

        Parameters
        ----------
        time : numpy.ndarray [days]
            Time values for light curve

        lc : numpy.ndarray (unitless)
            Light curve in units of source flux
            (i.e. will be multiplied by self.fabs)

        arf : string
            Filename of telescope ARF

        exposure : float [seconds]
            Exosure time for simulated image

        tnow : float [days]
            Time for calculating the halo image
            (Default: Last time value in light curve)

        dist : float [kpc]
            Distance to the object in kpc

        pix_scale : float [arcsec]
            Size of simulated pixels

        num_pix : ints (nx,ny)
            Size of pixel grid to use

        lmin : float
            Minimum halo.lam value
            (Default:None uses entire range)

        lmax : float
            Maximum halo.lam value
            (Default:None uses entire range)

        save_file : string (Default:None)
            Filename to use if you want to save the output to a FITS file

        Returns
        -------
        2D numpy.ndarray of shape (nx, ny), representing the image of
        a dust scattering halo. The halo intensity at different
        energies are converted into counts using the ARF. Then a
        Poisson distribution is used to simulate the number of counts
        in each pixel.

        If the user supplies a file name string using the save_file
        keyword, a FITS file will be saved.
        """
        assert np.all(time >= 0.0)
        if tnow is None:
            time_now = time[-1]
        else:
            time_now = tnow

        var_profile = self.variable_profile(time, lc, tnow=time_now, dist=dist)
        # intensity cube (NE x NTH), phot/cm^2/s/arcsec^2

        # Decide which energy indexes to use
        if lmin is None:
            imin = 0
        else:
            imin = min(np.arange(len(self.lam))[self.lam >= lmin])
        if lmax is None:
            iend = len(self.lam)
        else:
            iend = max(np.arange(len(self.lam))[self.lam <= lmax])

        # set up image grid
        xlen, ylen = num_pix
        xcen, ycen = xlen//2, ylen//2
        ccdx, ccdy = np.meshgrid(np.arange(xlen), np.arange(ylen))
        radius = np.sqrt((ccdx - xcen)**2 + (ccdy - ycen)**2)

        # Typical ARF files have columns 'ENERG_LO', 'ENERG_HI', 'SPECRESP'
        arf_data = fits.open(arf)['SPECRESP'].data
        arf_x = 0.5*(arf_data['ENERG_LO'] + arf_data['ENERG_HI'])
        arf_y = arf_data['SPECRESP']
        arf   = InterpolatedUnivariateSpline(arf_x, arf_y, k=1)

        # Conversion erg -> ct for each energy bin
        if self.lam_unit in ['Angs', 'Angstrom', 'angs', 'angstrom']:
            ener = self.lam * u.angstrom
        elif self.lam_unit in ['kev', 'keV']:
            ener = self.lam * u.keV
        else:
            ener = self.lam * u.Unit(self.lam_unit)
        int_conv = arf(ener.to(u.keV, equivalencies=u.spectral()))
        # cm^2 ct/phot

        r_asec = radius * pix_scale
        result = np.zeros_like(radius)
        for i in np.arange(imin, iend):
            h_interp = InterpolatedUnivariateSpline(
                    self.theta, var_profile[i,:] * int_conv[i], k=1,
                    ext=1) # ct/s/arcsec^2
            # corresponding counts at each radial value in the grid
            pix_flux = h_interp(r_asec) * pix_scale**2 * exposure # cts
            # use poisson statistics to get a random value
            pix_random = np.random.poisson(pix_flux)
            # add it to the final result
            result += pix_random

        if save_file is not None:
            hdu  = fits.PrimaryHDU(result)
            hdul = fits.HDUList([hdu])
            hdul.writeto(save_file, overwrite=True)

        return result

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
