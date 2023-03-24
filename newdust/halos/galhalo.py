import numpy as np
from scipy.integrate import trapz
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline

from scipy.special import erf
from scipy.special import gammaincc
from scipy.special import gamma
from scipy.special import expi

import astropy.units as u
import astropy.constants as c

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
        #print(intensity.unit)

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
    
#--------------- Analytic Halo Calculation --------------

class UniformGalHaloCP15(Halo):
    def __init__(self, *args, **kwargs):
        Halo.__init__(self, *args, **kwargs)
        self.description = 'Uniform CP15'
        self.md = None
    
    def calculate(self, md, amin=0.005*u.micron, amax=0.5*u.micron, p=3.5, rho = 3*u.Unit('g cm^-3'):
        """
        Calculate the X-ray scattering intensity for dust distributed
        uniformly along the line of sight utlizing the analytic solution
        according to the Corrales & Paerels 2015. 

        Parameters
        ----------
        amin: float or astropy.units.Quantity [micron] - minimum grain radius

        amax: float or astropy.units.Quantity [micron] - maximum grain radius

        p: float [unitless] - slope of power law distribution

        rho: float or astropy.units.Quantity [g cm^-3] - grain mass density

        md: float or astropy.units.Quantity [g cm^-2] -  dust mass column

        Returns
        -------
        None. Updates norm_int [arcsec^-2], and taux [unitless] attributes.
        """
        lam_keV = self.lam.to(u.keV, equivalencies=u.spectral())
        theta_arcsec = self.theta.to(u.arcsec)

        if isinstance(md, u.Quantity):
            md_u = md
        else:
            md_u = md * u.g * u.cm**(-2)
        md_u = md_u.to(u.g * u.cm**(-2))
        if isinstance(amin, u.Quantity):
            amin_um = amin
        else:
            amin_um = amin * u.micron
        amin_um = amin_um.to(u.micron)
        if isinstance(amax, u.Quantity):
            amax_um = amax
        else:
            amax_um = amax * u.micron
        amax_um = amax_um.to(u.micron)
        if isinstance(rho, u.Quantity):
            rho_u = rho
        else:
            rho_u = rho *u.g * u.cm**(-3)
        rho_u = rho_u.to(u.g * u.cm**(-3))

        NE, NTH = len(lam_keV), len(theta_arcsec)
        #here will be a function that calculation taux based on Energy input
        #Input energy would be an array with length NE, and the output taux would also have dimension NE
        self.taux = calculate_taux(lam_keV, amin_um, amax_um, p, rho_u, md_u)
        hfrac = np.tile(self.taux.reshape(NE,1), NTH ) # NE x NTH
        energy = np.tile(lam_keV.reshape(NE,1), NTH ) # NE x NTH
        theta = np.tile(theta_arcsec, (NE,1) ) #NE x NTH

        # single grain?

        charsig = 1.04 * 60.0 * u.arcsec/ (energy/u.keV) * u.micron
        const = hfrac / ( theta * charsig * np.sqrt(8.0*np.pi) )
        result = const * G_u(energy, theta, amin_um, amax_um, p) / G_p(amin_um, amax_um, p)

        self.norm_int = result

class ScreenGalHaloCP15(Halo):
    def __init__(self, *args, **kwargs):
        Halo.__init__(self, *args, **kwargs)
        self.description = 'Screen CP15'
        self.md = None
        self.x = None

    def calculate(self, md, amin=0.005*u.micron, amax=0.5*u.micron, p=3.5, rho = 3*u.Unit('g cm^-3'), x=0.5):
        """
        Calculate the X-ray scattering intensity for dust in an
        infinitesimally thin wall somewhere on the line of sight.

        Parameters
        ----------
        amin: float or astropy.units.Quantity [micron] - minimum grain radius

        amax: float or astropy.units.Quantity [micron] - maximum grain radius

        p: float [unitless] - slope of power law distribution

        rho: float or astropy.units.Quantity [g cm^-3] - grain mass density

        md: float or astropy.units.Quantity [g cm^-2] -  dust mass column

        x : float (0.0, 1.0] [unitless] - (distance to screen / distance to X-ray source)

        Returns
        -------
        None. Updates the md, x, norm_int, and taux attributes.
        """

        lam_keV = self.lam.to(u.keV)
        theta_arcsec = self.theta.to(u.arcsec)

        if isinstance(md, u.Quantity):
            md_u = md
        else:
            md_u = md * u.g * u.cm**(-2)
        md_u = md_u.to(u.g * u.cm**(-2))
        if isinstance(amin, u.Quantity):
            amin_um = amin
        else:
            amin_um = amin * u.micron
        amin_um = amin_um.to(u.micron)
        if isinstance(amax, u.Quantity):
            amax_um = amax
        else:
            amax_um = amax * u.micron
        amax_um = amax_um.to(u.micron)
        if isinstance(rho, u.Quantity):
            rho_u = rho
        else:
            rho_u = rho *u.g * u.cm**(-3)
        rho_u = rho_u.to(u.g * u.cm**(-3))

        NE, NTH = len(lam_keV), len(theta_arcsec)
        #here will be a function that calculation taux based on Energy input
        #Input energy would be an array with length NE, and the output taux would also have dimension NE
        self.taux = calculate_taux(lam_keV, amin_um, amax_um, p, rho_u, md_u)
        hfrac = np.tile(self.taux.reshape(NE,1), NTH ) # NE x NTH
        energy = np.tile(lam_keV.reshape(NE,1), NTH ) # NE x NTH
        theta = np.tile(theta_arcsec, (NE,1) ) #NE x NTH
        self.x = x

        charsig0 = 1.04 * 60.0 * u.arcsec/ (energy/u.keV) * u.micron
        const = hfrac / ( 2.0*np.pi*charsig0**2 )
        result = const / x**2 * G_s(energy, theta, amin_um, amax_um, p, x) / G_p(amin_um, amax_um, p)

        self.norm_int = result


# -------------- Useful Function -------------------

def gammainc_fun( a, z ):
    if np.any(z < 0):
        print('ERROR: z must be >= 0')
        return
    if a == 0:
        return -expi(-z)
    elif a < 0:
        return ( gammainc_fun(a+1,z) - np.power(z,a) * np.exp(-z) ) / a
    else:
        return gammaincc(a,z) * gamma(a)
    
def calculate_taux(lam, amin, amax, p, rho, md):
    """
        Calculate the integrated X-ray scattering opacity taux

        Parameters
        ----------
        lam: astropy.units.Quantity [KeV] - Energy

        amin: astropy.units.Quantity [micron] - minimum grain radius

        amax: astropy.units.Quantity [micron] - maximum grain radius

        p: float [unitless] - slope of power law distribution

        rho: astropy.units.Quantity [g cm^-3] - grain mass density

        md: astropy.units.Quantity [g cm^-2] -  dust mass column

        Returns
        -------
        taux: numpy.ndarray or astropy.units.Quantity (NE) - integrated X-ray scattering opacity
        """

    rho_um = rho.to(u.g*u.micron**(-3)) # g micron^-3
    constA = (6.27e-7 *u.cm**2) * (rho/(3*u.g*u.cm**(-3)))/(1*u.micron)**4 # cm^2 micron^-4
    E_keV = lam/u.keV # unitless
    
    # Calculate Normalization Constant C in Power Law Distribution
    if p == 4:
        constC = md/((4/3)*np.pi*rho_um*np.log(amax/amin))
    else:
        constC = (4-p)*md/((4/3)*np.pi*rho_um*(np.power(amax, 4-p) - np.power(amin, 4-p)))

    # Calculate and Return taux
    if p == 5:
        taux = constA*constC*np.log(amax/amin)*np.power(E_keV, -2)
    else:
        taux = constA*constC*(np.power(amax, 5-p) - np.power(amin, 5-p))*np.power(E_keV, -2)/(5-p)
    return taux
   
def G_p(amin, amax, p):
    '''
    Returns integral_a0^a1 a^(4-p) da
    '''
    if p == 5:
        return np.log( amax/amin )
    else:
        return 1.0/(5.0-p) * ( np.power(amax,5.0-p) - np.power(amin,5.0-p) )
        
def G_u(lam, theta, amin, amax, p):
    
    # input lam and theta must be NE x NTH
    power = 6.0 - p
    pfrac = (7.0-p) / 2.0
    charsig = 1.04 * 60.0 * u.arcsec / (lam / u.keV) * u.micron # arcsec micron
    const   = theta / charsig / np.sqrt(2.0) # micron^-1

    A1 = np.power(amax,power) * ( 1 - erf(const*amax) )
    A0 = np.power(amin,power) * ( 1 - erf(const*amin) )
    B1 = np.power(const,-power) * gammainc_fun( pfrac, (const**2 * amax**2).value ) / np.sqrt(np.pi)
    B0 = np.power(const,-power) * gammainc_fun( pfrac, (const**2 * amin**2).value ) / np.sqrt(np.pi)
    return ( (A1-B1) - (A0-B0) ) / power

def G_s(lam, theta, amin, amax, p, x):
    '''
    Function used for evaluating halo from power law distribution of grain sizes (Screen case)
    '''

    charsig0 = 1.04 * 60.0 * u.arcsec / (lam / u.keV) * u.micron # arcsec micron
    pfrac    = (7.0-p)/2.0
    const    = theta**2/(2.0*charsig0**2*x**2) # micron^-2
    gamma1   = gammainc_fun( pfrac, (const * amax**2).value )
    gamma0   = gammainc_fun( pfrac, (const * amin**2).value )
    return -0.5 * np.power( const, -pfrac ) * ( gamma1 - gamma0 )


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
    alpha_rad = alpha * u.arcsec.to('rad')
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
    d_cm    = dkpc * 1.e3 * u.pc.to('cm')  # cm
    return delta_x * (d_cm / c.c).to('s').value # seconds
