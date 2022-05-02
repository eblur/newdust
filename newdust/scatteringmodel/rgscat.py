import numpy as np
import astropy.units as u
from .. import helpers
from .scatteringmodel import ScatteringModel

__all__ = ['RGscattering']

CHARSIG       = 1.04  # characteristic scattering angle [arcmin E(keV)^-1 a(um)^-1]

class RGscattering(ScatteringModel):
    """
    Rayleigh-Gans scattering model. *See* Mauche & Gorenstein (1986), ApJ 302, 371; 
    Smith & Dwek (1998), ApJ, 503, 831
    """
    def __init__(self, **kwargs):
        ScatteringModel.__init__(self, **kwargs)
        self.stype = 'RGscat'
        self.citation = 'Calculating RG-Drude approximation\nMauche & Gorenstein (1986), ApJ 302, 371\nSmith & Dwek (1998), ApJ, 503, 831'

    def calculate(self, lam, a, cm, theta=0.0):
        """
        Calculate the extinction efficiences with the Rayleigh-Gans approximation.

        lam : astropy.units.Quantity -or- numpy.ndarray
            Wavelength or energy values for calculating the cross-sections;
            if no units specified, defaults to keV
        
        a : astropy.units.Quantity -or- numpy.ndarray
            Grain radius value(s) to use in the calculation;
            if no units specified, defaults to micron
        
        cm : newdust.graindist.composition object
            Holds the optical constants and density for the compound.
        
        theta : astropy.units.Quantity -or- numpy.ndarray -or- float
            Scattering angles for computing the differential scattering cross-section;
            if no units specified, defaults to radian
        """
        # Store the parameters
        lam_cm0, a_cm0, theta_rad0 = self._store_parameters(lam, a, cm, theta)
        NE, NA, NTH = np.size(lam_cm0), np.size(a_cm0), np.size(theta_rad0)

        # Make sure every variable is an array
        lam_cm_1d    = helpers._make_array(lam_cm0)
        a_cm_1d      = helpers._make_array(a_cm0)
        theta_rad_1d = helpers._make_array(theta_rad0)

        # Get the complex index of refraction minus one (m-1)
        cmi_1d    = cm.cm(lam_cm_1d * u.cm) - 1.0

        # Characteristic scattering angle (sigma in Gaussian approximation)
        sigma_rad = self.characteristic_angle(lam, a).to('radian').value

        # Make everything NE x NA
        a_cm   = np.repeat(a_cm_1d.reshape(1, NA), NE, axis=0)
        lam_cm = np.repeat(lam_cm_1d.reshape(NE, 1), NA, axis=1)
        mm1    = np.repeat(cmi_1d.reshape(NE, 1), NA, axis=1)
        # Size parameter (grain circumference to incoming wavelength)
        x      = 2.0 * np.pi * a_cm / lam_cm # (NE x NA)
        
        # Calculate the scattering efficiencies (1-d)
        qsca = _qsca(x, mm1)
        self.qsca = qsca
        self.qext = qsca
        self.qabs = self.qext - self.qsca

        # Calculate the differential scattering cross-section of shape (NE, NA, NTH)
        xs_sca    = _dsig(a_cm, x, mm1) # cm^2
        xs_sca_3d = np.repeat(xs_sca.reshape(NE, NA, 1), NTH, axis=2)

        # Calculate the angular dependence with shape (NE, NA, NTH)
        theta_3d  = np.repeat(
            np.repeat(theta_rad_1d.reshape(1, 1, NTH), NE, axis=0),
            NA, axis=1)
        sigma_3d   = np.repeat(sigma_rad.reshape(NE, NA, 1), NTH, axis=2)
        thdep     = _thdep(theta_3d, sigma_3d) # ster^-1

        # Divide by geometric area cross-section, assumes spherical grains
        geo    = np.pi * a_cm**2  # NE x NA
        geo_3d = np.repeat(geo.reshape(NE, NA, 1), NTH, axis=2)
        
        # Differential cross-section: amplitude * angular portion / geometric cross-section
        self.diff = xs_sca_3d * thdep / geo_3d  # ster^-1

    # Standard deviation on scattering angle distribution
    def characteristic_angle(self, lam, a):
        """
        Calculates the characteristic scattering angle under the Rayleigh-Gans approximation, 
        with the Gaussian approximation to the bessel functions.

        lam : astropy.units.Quantity -or- numpy.ndarray
            Wavelength or energy values for calculating the cross-sections;
            if no units specified, defaults to keV
        
        a : astropy.units.Quantity -or- numpy.ndarray
            Grain radius value(s) to use in the calculation;
            if no units specified, defaults to micron
        
        Returns
        -------
        astropy.units.Quantity using the formula 1.04 arcmin (E/keV)^-1 (a/micron)^-1
        """
        lam_keV = lam
        if isinstance(lam, u.Quantity):
            lam_keV = lam.to('keV', equivalencies=u.spectral()).value

        a_um = a
        if isinstance(a, u.Quantity):
            a_um = a.to('micron', equivalencies=u.spectral()).value
        
        return CHARSIG * u.arcmin / (lam_keV * a_um)

#--------------- Helper functions

def _qsca(x, mm1):  # NE x NA
    return 2.0 * np.power(x, 2) * np.power(np.abs(mm1), 2) # unitless

def _dsig(a_cm, x, mm1):  # NE x NA
    # Amplitude portion of the differential scattering cross-section
    return 2.0 * np.power(a_cm, 2) * np.power(x, 4) * np.power(np.abs(mm1), 2) # cm^2

def _thdep(theta_rad, sigma_rad):  # NE x NA x NTH
    # Angular portion of the differential scattering cross-section
    return 2./9. * np.exp(-0.5 * np.power(theta_rad/sigma_rad, 2))  # ster^-1
