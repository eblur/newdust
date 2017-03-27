import numpy as np
from ... import constants as c

__all__ = ['RGscat']

CHARSIG       = 1.04 * 60.0  # characteristic scattering angle [arcsec E(keV)^-1 a(um)^-1]
ALLOWED_UNITS = ['kev', 'angs']

def _lam_cm(lam, unit='kev'):
    assert unit in ALLOWED_UNITS
    if unit == 'kev':
        result  = c.hc / lam  # kev cm / kev
    if unit == 'angs':
        result  = c.angs2cm * lam  # cm/angs * angs
    return result  # cm

class RGscat(object):
    """
    | RAYLEIGH-GANS scattering model.
    | *see* Mauche & Gorenstein (1986), ApJ 302, 371
    | *see* Smith & Dwek (1998), ApJ, 503, 831
    |
    | **ATTRIBUTES**
    | stype : string : 'RGscat'
    |
    | *functions*
    | Qsca( lam, a, cm, unit= )
    |    *returns* scattering efficiency [unitless]
    | Char( lam, a, unit= )
    |    *returns* characteristc scattering angle [arcsec keV um]
    | Diff( theta, lam, a, cm, unit= )
    |    *returns* differential scattering cross-section [cm^2 ster^-1]
    """

    def __init__(self):
        self.stype = 'RGscat'
        self.cite  = 'Calculating RG-Drude approximation\nMauche & Gorenstein (1986), ApJ 302, 371\nSmith & Dwek (1998), ApJ, 503, 831'

    def Qsca(self, lam, a, cm, unit='kev'):
        a_cm   = a * c.micron2cm
        lam_cm = _lam_cm(lam, unit)

        x    = 2.0 * np.pi * a_cm / lam_cm

        mm1  = cm.rp(lam, unit=unit) - 1 + 1j * cm.ip(lam, unit=unit)
        return 2.0 * np.power(x, 2) * np.power(np.abs(mm1), 2)

    def Char(self, lam, a, unit='kev'):   # Standard deviation on scattering angle
        lam_cm = _lam_cm(lam, unit)
        E_kev  = c.hc / lam_cm            # kev cm / cm
        return CHARSIG / (E_kev * a)      # arcsec

    # Can take multiple theta, but should only use one 'a' value
    # Can take multiple E, but should be same size as theta
    def Diff(self, theta, lam, a, cm, unit='kev'):  # cm^2 ster^-1
        a_cm   = a * c.micron2cm      # cm
        lam_cm = _lam_cm(lam, unit)   # cm

        x      = 2.0 * np.pi * a_cm / lam_cm
        mm1    = cm.rp(lam, unit) + 1j * cm.ip(lam, unit) - 1

        thdep  = 2./9. * np.exp(-np.power(theta/self.Char(lam, a, unit), 2) / 2.0)
        dsig   = 2.0 * a_cm**2 * x**4 * np.abs(mm1)**2
        return dsig * thdep
