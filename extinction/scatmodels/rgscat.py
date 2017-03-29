import numpy as np
from ... import constants as c

__all__ = ['RGscat']

CHARSIG       = 1.04 * 60.0  # characteristic scattering angle [arcsec E(keV)^-1 a(um)^-1]

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
        self.pars  = None  # parameters used in running the calculation: lam, a, cm, theta, unit
        self.qsca  = None
        self.qext  = None
        self.diff  = None

    @property
    def qabs(self):
        return self.qext - self.qsca

    def calculate(self, lam, a, cm, unit='kev', theta=0.0):
        self.pars = dict(zip(['lam','a','cm','theta','lam_unit'],[lam, a, cm, theta, unit]))

        a_cm   = a * c.micron2cm
        lam_cm = c._lam_cm(lam, unit)

        x    = 2.0 * np.pi * a_cm / lam_cm

        mm1  = cm.cm(lam, unit) - 1.0
        qsca = 2.0 * np.power(x, 2) * np.power(np.abs(mm1), 2)

        self.qsca = qsca
        self.qext = qsca

        thdep  = 2./9. * np.exp(-np.power(theta/self.Char(lam, a, unit), 2) / 2.0)
        dsig   = 2.0 * a_cm**2 * x**4 * np.abs(mm1)**2
        self.diff = dsig * thdep  # cm^2 / ster

    def Char(self, lam, a, unit='kev'):   # Standard deviation on scattering angle
        E_kev  = c._lam_kev(lam, unit)
        return CHARSIG / (E_kev * a)      # arcsec
