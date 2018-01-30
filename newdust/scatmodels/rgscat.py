import numpy as np
from .. import constants as c
from .scatmodel import _ScatModel

__all__ = ['RGscat']

CHARSIG       = 1.04 * 60.0  # characteristic scattering angle [arcsec E(keV)^-1 a(um)^-1]

class RGscat(_ScatModel):
    """
    | RAYLEIGH-GANS scattering model.
    | *see* Mauche & Gorenstein (1986), ApJ 302, 371
    | *see* Smith & Dwek (1998), ApJ, 503, 831
    |
    | **ATTRIBUTES**
    | stype : string : 'RGscat'
    | citation : string : citation string
    | pars  : dict   : parameters used to run the calculation
    | qsca  : array  : scattering efficiency (unitless, per geometric area)
    | qext  : array  : extinction efficiency (unitless, per geometric area)
    | diff  : array  : differential scattering cross-section (cm^2 ster^-1)
    |
    | *properties*
    | qabs  : array  : absorption efficiency (unitless, per geometric area)
    |
    | *functions*
    | char( lam, a, unit='kev' )
    |    *returns* characteristc scattering angle [arcsec]
    | calculate( lam, a, cm, unit='kev', theta=0.0 )
    |    calculates the relevant values (qsca, qext, diff)
    """

    def __init__(self):
        self.stype = 'RGscat'
        self.citation = 'Calculating RG-Drude approximation\nMauche & Gorenstein (1986), ApJ 302, 371\nSmith & Dwek (1998), ApJ, 503, 831'
        self.pars  = None  # parameters used in running the calculation: lam, a, cm, theta, unit
        self.qsca  = None
        self.qext  = None
        self.diff  = None

    @property
    def qabs(self):
        return self.qext - self.qsca

    def calculate(self, lam, a, cm, unit='kev', theta=0.0):
        self.pars = dict(zip(['lam','a','cm','theta','lam_unit'],[lam, a, cm, theta, unit]))

        NE, NA, NTH = np.size(lam), np.size(a), np.size(theta)

        # Deal with the 1d stuff first
        # Make sure every variable is an array
        lam   = c._make_array(lam)
        a     = c._make_array(a)
        theta = c._make_array(theta)

        # Convert to the appropriate units
        a_cm_1d   = a * c.micron2cm
        lam_cm_1d = c._lam_cm(lam, unit)
        cmi_1d    = cm.cm(lam, unit) - 1.0

        # Make everything NE x NA
        a_cm   = np.repeat(a_cm_1d.reshape(1, NA), NE, axis=0)
        lam_cm = np.repeat(lam_cm_1d.reshape(NE, 1), NA, axis=1)
        mm1    = np.repeat(cmi_1d.reshape(NE, 1), NA, axis=1)
        char   = self.char(lam_cm, a_cm, unit='cm')
        x      = 2.0 * np.pi * a_cm / lam_cm

        # Calculate the scattering efficiencies (1-d)
        qsca = _qsca(x, mm1)
        self.qsca = qsca
        self.qext = qsca

        # Make the NE x NA x NTH stuff
        dsig        = _dsig(a_cm, x, mm1)
        dsig_3d     = np.repeat(dsig.reshape(NE, NA, 1), NTH, axis=2)

        theta_3d  = np.repeat(
            np.repeat(theta.reshape(1, 1, NTH), NE, axis=0),
            NA, axis=1)
        char_3d   = np.repeat(char.reshape(NE, NA, 1), NTH, axis=2)
        thdep     = _thdep(theta_3d, char_3d)

        self.diff = dsig_3d * thdep  # cm^2 / ster

    # Standard deviation on scattering angle distribution
    def char(self, lam, a, unit='kev'):
        # for cases where I have everything in units of cm
        if unit == 'cm':
            E_kev = c.hc / lam
            a_um  = a * 1.e4
        # otherwise, do the usual
        else:
            E_kev  = c._lam_kev(lam, unit)
            a_um = a
        return CHARSIG / (E_kev * a_um)      # arcsec

#--------------- Helper functions

def _qsca(x, mm1):  # NE x NA
    return 2.0 * np.power(x, 2) * np.power(np.abs(mm1), 2)

def _dsig(a_cm, x, mm1):  # NE x NA
    return 2.0 * np.power(a_cm, 2) * np.power(x, 4) * np.power(np.abs(mm1), 2)

def _thdep(theta, char):  # NE x NA x NTH
    return 2./9. * np.exp(-0.5 * np.power(theta/char, 2))  # NE x NA x NTH
