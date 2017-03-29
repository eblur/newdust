import numpy as np
from newdust import constants as c

__all__ = ['CmDrude']

RHO_DRUDE  = 3.0  # g cm^-3
LAM_MAX    = c.hc / 0.01 # maximal wavelength that we will allow for RG-Drude

class CmDrude(object):
    """
    | **ATTRIBUTES**
    | cmtype : 'Drude'
    | rho    : grain density [g cm^-3]
    | citation : A string containing citation to original work
    |
    | *functions*
    | rp(lam, unit='kev') : Returns real part (unit='kev'|'angs')
    | ip(lam, unit='kev') : Returns imaginary part (always 0.0)
    | cm(lam, unit='kev') : Complex index of refraction of dtype='complex'
    | plot(lam, unit='kev') : Plots Re(m-1)
    """
    def __init__(self, rho=RHO_DRUDE):  # Returns a CM using the Drude approximation
        self.cmtype = 'Drude'
        self.rho    = rho
        self.citation = "Using the Drude approximation.\nBohren, C. F. & Huffman, D. R., 1983, Absorption and Scattering of Light by Small Particles (New York: Wiley)"

    def rp(self, lam, unit='kev'):
        assert unit in c.ALLOWED_LAM_UNITS
        lam_cm = c._lam_cm(lam, unit)

        # Returns 1 if the wavelength supplied is too low energy (i.e. inappropriate for applying Drude)
        mm1 = np.zeros(np.size(lam_cm))
        if (np.size(lam_cm) == 1):
            if lam_cm >= LAM_MAX:
                pass
            else:
                mm1 = self.rho / (2.0*c.m_p) * c.r_e/(2.0*np.pi) * np.power(lam_cm, 2)
        else:
            ii = (lam_cm <= LAM_MAX)
            mm1[ii] = self.rho / (2.0*c.m_p) * c.r_e/(2.0*np.pi) * np.power(lam_cm[ii], 2)
        return mm1 + 1.0

    def ip(self, lam, unit='kev'):
        if np.size(lam) > 1:
            return np.zeros(np.size(lam))
        else:
            return 0.0

    def cm(self, lam, unit='kev'):
        return self.rp(lam, unit=unit) + 0j

    def plot(self, ax, lam, unit='kev', **kwargs):
        assert unit in c.ALLOWED_LAM_UNITS
        rp = self.rp(lam, unit=unit)
        ax.plot(lam, rp-1.0, **kwargs)
        ax.set_ylabel("m-1")
        if unit == 'kev':
            ax.set_xlabel("Energy (keV)")
        if unit == 'angs':
            ax.set_xlabel("Wavelength (Angstroms)")
