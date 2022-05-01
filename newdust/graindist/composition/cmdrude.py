import numpy as np
import astropy.units as u
import astropy.constants as c

from newdust.graindist.composition import Composition

__all__ = ['CmDrude']

RHO_DRUDE  = 3.0  # g cm^-3

## -- Some constants to use for the calculation
# Classical electron radius
RE_CM = 2.8179403227e-15 * u.m.to('cm')
# Mass of proton
MP_G = c.m_p.to('g').value 

class CmDrude(Composition):
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
        Composition.__init__(self)
        self.cmtype = 'Drude'
        self.rho    = rho
        self.citation = "Using the Drude approximation.\nBohren, C. F. & Huffman, D. R., 1983, Absorption and Scattering of Light by Small Particles (New York: Wiley)"

        # Set up default values so that inherited plotting method from Composition will work
        self.wavel = np.linspace(1.0, 10.0, 50) * u.keV
        self.revals = self.rp(self.wavel)
        self.imvals = self.ip(self.wavel)

    def rp(self, x):
        if isinstance(x, u.Quantity):
            lam_cm = x.to('cm', equivalencies=u.spectral()).value
        else:
            lam_cm = (x * u.keV).to('cm', equivalencies=u.spectral()).value

        mm1 = self.rho / (2.0*MP_G) * RE_CM/(2.0*np.pi) * np.power(lam_cm, 2)
        return mm1 + 1.0

    def ip(self, x):
        if np.size(x) > 1:
            return np.zeros(np.size(x))
        else:
            return 0.0
