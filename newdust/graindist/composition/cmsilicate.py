import numpy as np
from astropy.io import ascii
from astropy import units as u

from newdust.graindist.composition import _find_cmfile, Composition

__all__ = ['CmSilicate']

RHO_SIL       = 3.8  # g cm^-3

class CmSilicate(Composition):
    """
    Optical constants for Silicate from Draine (2003)
    """
    def __init__(self, rho=RHO_SIL):
        Composition.__init__(self)
        self.cmtype = 'Silicate'
        self.rho    = rho
        self.citation = "Using optical constants for astrosilicate,\nDraine, B. T. 2003, ApJ, 598, 1026\nhttp://adsabs.harvard.edu/abs/2003ApJ...598.1026D"

        D03file = _find_cmfile('callindex.out_sil.D03')
        D03dat  = ascii.read(D03file, header_start=4, data_start=5)

        self.wavel = D03dat['wave(um)'] * u.micron
        self.revals = 1.0 + D03dat['Re(n)-1']
        self.imvals = D03dat['Im(n)']
