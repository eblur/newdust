import numpy as np
from astropy.io import ascii
from astropy import units as u

from newdust.graindist.composition import _find_cmfile, Composition

__all__ = ['CmGraphite']

RHO_GRA       = 2.2  # g cm^-3

class CmGraphite(Composition):
    """
    Optical constants for Graphite from Draine (2003)

    | Additional Attributes
    | ---------------------
    | size   : 'big' or 'small'; 
    |   'big' gives results for 0.1 um sized graphite grains at 20 K [Draine (2003)];
    |   'small' gives results for 0.01 um sized grains at 20 K
    | orient : 'perp' or 'para'
    |   'perp' gives results for E-field perpendicular to c-axis
    |   'para' gives results for E-field parallel to c-axis
    """
    def __init__(self, rho=RHO_GRA, size='big', orient='perp'):
        Composition.__init__(self)
        self.cmtype = 'Graphite'
        self.rho    = rho
        self.citation = "Using optical constants for graphite,\nDraine, B. T. 2003, ApJ, 598, 1026\nhttp://adsabs.harvard.edu/abs/2003ApJ...598.1026D"
        # Additional info not in default class
        self.size   = size
        self.orient = orient
        
        # Populate the wavelength and optical constants info
        if size == 'big':
            D03file_para = _find_cmfile('callindex.out_CpaD03_0.10')
            D03file_perp = _find_cmfile('callindex.out_CpeD03_0.10')
        if size == 'small':
            D03file_para = _find_cmfile('callindex.out_CpaD03_0.01')
            D03file_perp = _find_cmfile('callindex.out_CpeD03_0.01')

        D03dat_para = ascii.read(D03file_para, header_start=4, data_start=5)
        D03dat_perp = ascii.read(D03file_perp, header_start=4, data_start=5)

        # The wavelength grid needs to be in ascending order 
        # for np.interp to run correctly
        if orient == 'perp':
            wavel = D03dat_perp['wave(um)'] * u.micron
            wsort = np.argsort(wavel.value)
            self.wavel = wavel[wsort]
            self.revals  = 1.0 + D03dat_perp['Re(n)-1'][wsort]
            self.imvals  = D03dat_perp['Im(n)'][wsort]

        if orient == 'para':
            wavel = D03dat_para['wave(um)'] * u.micron
            wsort = np.argsort(wavel)
            self.wavel = wavel[wsort]
            self.revals  = 1.0 + D03dat_para['Re(n)-1'][wsort]
            self.imvals  = D03dat_para['Im(n)'][wsort]
