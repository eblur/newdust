import numpy as np
from scipy.interpolate import interp1d

from newdust import constants as c
from newdust.graindist.composition import _find_cmfile

__all__ = ['CmGraphite']

ALLOWED_UNITS = ['kev', 'angs']
RHO_GRA       = 2.2  # g cm^-3

class CmGraphite(object):
    """
    | **ATTRIBUTES**
    | cmtype : 'Graphite'
    | rho : grain material density (g cm^-3)
    | size   : 'big' or 'small'
    |   'big' gives results for 0.1 um sized graphite grains at 20 K [Draine (2003)]
    |   'small' gives results for 0.01 um sized grains at 20 K
    | orient : 'perp' or 'para'
    |   'perp' gives results for E-field perpendicular to c-axis
    |   'para' gives results for E-field parallel to c-axis
    | citation : A string containing citation to original work
    | interps  : A tuple containing scipy.interp1d objects (rp, ip)
    |
    | *functions*
    | rp(lam, unit='kev') : Returns real part (unit='kev'|'angs')
    | ip(lam, unit='kev') : Returns imaginary part (unit='kev'|'angs')
    | cm(lam, unit='kev') : Complex index of refraction of dtype='complex'
    | plot(lam=None, unit='kev') : Plots Re(m-1) and Im(m)
    |   if lam is *None*, plots the original interp objects
    |   otherwise, plots with user defined wavelength (lam)
    """
    def __init__(self, rho=RHO_GRA, size='big', orient='perp'):
        self.cmtype = 'Graphite'
        self.rho    = rho
        self.size   = size
        self.orient = orient
        self.citation = "Using optical constants for graphite,\nDraine, B. T. 2003, ApJ, 598, 1026\nhttp://adsabs.harvard.edu/abs/2003ApJ...598.1026D"

        D03file = _find_cmfile('CM_D03.pysav')  # look up file
        D03vals = c.restore(D03file)  # read in index values

        if size == 'big':
            if orient == 'perp':
                lamvals = D03vals['Cpe_010_lam']
                revals  = D03vals['Cpe_010_re']
                imvals  = D03vals['Cpe_010_im']

            if orient == 'para':
                lamvals = D03vals['Cpa_010_lam']
                revals  = D03vals['Cpa_010_re']
                imvals  = D03vals['Cpa_010_im']

        if size == 'small':

            if orient == 'perp':
                lamvals = D03vals['Cpe_001_lam']
                revals  = D03vals['Cpe_001_re']
                imvals  = D03vals['Cpe_001_im']

            if orient == 'para':
                lamvals = D03vals['Cpa_001_lam']
                revals  = D03vals['Cpa_001_re']
                imvals  = D03vals['Cpa_001_im']

        lamEvals = c.hc / c.micron2cm / lamvals  # keV
        rp  = interp1d(lamEvals, revals)
        ip  = interp1d(lamEvals, imvals)
        self.interps = (rp, ip)

    def rp(self, lam, unit='kev'):
        assert unit in ALLOWED_UNITS
        if unit == 'kev':
            E = lam
        if unit == 'angs':
            E = c.hc_angs / lam
        return self.interps[0](E)

    def ip(self, lam, unit='kev'):
        assert unit in ALLOWED_UNITS
        if unit == 'kev':
            E = lam
        if unit == 'angs':
            E = c.hc_angs / lam
        return self.interps[1](E)

    def cm(self, lam, unit='kev'):
        return self.rp(lam, unit=unit) + 1j * self.ip(lam, unit=unit)

    def plot(self, ax, lam=None, unit='kev', rppart=True, impart=True):
        if lam is None:
            rp_m1 = np.abs(self.interps[0].y - 1.0)
            ip = self.interps[1].y
            x  = self.interps[0].x
            xlabel = "Energy (keV)"
        else:
            rp_m1 = np.abs(self.rp(lam, unit=unit)-1.0)
            ip = self.ip(lam, unit=unit)
            x  = lam
            assert unit in ALLOWED_UNITS
            if unit == 'kev': xlabel = "Energy (keV)"
            if unit == 'angs': xlabel = "Wavelength (Angstroms)"
        if rppart:
            ax.plot(x, rp_m1, ls='-', label='|Re(m-1)|')
        if impart:
            ax.plot(x, ip, ls='--', label='Im(m)')
        ax.set_xlabel(xlabel)
        ax.legend()
