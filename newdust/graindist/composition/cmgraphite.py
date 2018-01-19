import numpy as np
from scipy.interpolate import interp1d
from astropy.io import ascii
from astropy import units as u

from newdust import constants as c
from newdust.graindist.composition import _find_cmfile

__all__ = ['CmGraphite']

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

        if size == 'big':
            D03file_para = _find_cmfile('callindex.out_CpaD03_0.10')
            D03file_perp = _find_cmfile('callindex.out_CpeD03_0.10')
        if size == 'small':
            D03file_para = _find_cmfile('callindex.out_CpaD03_0.01')
            D03file_perp = _find_cmfile('callindex.out_CpeD03_0.01')

        D03dat_para = ascii.read(D03file_para, header_start=4, data_start=5)
        D03dat_perp = ascii.read(D03file_perp, header_start=4, data_start=5)

        if orient == 'perp':
            wavel = D03dat_perp['wave(um)'] * u.micron
            lamvals = wavel.to(u.cm).value
            revals  = 1.0 + D03dat_perp['Re(n)-1']
            imvals  = D03dat_perp['Im(n)']

        if orient == 'para':
            wavel = D03dat_para['wave(um)'] * u.micron
            lamvals = wavel.to(u.cm).value
            revals  = 1.0 + D03dat_para['Re(n)-1']
            imvals  = D03dat_para['Im(n)']

        rp  = interp1d(lamvals * c.micron2cm, revals)  # wavelength (cm), rp
        ip  = interp1d(lamvals * c.micron2cm, imvals)  # wavelength (cm), ip
        self.interps = (rp, ip)

    def _interp_helper(self, lam_cm, interp, rp=False):
        # Returns zero for wavelengths not covered by the interpolation object
        # If the real part is needed, returns 1 (consistent with vacuum)
        result = np.zeros(np.size(lam_cm))
        if rp: result += 1

        if np.size(lam_cm) == 1:
            if (lam_cm >= np.min(interp.x)) & (lam_cm <= np.max(interp.x)):
                result = interp(lam_cm)
        else:
            ii = (lam_cm >= np.min(interp.x)) & (lam_cm <= np.max(interp.x))
            result[ii] = interp(lam_cm[ii])
        return result

    def rp(self, lam, unit='kev'):
        lam_cm = c._lam_cm(lam, unit)
        return self._interp_helper(lam_cm, self.interps[0], rp=True)

    def ip(self, lam, unit='kev'):
        lam_cm = c._lam_cm(lam, unit)
        return self._interp_helper(lam_cm, self.interps[1])

    def cm(self, lam, unit='kev'):
        return self.rp(lam, unit=unit) + 1j * self.ip(lam, unit=unit)

    def plot(self, ax, lam=None, unit='kev', rppart=True, impart=True):
        if lam is None:
            rp_m1 = np.abs(self.interps[0].y - 1.0)
            ip = self.interps[1].y
            x  = self.interps[0].x / c.micron2cm  # cm / (cm/um)
            xlabel = "Wavelength (um)"
        else:
            rp_m1 = np.abs(self.rp(lam, unit)-1.0)
            ip = self.ip(lam, unit)
            x  = lam
            assert unit in c.ALLOWED_LAM_UNITS
            if unit == 'kev': xlabel = "Energy (keV)"
            if unit == 'angs': xlabel = "Wavelength (Angstroms)"
        if rppart:
            ax.plot(x, rp_m1, ls='-', label='|Re(m-1)|')
        if impart:
            ax.plot(x, ip, ls='--', label='Im(m)')
        ax.set_xlabel(xlabel)
        ax.legend()
