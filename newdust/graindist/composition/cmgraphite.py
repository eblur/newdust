import numpy as np
from scipy.interpolate import interp1d
from astropy.io import ascii
from astropy import units as u

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
    | wavel : wavelengths from the optical constants file
    | revals : real part of the complex index of refraction
    | imvals : imaginary part of the complex index of refraction
    |
    | *functions*
    | rp(lam) : Returns real part
    | ip(lam) : Returns imaginary part
    | cm(lam) : Complex index of refraction of dtype='complex'
    | plot(lam=None) : Plots Re(m-1) and Im(m)
    |   if lam is *None*, plots the original interp objects
    |   otherwise, plots with user defined wavelength (lam)
    """
    def __init__(self, rho=RHO_GRA, size='big', orient='perp'):
        self.cmtype = 'Graphite'
        self.rho    = rho
        self.size   = size
        self.orient = orient
        self.citation = "Using optical constants for graphite,\nDraine, B. T. 2003, ApJ, 598, 1026\nhttp://adsabs.harvard.edu/abs/2003ApJ...598.1026D"
        self.wavel = None
        self.revals = None
        self.imvals = None

        if size == 'big':
            D03file_para = _find_cmfile('callindex.out_CpaD03_0.10')
            D03file_perp = _find_cmfile('callindex.out_CpeD03_0.10')
        if size == 'small':
            D03file_para = _find_cmfile('callindex.out_CpaD03_0.01')
            D03file_perp = _find_cmfile('callindex.out_CpeD03_0.01')

        D03dat_para = ascii.read(D03file_para, header_start=4, data_start=5)
        D03dat_perp = ascii.read(D03file_perp, header_start=4, data_start=5)

        if orient == 'perp':
            self.wavel = D03dat_perp['wave(um)'] * u.micron
            self.revals  = 1.0 + D03dat_perp['Re(n)-1']
            self.imvals  = D03dat_perp['Im(n)']

        if orient == 'para':
            self.wavel = D03dat_para['wave(um)'] * u.micron
            self.revals  = 1.0 + D03dat_para['Re(n)-1']
            self.imvals  = D03dat_para['Im(n)']

    def rp(self, lam):
        # If the input is an astropy quantity, convert it to the same unit as wavel
        if isinstance(lam, u.Quantity):
            new_x = lam.to(self.wavel.unit, equivalencies=u.spectral()).value
        # Otherwise, assume the unit is keV
        else:
            new_x = (lam * u.keV).to(self.wavel.unit, equivalencies=u.spectral()).value
        return np.interp(new_x, self.wavel.value, self.revals, left=1.0, right=1.0)

    def ip(self, lam):
        # If the input is an astropy quantity, convert it to the same unit as wavel
        if isinstance(lam, u.Quantity):
            new_x = lam.to(self.wavel.unit, equivalencies=u.spectral()).value
        # Otherwise, assume the unit is keV
        else:
            new_x = (lam * u.keV).to(self.wavel.unit, equivalencies=u.spectral()).value
        return np.interp(new_x, self.wavel.value, self.imvals, left=0.0, right=0.0)

    def cm(self, lam):
        return self.rp(lam) + 1j * self.ip(lam)

    def plot(self, ax, lam, rppart=True, impart=True, label=''):
        # If no grid specified, plot the default one
        if lam is None:
            rp_m1 = np.abs(self.revals - 1.0)
            ip = self.imvals
            x  = self.wavel.value
            xlabel = self.wavel.unit
        # Else, plot the interpolated values
        else:
            rp_m1 = np.abs(self.rp(lam)-1.0)
            ip = self.ip(lam)
            # Check if the input value had units
            if isinstance(lam, u.Quantity):
                x = lam.value
                xlabel = lam.unit
            # If not, assume keV units
            else:
                x = lam
                xlabel = 'keV'
        # If the user wants to plot Real Part
        if rppart:
            ax.plot(x, rp_m1, ls='-', label='{} |Re(m-1)|'.format(label))
        # If the user wants to plot Imaginary Part
        if impart:
            ax.plot(x, ip, ls='--', label='{} Im(m)'.format(label))
        ax.set_xlabel(xlabel)
        ax.legend()
