import numpy as np
import astropy.units as u

__all__ = ['Composition']

class Composition(object):
    """
    Composition class for storing information about grain material
    
    Attributes
    ----------
    cmtype : string : label for the compound

    rho : float : density of the material (g cm^-3)

    citation : string : citation for the optical constants loaded

    wavel : astropy.units.Quantity : wavelength grid for optical constants

    revals : numpy.ndarray : real part of the complex index of refraction
    
    imvals : numpy.ndarray : imaginary part of the complex index of refraction
    """
    def __init__(self):
        self.cmtype = None
        self.rho = None
        self.citation = None
        self.wavel = None
        self.revals = 1.0
        self.imvals = 0.0
    
    def rp(self, x):
        """
        Interpolate over the input wavelength grid to get the real part of the 
        complex index of refraction.

        Inputs
        ------
        x : if astropy.units.Quantity, convert to same units as self.wavel;
            if numpy.ndarray, assume keV units
        
        Returns
        -------
        np.interp(x, self.wavel, self.revals, left=1.0, right=1.0)
        """
        # If the input is an astropy quantity, convert it to the same unit as wavel
        if isinstance(x, u.Quantity):
            new_x = x.to(self.wavel.unit, equivalencies=u.spectral()).value
        # Otherwise, assume the unit is keV
        else:
            new_x = (x * u.keV).to(self.wavel.unit, equivalencies=u.spectral()).value
        return np.interp(new_x, self.wavel.value, self.revals, left=1.0, right=1.0)
    
    def ip(self, x):
        """
        Interpolate over the input wavelength grid to get the imaginary part of the 
        complex index of refraction.

        Inputs
        ------
        x : if astropy.units.Quantity, convert to same units as self.wavel;
            if numpy.ndarray, assume keV units
        
        Returns
        -------
        np.interp(x, self.wavel, self.imvals, left=1.0, right=1.0)
        """
        # If the input is an astropy quantity, convert it to the same unit as wavel
        if isinstance(x, u.Quantity):
            new_x = x.to(self.wavel.unit, equivalencies=u.spectral()).value
        # Otherwise, assume the unit is keV
        else:
            new_x = (x * u.keV).to(self.wavel.unit, equivalencies=u.spectral()).value
        return np.interp(new_x, self.wavel.value, self.imvals, left=0.0, right=0.0)
    
    def cm(self, x):
        """
        Returns the complex index of refraction using Python complex numbers

        Inputs
        ------
        x : if astropy.units.Quantity, convert to same units as self.wavel;
            if numpy.ndarray, assume keV units
        """
        return self.rp(x) + 1j * self.ip(x)
    
    def plot(self, ax, lam=None, rppart=True, impart=True, label=''):
        """
        Plots the different parts of the complex index of refraction on a matplotlib axis

        Inputs
        ------
        ax : matplotlib.pyplot.axes object : the axes on which to plot
        lam : if None, will plot the default values; if not None, will interpolate onto the new grid;
            if not an astropy.units.Quantity, the units are assumed to be keV
        rppart : bool (True) : whether or not to plot the real part
        impart : bool (True) : whether or not to plot the imaginary part
        label : string : to add to the labels
        """
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