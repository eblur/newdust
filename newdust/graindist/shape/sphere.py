import numpy as np
import astropy.units as u

__all__ = ['Sphere']

class Sphere(object):
    """
    Attributes
    ----------
    shape : string : Describes the shape ('sphere')
    """
    def __init__(self):
        self.shape = 'Sphere'

    def vol(self, a):
        """
        Return the grain's volume in units of cm^3

        Inputs
        ------
        a : astropy.units.Quantity -or- float : if a float, unit of microns assumed

        Returns
        -------
        (4/3) * pi * a^3
        """
        if isinstance(a, u.Quantity):
            a_cm = a.to('cm').value
        else:
            a_cm = (a * u.micron).to('cm').value
        return (4.0/3.0) * np.pi * np.power(a_cm, 3)  # cm^3

    def cgeo(self, a):
        """
        Return the geometric cross-section of a spherical particle, in units of cm^2

        Inputs
        ------
        a : astropy.units.Quantity -or- float : if a float, unit of microns assumed

        Returns
        -------
        pi * a^2
        """
        if isinstance(a, u.Quantity):
            a_cm = a.to('cm').value
        else:
            a_cm = (a * u.micron).to('cm').value
        return np.pi * np.power(a_cm, 2)  # cm^2
