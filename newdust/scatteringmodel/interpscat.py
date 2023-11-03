import numpy as np
import astropy.units as u
from .. import helpers
from .scatteringmodel import ScatteringModel
from scipy.interpolate import griddata, RegularGridInterpolator, CloughTocher2DInterpolator

__all__ = ['InterpolateScattering']

## created November 3, 2023
## Load pre-calculated scattering models and interpolate over range of values of interest

class InterpolateScattering(ScatteringModel):
    """
    This object acts a bit differently from a standard ScatteringModel
    loaded with the `from_file` keyword.

    That's because the results of ScatteringModel.calculate are stored
    as attributes, and we don't want to overwrite the tables we are
    using to interpolate!

    So when this Class is called, it stores the loaded table values as
    `qsca_data`, etc.

    Attributes
    ----------
    In addition to those inherited from ScatteringModel

    filename : string : name of the file loaded

    data_pars : dict : stores the run parameters for the original table

    qext_data : np.array : original extinciton efficiency values

    qabs_data : np.array : original absorption efficiency values

    qsca_data : np.array : original scattering efficiency values
    
    diff_data : np.array : original differential scattering efficiencly values (ster^-1)
    """
    def __init__(self, filename):
        ScatteringModel.__init__(self)
        self.filename = filename
        ff = fits.open(infile)
        # Load parameteric information
        lam   = ff[1].data['lam'] * u.Unit(ff[1].header['TUNIT1'])
        a     = ff[2].data['a'] * u.Unit(ff[2].header['TUNIT1'])
        theta = ff[3].data['theta'] * u.Unit(ff[3].header['TUNIT1'])
        self.data_pars = {'lam':lam, 'a':a, 'theta':theta}

        # Load extinction information
        qvals = dict()
        for i in range(4,8):  # runs on hdus 4,5,6,7
            htype = ff[i].header['TYPE']
            qvals[htype] = ff[i].data
        self.qext_data = RegularGridInterpolator((lam.value, a.value), qvals['Qext'],
                                                 method='linear', bounds_error=False, fill_value=None)
        self.qabs_data = RegularGridInterpolator((lam.value, a.value), qvals['Qabs'],
                                                 method='linear', bounds_error=False, fill_value=None)
        self.qsca_data = RegularGridInterpolator((lam.value, a.value), qvals['Qsca'],
                                                 method='linear', bounds_error=False, fill_value=None)
        self.diff_data = RegularGridInterpolator((lam.value, a.value, theta.value), qvals['Diff-xsect (ster^-1)'],
                                                 method='linear', bounds_error=False, fill_value=None)

    def calculate(self, lam, a, theta=0.0):
        """
        Interpolate the loaded tables across the values of interest

        lam : astropy.units.Quantity -or- numpy.ndarray
            Wavelength or energy values for calculating the cross-sections;
            if no units specified, defaults to keV
        
        a : astropy.units.Quantity -or- numpy.ndarray
            Grain radius value(s) to use in the calculation;
            if no units specified, defaults to micron
        
        theta : astropy.units.Quantity -or- numpy.ndarray -or- float
            Scattering angles for computing the differential scattering cross-section;
            if no units specified, defaults to radian

        Updates the `qsca`, `qext`, `qabs`, `diff`, `gsca`, and `qback` attributes
        """
        # Store the parameters
        _ = self._store_parameters(lam, a, None, theta)
        NE, NA, NTH = np.size(lam), np.size(a), np.size(theta)

        lam_value = lam.to(self.data_pars['lam'].unit, equivalencies=u.spectral()).value
        a_value   = a.to(self.data_pars['a'].unit).value
        th_value  = theta.to(self.data_pars['theta'].unit).value

        lam_mesh, a_mesh, th_mesh = np.meshgrid(lam_value, a_value, th_value)
        
        self.qext = self.qext_data(lam_mesh[:,:,0], a_mesh[:,:,0])
        self.qabs = self.qabs_data(lam_mesh[:,:,0], a_mesh[:,:,0])
        self.qsca = self.qsca_data(lam_mesh[:,:,0], a_mesh[:,:,0])
        self.diff = self.diff_data(lam_mesh, a_mesh, th_mesh)
        
        
