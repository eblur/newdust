from .rgscat import RGscat
from .miescat import Mie
from .. import constants as c

"""
--------------------------------------------------------------
    API for Abstract Class 'ScatModel'
--------------------------------------------------------------
 A dust scattering model should contain the following attributes

calculate( lam : scalar or np.array [wavelength or energy grid, keV default]
           a   : scalar [grain size, micron]
           cm  : newdust.graindist.composition cm object (abstract class)
           unit = : string ['kev', 'angs']
           theta = : scalar or np.array [angles to calculate differential scattering, arcsec, default 0.0]
           **kwargs
           )

qsca : np.array, scattering efficiency [unitless]
qabs : np.array, absorption efficiency [unitless]
qext : np.array, extinction efficiency [unitless]
self.diff : np.array, differentifal scattering cross section [cm**2/ster]
"""
