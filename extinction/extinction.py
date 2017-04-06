import numpy as np
from scipy.integrate import trapz

__all__ = ['Extinction']

class Extinction(object):
    """
    | An extinction object contains information about the extinction properties of a particular dust population
    |
    | **ATTRIBUTES**
    | scatm   : The scattering model used
    | tau_sca
    | tau_abs
    | tau_ext (calculated form tau_sca and tau_abs)
    |
    | *functions*
    | calculate(gdist, lam, unit='kev')
    |   runs the scattering model calculation using and integrates over grain size distribution to get tau_sca, abs, and ext
    """

    def __init__(self, scatm):
        self.scatm = scatm
        self.tau_sca = None
        self.tau_abs = None
        self.tau_ext = None

    # Outputs should all be single arrays of length NE
    def calculate(self, gdist, lam, unit='kev', theta=0.0):
        self.scatm.calculate(lam, gdist.a, gdist.comp, unit=unit, theta=theta)
        NE, NA = np.shape(self.scatm.qext)

        # In single size grain case
        if len(gdist.a) == 1:
            self.tau_ext = gdist.ndens * self.scatm.qext[:,0] * gdist.cgeo
            self.tau_sca = gdist.ndens * self.scatm.qsca[:,0] * gdist.cgeo
            self.tau_abs = gdist.ndens * self.scatm.qabs[:,0] * gdist.cgeo
        # Otherwise, integrate over grain size (axis=1)
        else:
            geo_fac = gdist.ndens * gdist.cgeo  # array of length NA, unit is um^-1
            geo_2d  = np.repeat(geo_fac.reshape(1, NA), NE, axis=0)  # NE x NA
            self.tau_ext = trapz(geo_2d * self.scatm.qext, gdist.a, axis=1)
            self.tau_sca = trapz(geo_2d * self.scatm.qsca, gdist.a, axis=1)
            self.tau_abs = trapz(geo_2d * self.scatm.qabs, gdist.a, axis=1)
