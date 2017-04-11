import numpy as np
from scipy.integrate import trapz

import scatmodels

__all__ = ['Extinction','make_Extinction']

ALLOWED_SCATM = ['RG','Mie']
UNIT_LABELS = {'kev':'Energy (keV)', 'angs':'Wavelength (angs)'}

class Extinction(object):
    """
    | An extinction object contains information about the extinction properties of a particular dust population
    |
    | **ATTRIBUTES**
    | scatm    : The scattering model used
    | lam      : The wavelength / energy grid used for calculation
    | lam_unit : The unit on the wavelength ('angs') or energy ('kev')
    | tau_sca  : Optical depth to scattering as a function of wavelength / energy
    | tau_abs  : Optical depth to absorption as a function of wavelength / energy
    | tau_ext  : Total extinction optical depth as a function of wavelength / energy
    |
    | *functions*
    | calculate(gdist, lam, unit = "kev")
    |   runs the scattering model calculation using and integrates over grain size distribution to get tau_sca, tau_abs, and tau_ext
    |   - ``gdist`` is an astrodust.graindist.GrainDist object
    |   - ``lam`` is the wavelength (unit = "angs") or energy (unit = "kev")
    """

    def __init__(self, scatm):
        self.scatm    = scatm
        self.tau_sca  = None
        self.tau_abs  = None
        self.tau_ext  = None
        self.lam      = None
        self.lam_unit = None

    # Outputs should all be single arrays of length NE
    def calculate(self, gdist, lam, unit='kev', theta=0.0):
        self.scatm.calculate(lam, gdist.a, gdist.comp, unit=unit, theta=theta)
        self.lam      = lam
        self.lam_unit = unit
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

    def plot(self, ax, keyword, **kwargs):
        assert keyword in ['ext','sca','abs','all']
        if keyword == 'ext':
            ax.plot(self.lam, self.tau_ext, **kwargs)
            ax.set_xlabel(UNIT_LABELS[self.lam_unit])
            ax.set_ylabel(r"$\tau_{ext}$")
        if keyword == 'sca':
            ax.plot(self.lam, self.tau_sca, **kwargs)
            ax.set_xlabel(UNIT_LABELS[self.lam_unit])
            ax.set_ylabel(r"$\tau_{sca}$")
        if keyword == 'abs':
            ax.plot(self.lam, self.tau_abs, **kwargs)
            ax.set_xlabel(UNIT_LABELS[self.lam_unit])
            ax.set_ylabel(r"$\tau_{abs}$")
        if keyword == 'all':
            ax.plot(self.lam, self.tau_ext, 'k-', lw=2, label='Extinction')
            ax.plot(self.lam, self.tau_sca, 'r--', label='Scattering')
            ax.plot(self.lam, self.tau_abs, 'r:', label='Absorption')
            ax.set_xlabel(UNIT_LABELS[self.lam_unit])
            ax.set_ylabel(r"$\tau$")
            ax.legend(**kwargs)


#---------- Helper functions

def make_Extinction(estring):
    """
    | Return an Extinction object with a particular scattering model
    |
    | **INPUTS**
    | estring : 'RG' or 'Mie'
    """
    assert estring in ALLOWED_SCATM
    if estring == 'RG':
        return Extinction(scatmodels.RGscat())
    if estring == 'Mie':
        return Extinction(scatmodels.Mie())
