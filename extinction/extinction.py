import numpy as np
from scipy.integrate import trapz

import scatmodels
from .. import constants as c

__all__ = ['Extinction']

ALLOWED_SCATM = ['RG','Mie']
UNIT_LABELS = {'kev':'Energy (keV)', 'angs':'Wavelength (angs)'}

class Extinction(object):
    """
    | An extinction object contains information about the extinction properties of a particular dust population
    |
    | **ATTRIBUTES**
    | scatm    : string : The scattering model to use ('RG' or 'Mie')
    | lam      : The wavelength / energy grid used for calculation
    | lam_unit : The unit on the wavelength ('angs') or energy ('kev')
    | tau_sca  : Optical depth to scattering as a function of wavelength / energy
    | tau_abs  : Optical depth to absorption as a function of wavelength / energy
    | tau_ext  : Total extinction optical depth as a function of wavelength / energy
    | diff     : The differential scattering cross section [cm^2 / arcsec^2]
    | int_diff : GrainDist integrated differential scattering cross section [arcsec^2]
    |
    | *functions*
    | calculate(gdist, lam, unit = "kev")
    |   runs the scattering model calculation using and integrates over grain size distribution to get tau_sca, tau_abs, and tau_ext
    |   - ``gdist`` is an astrodust.graindist.GrainDist object
    |   - ``lam`` is the wavelength (unit = "angs") or energy (unit = "kev")
    | plot(ax, keyword) plots the extinction property specified by keyword
    |   - ``keyword`` options are "ext", "sca", "abs", "all"
    """

    def __init__(self, stype):
        assert stype in ALLOWED_SCATM
        if stype == 'RG':
            self.scatm = scatmodels.RGscat()
        if stype == 'Mie':
            self.scatm = scatmodels.Mie()

        self.tau_sca  = None  # NE
        self.tau_abs  = None  # NE
        self.tau_ext  = None  # NE
        self.diff     = None  # NE x NA x NTH [cm^2 / arcsec^2]
        self.int_diff = None  # NE x NTH [arcsec^2]
        self.lam      = None  # NE
        self.lam_unit = None  # string

    # Outputs should all be single arrays of length NE
    def calculate(self, gdist, lam, unit='kev', theta=0.0):
        self.scatm.calculate(lam, gdist.a, gdist.comp, unit=unit, theta=theta)
        self.lam      = lam
        self.lam_unit = unit
        NE, NA, NTH = np.shape(self.scatm.diff)
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

        # NE x NA x NTH
        self.diff     = self.scatm.diff * c.arcs2rad**2  # NE x NA x NTH, [cm^2 arcsec^-2]

        if np.size(gdist.a) == 1:
            int_diff = np.sum(self.scatm.diff * gdist.ndens[0] * c.arcs2rad**2, axis=1)
        else:
            agrid        = np.repeat(
                np.repeat(gdist.a.reshape(1, NA, 1), NE, axis=0),
                NTH, axis=2)
            ndgrid       = np.repeat(
                np.repeat(gdist.ndens.reshape(1, NA, 1), NE, axis=0),
                NTH, axis=2)
            int_diff = trapz(self.scatm.diff * ndgrid, agrid, axis=1) * c.arcs2rad**2

        self.int_diff = int_diff  # NE x NTH, [arcsec^-2]

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
