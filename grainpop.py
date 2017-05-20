import numpy as np
from scipy.integrate import trapz

import graindist
import scatmodels
import constants as c

__all__ = ['SingleGrainPop','GrainPop','make_MRN','make_MRN_drude']

MD_DEFAULT    = 1.e-4  # g cm^-2
AMIN, AMAX, P = 0.005, 0.3, 3.5  # um, um, unitless
RHO_AVG       = 3.0  # g cm^-3
UNIT_LABELS   = {'kev':'Energy (keV)', 'angs':'Wavelength (angs)'}

ALLOWED_SCATM = ['RG','Mie']

# Make this a subclass of GrainDist at some point
class SingleGrainPop(object):
    """
    | A single dust grain population. Can add a string describing the Grain population using the *description* keyword
    |
    | **ATTRIBUTES**
    | description : a string describing the grain population
    | gdist : GrainDist object
    | stype : string : Scattering model to use
    |
    | *The following attributes are inherited form the GrainDist object*
    | a, ndens, mdens, cgeo, vol
    | *The following attributes are inherited from the Extinction object*
    | lam, lam_unit, tau_ext, tau_sca, tau_abs
    |
    | *functions*
    | calculate_ext(lam, unit='kev', **kwargs) runs the extinction calculation on the wavelength grid specified by lam and unit
    | plot_sizes(ax, **kwargs) plots the size distribution (see *astrodust.graindist.sizedist*)
    | plot_ext(ax, keyword, **kwargs) plots the extinction properties (see *astrodust.extinction*)
    |   - ``keyword`` options are "ext", "sca", "abs", "all"
    | info() prints information about the dust grain properties
    """
    def __init__(self, graindist, stype, description='Custom'):
        self.description  = description
        self.gdist        = graindist
        assert stype in ALLOWED_SCATM
        if stype == 'RG':
            self.scatm = scatmodels.RGscat()
        if stype == 'Mie':
            self.scatm = scatmodels.Mie()

        self.lam      = None  # NE
        self.lam_unit = None  # string
        self.tau_sca  = None  # NE
        self.tau_abs  = None  # NE
        self.tau_ext  = None  # NE
        self.diff     = None  # NE x NA x NTH [cm^2 / arcsec^2]
        self.int_diff = None  # NE x NTH [arcsec^2], differential xsect integrated over grain size

    def calculate_ext(self, lam, unit='kev', theta=0.0):
        self.scatm.calculate(lam, self.gdist.a, self.gdist.comp, unit=unit, theta=theta)
        self.lam      = lam
        self.lam_unit = unit
        NE, NA, NTH = np.shape(self.scatm.diff)
        # In single size grain case
        if len(self.gdist.a) == 1:
            self.tau_ext = self.gdist.ndens * self.scatm.qext[:,0] * self.gdist.cgeo
            self.tau_sca = self.gdist.ndens * self.scatm.qsca[:,0] * self.gdist.cgeo
            self.tau_abs = self.gdist.ndens * self.scatm.qabs[:,0] * self.gdist.cgeo
        # Otherwise, integrate over grain size (axis=1)
        else:
            geo_fac = self.gdist.ndens * self.gdist.cgeo  # array of length NA, unit is um^-1
            geo_2d  = np.repeat(geo_fac.reshape(1, NA), NE, axis=0)  # NE x NA
            self.tau_ext = trapz(geo_2d * self.scatm.qext, self.gdist.a, axis=1)
            self.tau_sca = trapz(geo_2d * self.scatm.qsca, self.gdist.a, axis=1)
            self.tau_abs = trapz(geo_2d * self.scatm.qabs, self.gdist.a, axis=1)

        # NE x NA x NTH
        self.diff     = self.scatm.diff * c.arcs2rad**2  # NE x NA x NTH, [cm^2 arcsec^-2]

        if np.size(self.gdist.a) == 1:
            int_diff = np.sum(self.scatm.diff * self.gdist.ndens[0] * c.arcs2rad**2, axis=1)
        else:
            agrid        = np.repeat(
                np.repeat(self.gdist.a.reshape(1, NA, 1), NE, axis=0),
                NTH, axis=2)
            ndgrid       = np.repeat(
                np.repeat(self.gdist.ndens.reshape(1, NA, 1), NE, axis=0),
                NTH, axis=2)
            int_diff = trapz(self.scatm.diff * ndgrid, agrid, axis=1) * c.arcs2rad**2

        self.int_diff = int_diff  # NE x NTH, [arcsec^-2]

    # Inheritance from gdist
    @property
    def a(self):
        return self.gdist.a

    @property
    def ndens(self):
        return self.gdist.ndens

    @property
    def mdens(self):
        return self.gdist.mdens

    @property
    def cgeo(self):
        return self.gdist.cgeo

    @property
    def vol(self):
        return self.gdist.vol

    # Plotting things
    def plot_sdist(self, ax=None, **kwargs):
        self.gdist.plot(ax, **kwargs)

    def plot_ext(self, ax, keyword, **kwargs):
        assert keyword in ['ext','sca','abs','all']
        try:
            assert self.lam is not None
        except:
            print("Need to run calculate_ext")
            pass
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

    # Printing information
    def info(self):
        print("Grain Population: %s" % self.description)
        print("Size distribution: %s" % self.gdist.size.dtype)
        print("Extinction calculated with: %s" % self.scatm.stype)
        print("Grain composition: %s" % self.gdist.comp.cmtype)
        print("rho = %.2f g cm^-3, M_d = %.2e g cm^-2" % (self.gdist.rho, self.gdist.md))

class GrainPop(object):
    """
    | A collection of dust grain distributions (SingeGrainPop).
    | Can add a string describing this Grain population using the `description` keyword
    |
    | **ATTRIBUTES**
    | keys     : A list of keys corresponding to each SingleGrainPop (default: list of integers starting with 0)
    | gpoplist : A list of SingleGrainPop objects
    | description : A string describing this collection
    | lam      : The energy / wavelength used for calculating extinction
    | lam_unit : The unit for energy ('kev') or wavelength ('angs') used for calculating extinction
    |
    | *properties*
    | tau_ext : Total extinction optical depth as a function of wavelength / energy
    | tau_sca : Total scattering optical depth as a function of wavelength / energy
    | tau_abs : Total absorption optical depth as a function of wavelength / energy
    |
    | *functions*
    | __getitem__(key) will return the SingleGrainPop indexed by ``key``
    | calculate_ext(lam, unit='kev', **kwargs) runs the extinction calculation on the wavelength grid specified by lam and unit
    | plot_ext(ax, keyword, **kwargs) plots the extinction properties (see *astrodust.extinction*)
    |   - ``keyword`` options are "ext", "sca", "abs", "all"
    | info(key=None) prints information about the SingleGrainPop indexed by ``key``
    |   - if ``key`` is *None*, information about every grain population will be printed to screen
    """
    def __init__(self, gpoplist, keys=None, description='Custom_GrainPopDict'):
        assert isinstance(gpoplist, list)
        if keys is None:
            self.keys = list(range(len(gpoplist)))
        else:
            self.keys = keys
        self.description = description
        self.gpoplist    = gpoplist
        for k in self.keys:
            i = self.keys.index(k)
            self.gpoplist[i].description = str(self.keys[i])
        self.lam = None
        self.lam_unit = None

    def calculate_ext(self, lam, unit='kev', **kwargs):
        for gp in self.gpoplist:
            gp.calculate_ext(lam, unit=unit, **kwargs)
        self.lam = lam
        self.lam_unit = unit

    def __getitem__(self, key):
        assert key in self.keys
        k = self.keys.index(key)
        return self.gpoplist[k]

    @property
    def md(self):
        result = 0.0
        for gp in self.gpoplist:
            result += gp.gdist.md
        return result

    @property
    def tau_ext(self):
        result = 0.0
        if self.lam is None:
            print("ERROR: Extinction properties need to be calculated")
        else:
            for gp in self.gpoplist:
                result += gp.tau_ext
        return result

    @property
    def tau_sca(self):
        result = 0.0
        if self.lam is None:
            print("ERROR: Extinction properties need to be calculated")
        else:
            for gp in self.gpoplist:
                result += gp.tau_sca
        return result

    @property
    def tau_abs(self):
        result = 0.0
        if self.lam is None:
            print("ERROR: Extinction properties need to be calculated")
        else:
            for gp in self.gpoplist:
                result += gp.tau_abs
        return result

    def plot_ext(self, ax, keyword, **kwargs):
        assert keyword in ['all','ext','sca','abs']
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
            ax.set_title(self.description)
            ax.legend(**kwargs)

    def info(self, key=None):
        if key is None:
            print("General information for %s dust grain population" % self.description)
            for gp in self.gpoplist:
                print("---")
                gp.info()
        else:
            assert key in self.keys
            self[key].info()


#---------- Basic helper functions for fast production of GrainPop objects

def make_MRN(amin=AMIN, amax=AMAX, p=P, md=MD_DEFAULT, fsil=0.6):
    """
    | Returns a GrainPop describing an MRN dust grain size distribution, which is a mixture of silicate and graphite grains.
    | Applies the 1/3 parallel, 2/3 perpendicular assumption of graphite grain orientations.
    |
    | **INPUTS**
    | amin : minimum grain size in microns
    | amax : maximum grain size in microns
    | p    : power law slope for grain size distribution
    | md   : dust mass column [g cm^-2]
    | fsil : fraction of dust mass in silicate grains
    """
    md_sil  = fsil * md
    # Graphite grain assumption: 1/3 parallel and 2/3 perpendicular
    md_gra_para = (1.0 - fsil) * md * (1.0/3.0)
    md_gra_perp = (1.0 - fsil) * md * (2.0/3.0)

    pl      = graindist.sizedist.Powerlaw(amin=amin, amax=amax, p=p)

    mrn_sil = graindist.GrainDist(pl, graindist.composition.CmSilicate(), md=md_sil)
    mrn_gra_para = graindist.GrainDist(pl, graindist.composition.CmGraphite(orient='para'), md=md_gra_para)
    mrn_gra_perp = graindist.GrainDist(pl, graindist.composition.CmGraphite(orient='perp'), md=md_gra_perp)

    gplist = [SingleGrainPop(mrn_sil, 'Mie'),
              SingleGrainPop(mrn_gra_para, 'Mie'),
              SingleGrainPop(mrn_gra_perp, 'Mie')]
    keys   = ['sil','gra_para','gra_perp']
    return GrainPop(gplist, keys=keys, description='MRN')

def make_MRN_drude(amin=AMIN, amax=AMAX, p=P, rho=RHO_AVG, md=MD_DEFAULT, **kwargs):
    """
    | Returns a GrainPop describing an MRN dust grain size distribution, and uses the Drude approximation,
    | which approximates the dust grain as a sphere of free electrons
    |
    | **INPUTS**
    | amin : minimum grain size in microns
    | amax : maximum grain size in microns
    | p    : power law slope for grain size distribution
    | rho  : density of dust grain material [g cm^-3]
    | md   : dust mass column [g cm^-2]
    """
    pl      = graindist.sizedist.Powerlaw(amin=amin, amax=amax, p=p, **kwargs)
    mrn_dru = graindist.GrainDist(pl, graindist.composition.CmDrude(), md=md)
    gplist  = [SingleGrainPop(mrn_dru, 'RG')]
    keys    = ['RGD']
    return GrainPop(gplist, keys=keys, description='MRN_rgd')
