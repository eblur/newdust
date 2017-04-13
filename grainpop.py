import numpy as np
import graindist
import extinction

__all__ = ['SingleGrainPop','GrainPop','make_MRN','make_MRN_drude']

MD_DEFAULT    = 1.e-4  # g cm^-2
AMIN, AMAX, P = 0.005, 0.3, 3.5  # um, um, unitless
UNIT_LABELS   = {'kev':'Energy (keV)', 'angs':'Wavelength (angs)'}

class SingleGrainPop(object):
    """
    | A single dust grain population.
    | Can add a string describing the Grain population using the `description` keyword
    |
    | **ATTRIBUTES**
    | description : a string describing the grain population
    | gdist : GrainDist object
    | ext   : Extinction object
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
    def __init__(self, graindist, extinction, description='Custom'):
        self.description  = description
        self.gdist        = graindist
        self.ext          = extinction

    # Hard code inheritance. This is annoying.
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

    # Inheritance from extinction
    @property
    def tau_ext(self):
        return self.ext.tau_ext

    @property
    def tau_sca(self):
        return self.ext.tau_sca

    @property
    def tau_abs(self):
        return self.ext.tau_abs

    @property
    def lam(self):
        return self.ext.lam

    @property
    def lam_unit(self):
        return self.ext.lam_unit

    # Calculating the extinction properties
    def calculate_ext(self, lam, unit='kev', **kwargs):
        self.ext.calculate(self.gdist, lam, unit=unit, **kwargs)

    # Plotting things
    def plot_sizes(self, ax=None, **kwargs):
        self.gdist.plot(ax, **kwargs)

    def plot_ext(self, ax, keyword, **kwargs):
        self.ext.plot(ax, keyword, **kwargs)

    # Printing information
    def info(self):
        print("Grain Population: %s" % self.description)
        print("Size distribution: %s" % self.gdist.size.dtype)
        print("Extinction calculated with: %s" % self.ext.scatm.stype)
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
    #assert fsil != 0.0
    #assert fsil != 1.0
    md_sil  = fsil * md
    # Graphite grain assumption: 1/3 parallel and 2/3 perpendicular
    md_gra_para = (1.0 - fsil) * md * (1.0/3.0)
    md_gra_perp = (1.0 - fsil) * md * (2.0/3.0)

    pl      = graindist.sizedist.Powerlaw(amin=amin, amax=amax, p=p)

    mrn_sil = graindist.GrainDist(pl, graindist.composition.CmSilicate(), md=md_sil)
    mrn_gra_para = graindist.GrainDist(pl, graindist.composition.CmGraphite(orient='para'), md=md_gra_para)
    mrn_gra_perp = graindist.GrainDist(pl, graindist.composition.CmGraphite(orient='perp'), md=md_gra_perp)

    gplist = [SingleGrainPop(mrn_sil, extinction.make_Extinction('Mie')),
              SingleGrainPop(mrn_gra_para, extinction.make_Extinction('Mie')),
              SingleGrainPop(mrn_gra_perp, extinction.make_Extinction('Mie'))]
    keys   = ['sil','gra_para','gra_perp']
    return GrainPop(gplist, keys=keys, description='MRN')

def make_MRN_drude(amin=AMIN, amax=AMAX, p=P, md=MD_DEFAULT):
    pl      = graindist.sizedist.Powerlaw(amin=amin, amax=amax, p=p)
    mrn_dru = graindist.GrainDist(pl, graindist.composition.CmDrude(), md=md)
    gplist  = [SingleGrainPop(mrn_dru, extinction.make_Extinction('RG'))]
    keys    = ['RGD']
    return GrainPop(gplist, keys=keys, description='MRN_rgd')
