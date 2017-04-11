import numpy as np

import graindist
import extinction

__all__ = ['SingleGrainPop','GrainPop','make_MRN','make_MRN_drude']

AMIN, AMAX, P = 0.005, 0.3, 3.5  # um, um, unitless
UNIT_LABELS = {'kev':'Energy (keV)', 'angs':'Wavelength (angs)'}

class SingleGrainPop(object):
    """
    | A single dust grain population.
    | Can add a string describing the Grain population using the `description` keyword
    |
    | **ATTRIBUTES**
    | gdist : GrainDist object
    | ext   : Extinction object
    | description : a string describing the grain population
    |
    | *properties*
    | tau_ext : Total extinction optical depth as a function of wavelength / energy
    | tau_sca : Total scattering optical depth as a function of wavelength / energy
    | tau_abs : Total absorption optical depth as a function of wavelength / energy
    |
    | *functions*
    | __getitem__(key) Returns a tuple containing gdist and ext for the keys specified
    | calculate_ext(lam, unit='kev', **kwargs) runs the extinction calculation on the wavelength grid specified by lam and unit
    | plot_sizes(ax, **kwargs) plots the size distribution (see *astrodust.graindist.sizedist.plot*)
    | plot_ext(ax, keyword, **kwargs) plots the extinction property specified by keyword
    |   - ``keyword`` options are "ext", "sca", "abs", "all"
    """
    def __init__(self, graindist, extinction, description='Custom'):
        self.description  = description
        self.gdist        = graindist
        self.ext          = extinction

    def calculate_ext(self, lam, unit='kev', **kwargs):
        self.ext.calculate(self.gdist, lam, unit=unit, **kwargs)

    def plot_sizes(self, ax=None, **kwargs):
        self.gdist.plot(ax, **kwargs)

    def plot_ext(self, ax, keyword, **kwargs):
        self.ext.plot(ax, keyword, **kwargs)

class GrainPop(object):
    def __init__(self, gpoplist, keys=None, description='Custom_GrainPopDict'):
        assert isinstance(gpoplist, list)
        if keys is None:
            self.keys = list(range(len(gpoplist)))
        else:
            self.keys = keys
        self.description = description
        self.gpops    = gpoplist

    def calculate_ext(self, lam, unit='kev', **kwargs):
        for gp in self.gpops:
            gp.calculate_ext(lam, unit=unit, **kwargs)
        self.lam = lam
        self.lam_unit = unit

    def __getitem__(self, key):
        assert key in self.keys
        k = self.keys.index(key)
        return self.gpops[k]

    @property
    def tau_ext(self):
        result = 0.0
        for gp in self.gpops:
            if gp.ext.tau_ext is None:
                print("ERROR: Extinction properties need to be calculated")
                return 0.0
            else:
                result += gp.ext.tau_ext
        return result

    @property
    def tau_sca(self):
        result = 0.0
        for gp in self.gpops:
            if gp.ext.tau_sca is None:
                print("ERROR: Extinction properties need to be calculated")
                return 0.0
            else:
                result += gp.ext.tau_sca
        return result

    @property
    def tau_abs(self):
        result = 0.0
        for gp in self.gpops:
            if gp.ext.tau_abs is None:
                print("ERROR: Extinction properties need to be calculated")
                return 0.0
            else:
                result += gp.ext.tau_abs
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
            ax.legend(**kwargs)


    def info(self, key=None):
        def _print(k):
            gd, ex = self[k]
            print("Grain Distribution of type %s" % gd.size.dtype)
            print("Extinction uses scattering model %s" % ex.scatm.stype)
        if key is None:
            for k in self.keys:
                print("Grain population %s:" % str(k))
                _print(k)
        else:
            assert key in self.keys
            _print(key)


#---------- Basic helper functions for fast production of GrainPop objects

def make_MRN(amin=AMIN, amax=AMAX, p=P):
    pl      = graindist.sizedist.Powerlaw(amin=amin, amax=amax, p=p)
    mrn_gra = graindist.GrainDist(pl, graindist.composition.CmGraphite())
    mrn_sil = graindist.GrainDist(pl, graindist.composition.CmSilicate())
    ext_gra = extinction.make_Extinction('Mie')
    ext_sil = extinction.make_Extinction('Mie')
    keys = ['gra','sil']
    pars = [(mrn_gra, ext_gra), (mrn_sil, ext_sil)]
    return GrainPop(pars, keys=keys, description='MRN')

def make_MRN_drude(amin=AMIN, amax=AMAX, p=P):
    pl      = graindist.sizedist.Powerlaw(amin=amin, amax=amax, p=p)
    mrn_dru = graindist.GrainDist(pl, graindist.composition.CmDrude())
    ext_rgd = extinction.make_Extinction('RG')
    keys = ['RGD']
    pars = [(mrn_dru, ext_rgd)]
    return GrainPop(pars, keys=keys, description='MRN_drude')
