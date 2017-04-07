import numpy as np

import graindist
import extinction

__all__ = ['GrainPop','make_MRN','make_MRN_drude']

UNIT_LABELS = {'kev':'Energy (keV)', 'angs':'Wavelength (angs)'}
AMIN, AMAX, P = 0.005, 0.3, 3.5  # um, um, unitless

class GrainPop(object):
    """
    | A dust grain population.  Can contain multiple dust grain size distributions (GrainDist) and extinction models (Extinction).
    |
    | To initialize, input a list of tuples of GrainDist and Extinction pairs.
    | Keywords to describe each tuple are optional.  Default of keys=*None* will use integer index values as keys.
    |
    | Can add a string describing the Grain population using the `description` keyword
    |
    | **ATTRIBUTES**
    | gdist : GrainDist object
    | ext   : Extinction object
    | keys  : Key words to access GrainDist and Extinction tuples [default is to use integer index]
    | lam   : wavelength / energy grid for which the calculation was run
    | unit  : unit for the wavelength / energy grid
    | description : a string describing the grain population
    |
    | *properties*
    | tau_ext : Total extinction optical depth
    | tau_sca : Total scattering optical depth
    | tau_abs : Total absorption optical depth
    |
    | *functions*
    | __getitem__(key) Returns a tuple containing gdist and ext for the keys specified
    | calculate_ext(lam, unit='kev', **kwargs) runs the extinction calculation on the wavelength grid specified by lam and unit
    | plot(ax, keyword, **kwargs) plots the extinction property specified by keyword
    |   - ``keyword`` options are "ext", "sca", "abs", "all"
    """
    def __init__(self, pars, keys=None, description='Custom'):
        self.description  = description
        self.lam   = None
        self.unit  = None
        self.gdist = []
        self.ext   = []
        assert isinstance(pars, list)
        for p in pars:
            assert isinstance(p, tuple)
            gd, ex = p
            self.gdist.append(gd)
            self.ext.append(ex)
        if keys is None:
            self.keys = list(range(len(pars)))
        else:
            self.keys = keys

    def __getitem__(self, key):
        assert key in self.keys
        k = self.keys.index(key)
        return (self.gdist[k], self.ext[k])

    def calculate_ext(self, lam, unit='kev', **kwargs):
        for i in range(len(self.gdist)):
            self.ext[i].calculate(self.gdist[i], lam, unit=unit, **kwargs)
        self.lam  = lam
        self.unit = unit

    @property
    def tau_ext(self):
        result = 0.0
        for ee in self.ext:
            if ee.tau_ext is None:
                print("ERROR: Extinction properties need to be calculated")
                return 0.0
            else:
                result += ee.tau_ext
        return result

    @property
    def tau_sca(self):
        result = 0.0
        for ee in self.ext:
            if ee.tau_sca is None:
                print("ERROR: Extinction properties need to be calculated")
                return 0.0
            else:
                result += ee.tau_sca
        return result

    @property
    def tau_abs(self):
        result = 0.0
        for ee in self.ext:
            if ee.tau_abs is None:
                print("ERROR: Extinction properties need to be calculated")
                return 0.0
            else:
                result += ee.tau_abs
        return result

    def plot_ext(self, ax, keyword, **kwargs):
        assert keyword in ['ext','sca','abs','all']
        if keyword == 'ext':
            ax.plot(self.lam, self.tau_ext, **kwargs)
            ax.set_xlabel(UNIT_LABELS[self.unit])
            ax.set_ylabel(r"$\tau_{ext}$")
        if keyword == 'sca':
            ax.plot(self.lam, self.tau_sca, **kwargs)
            ax.set_xlabel(UNIT_LABELS[self.unit])
            ax.set_ylabel(r"$\tau_{sca}$")
        if keyword == 'abs':
            ax.plot(self.lam, self.tau_abs, **kwargs)
            ax.set_xlabel(UNIT_LABELS[self.unit])
            ax.set_ylabel(r"$\tau_{abs}$")
        if keyword == 'all':
            ax.plot(self.lam, self.tau_ext, 'k-', lw=2, label='Extinction')
            ax.plot(self.lam, self.tau_sca, 'r--', label='Scattering')
            ax.plot(self.lam, self.tau_abs, 'r:', label='Absorption')
            ax.set_xlabel(UNIT_LABELS[self.unit])
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
