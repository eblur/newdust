import numpy as np

import graindist
import extinction

__all__ = ['GrainPop']

UNIT_LABELS = {'kev':'Energy (keV)', 'angs':'Wavelength (angs)'}

class GrainPop(object):
    """
    | A dust grain population.  Can contain multiple dust grain size distributions (GrainDist) and extinction models (Extinction).
    |
    | To initialize, give a list of tuples of GrainDist and Extinction pairs.  Keywords to describe each tuple are optional.  Default of *None* will use integer index values as keys.
    |
    | **ATTRIBUTES**
    | gdist : GrainDist object
    | ext   : Extinction object
    | keys  : Key words to access GrainDist and Extinction tuples [default is to use integer index]
    | lam   : wavelength / energy grid for which the calculation was run
    | unit  : unit for the wavelength / energy grid
    |
    | *properties*
    | tau_ext : Total extinction
    | tau_sca : Total scattering
    | tau_abs : Total absorption
    |
    | *functions*
    | __getitem__(key): Returns a tuple containing gdist and ext for the keys specified
    """
    def __init__(self, pars, keys=None):
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
