import shape
import sizedist
import composition

MD_DEFAULT = 1.e-4  # g cm^-2
SHAPE      = shape.Sphere()

__all__ = ['GrainDist','make_GrainDist']

class GrainDist(object):
    """
    | **ATTRIBUTES**
    | sizedist
    | composition
    | shape
    | md
    |
    | *functions*
    | a
    | ndens
    """
    def __init__(self, sizedist, composition, shape=shape.Sphere(), md=MD_DEFAULT):
        self.size = sizedist
        self.comp = composition
        self.shape = shape
        self.md    = md

    @property
    def a(self):
        return self.size.a

    @property
    def ndens(self):
        return self.size.ndens(self.md, rho=self.comp.rho, shape=self.shape)

#-- Helper functions
ALLOWED_SIZES = ['Grain','Powerlaw','ExpCutoff']
ALLOWED_COMPS = ['Drude','Silicate','Graphite']
AMAX = 0.3  # um

def make_GrainDist(sstring, cstring, amax=AMAX, md=MD_DEFAULT):
    """
    | A shortcut for creating GrainDist objects
    """
    assert sstring in ALLOWED_SIZES
    assert cstring in ALLOWED_COMPS
    if sstring == 'Grain':
        sdist = sizedist.Grain(rad=amax)
    if sstring == 'Powerlaw':
        sdist = sizedist.Powerlaw(amax=amax)
    if sstring == 'ExpCutoff':
        sdist = sizedist.ExpCutoff(acut=amax)
    if cstring == 'Drude':
        cmi = composition.CmDrude()
    if cstring == 'Silicate':
        cmi = composition.CmSilicate()
    if cstring == 'Graphite':
        cmi = composition.CmGraphite()
    return GrainDist(sdist, cmi, md=md)
