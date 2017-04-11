import shape
import sizedist
import composition

MD_DEFAULT = 1.e-4  # g cm^-2
RHO        = 3.0    # g cm^-3
SHAPE      = shape.Sphere()

__all__ = ['GrainDist','make_GrainDist']

class GrainDist(object):
    """
    | **ATTRIBUTES**
    | size  : Abstract class from astrodust.sizedist
    | comp  : Abstract class from astrodust.composition
    | shape : Abstract class from astrodust.shape
    | md    : float
    |
    | *properties*
    | a     : grain radii from size.a
    | ndens : number density from size.ndens using the other attributes as input
    | mdens : mass density as a function of grain size
    | cgeo  : physical cross-sectional area based on grain shape
    | vol   : physical grain volume based on grain shape
    |
    | *functions*
    | plot(ax, kwargs) : Plots the number density of dust grains via size.plot()
    """
    def __init__(self, sizedist, composition, shape=SHAPE, md=MD_DEFAULT):
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

    @property
    def mdens(self):
        mg = self.shape.vol(self.a) * self.comp.rho  # mass of each dust grain [g]
        return self.ndens * mg

    @property
    def rho(self):
        return self.comp.rho

    @property
    def cgeo(self):
        return self.shape.cgeo(self.a)

    @property
    def vol(self):
        return self.shape.vol(self.a)

    def plot(self, ax=None, **kwargs):
        if isinstance(self.size, sizedist.Grain):
            print("Number density of dust grains = %.2e cm^-2" % self.ndens)
        else:
            self.size.plot(ax, md=self.md, rho=self.comp.rho, shape=self.shape, **kwargs)

#-- Helper functions
ALLOWED_SIZES = ['Grain','Powerlaw','ExpCutoff']
ALLOWED_COMPS = ['Drude','Silicate','Graphite']
AMAX = 0.3  # um

def make_GrainDist(sstring, cstring, amax=AMAX, rho=None, md=MD_DEFAULT):
    """
    | A shortcut function for creating GrainDist objects
    |
    | **INPUTS**
    | sstring : 'Grain', 'Powerlaw' or 'ExpCutoff' (defines the grain size distribution)
    | cstring : 'Drude', 'Silicate' or 'Graphite' (defines the composition)
    | amax    : Defines the grain size distribution properties
    |   *Grain:* defines the singular grain size
    |   *Powerlaw:* defines the maximum grain size
    |   *ExpCutoff:* defines the *acut* value
    | rho     : if defined, will alter the rho keyword in composition
    | md      : dust mass column (g cm^-2)
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
        if rho is not None: cmi = composition.CmDrude(rho=rho)
        else: cmi = composition.CmDrude()
    if cstring == 'Silicate':
        if rho is not None: cmi = composition.CmSilicate(rho=rho)
        else: cmi = composition.CmSilicate()
    if cstring == 'Graphite':
        if rho is not None: cmi = composition.CmGraphite(rho=rho)
        else: cmi = composition.CmGraphite()
    return GrainDist(sdist, cmi, md=md)
