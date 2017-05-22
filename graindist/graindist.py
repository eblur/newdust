import numpy as np
import sizedist
import composition
import shape as sh

MD_DEFAULT = 1.e-4  # g cm^-2
RHO        = 3.0    # g cm^-3
AMAX = 0.3  # um

ALLOWED_SIZES = ['Grain','Powerlaw','ExpCutoff']
ALLOWED_COMPS = ['Drude','Silicate','Graphite']
ALLOWED_SHAPE = ['Sphere']

__all__ = ['GrainDist']

class GrainDist(object):
    """
    | **ATTRIBUTES**
    | size  : Abstract class from astrodust.sizedist
    | comp  : Abstract class from astrodust.composition
    | shape : Abstract class from astrodust.shape
    | md    : float
    |
    | **__init__**
    | dtype   : 'Grain', 'Powerlaw' or 'ExpCutoff' (defines the grain size distribution)
    | cmtype  : 'Drude', 'Silicate' or 'Graphite' (defines the composition)
    | shape   : 'Sphere' is only option, otherwise use custom defined shape
    | md      : dust mass column (g cm^-2)
    | amax    : Defines the grain size distribution properties
    |   *Grain:* defines the singular grain size
    |   *Powerlaw:* defines the maximum grain size
    |   *ExpCutoff:* defines the *acut* value
    | rho     : if defined, will alter the rho keyword in composition
    | custom  : boolean : (default:False)
    |   if True, will set attributes of size, comp, and shape with raw input values
    | **kwargs : extra input to the size dist functions
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
    def __init__(self, dtype, cmtype, shape='Sphere', md=MD_DEFAULT,
                 amax=AMAX, rho=None, custom=False, **kwargs):
        self.md    = md
        if custom:
            self.size  = dtype
            self.comp  = cmtype
            self.shape = shape
        else:
            assert dtype in ALLOWED_SIZES
            if dtype == 'Grain':
                self.size = sizedist.Grain(rad=amax)
            if dtype == 'Powerlaw':
                self.size = sizedist.Powerlaw(amax=amax, **kwargs)
            if dtype == 'ExpCutoff':
                self.size = sizedist.ExpCutoff(acut=amax, **kwargs)

            assert cmtype in ALLOWED_COMPS
            if cmtype == 'Drude':
                if rho is not None: self.comp = composition.CmDrude(rho=rho)
                else: self.comp = composition.CmDrude()
            if cmtype == 'Silicate':
                if rho is not None: self.comp = composition.CmSilicate(rho=rho)
                else: self.comp = composition.CmSilicate()
            if cmtype == 'Graphite':
                if rho is not None: self.comp = composition.CmGraphite(rho=rho)
                else: self.comp = composition.CmGraphite()

            assert shape in ALLOWED_SHAPE
            if shape == 'Sphere':
                self.shape = sh.Sphere()

    @property
    def a(self):
        return self.size.a

    @property
    def ndens(self):
        return self.size.ndens(self.md, rho=self.comp.rho, shape=self.shape)

    @property
    def mdens(self):
        mg = self.sh.vol(self.a) * self.comp.rho  # mass of each dust grain [g]
        return self.ndens * mg

    @property
    def rho(self):
        return self.comp.rho

    @property
    def cgeo(self):
        return self.sh.cgeo(self.a)

    @property
    def vol(self):
        return self.sh.vol(self.a)

    def plot(self, ax=None, **kwargs):
        if isinstance(self.size, sizedist.Grain):
            print("Number density of dust grains = %.2e cm^-2" % self.ndens)
        else:
            ax.plot(self.a, self.ndens * np.power(self.a, 4), **kwargs)
            ax.set_xlabel("Radius (um)")
            ax.set_ylabel("$(dn/da) a^4$ (cm$^{-2}$ um$^{3}$)")
            ax.set_xscale('log')
            ax.set_yscale('log')
