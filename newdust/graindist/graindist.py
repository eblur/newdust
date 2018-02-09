import numpy as np
from . import sizedist
from . import composition
from . import shape as sh

MD_DEFAULT = 1.e-4  # g cm^-2
RHO        = 3.0    # g cm^-3
AMAX = 0.3  # um

ALLOWED_SIZES  = ['Grain','Powerlaw','ExpCutoff']
ALLOWED_COMPS  = ['Drude', 'Silicate', 'Graphite']
SHAPES = {'Sphere':sh.Sphere()}

__all__ = ['GrainDist']

class GrainDist(object):
    """
    | **ATTRIBUTES**
    | size  : Abstract class from newdust.graindist.sizedist
    | comp  : Abstract class from newdust.graindist.composition
    | shape : Abstract class from newdust.graindist.shape
    | md    : float
    | a     : grain radii from size.a
    | ndens : number density from size.ndens using the other attributes as input
    | mdens : mass density as a function of grain size
    | cgeo  : physical cross-sectional area based on grain shape
    | vol   : physical grain volume based on grain shape
    |
    | *methods*
    | plot(ax, kwargs) : Plots the number density of dust grains via size.plot()
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
    |   if True, will set attributes of size, comp, and shape with raw input values
    | **kwargs : extra input to the size dist functions
    |
    """
    def __init__(self, dtype, cmtype, shape='Sphere', md=MD_DEFAULT,
                 amax=AMAX, rho=None, **kwargs):

        self.md = md

        if isinstance(dtype, str):
            self._assign_sizedist_from_string(dtype, amax, **kwargs)
        else:
            self.size = dtype

        if isinstance(cmtype, str):
            self._assign_comp_from_string(cmtype, rho)
        else:
            self.comp = cmtype

        if isinstance(shape, str):
            self._assign_shape_from_string(shape)
        else:
            self.shape = shape


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

    def plot(self, ax, **kwargs):
        ax.plot(self.a, self.ndens * np.power(self.a, 4), **kwargs)
        ax.set_xlabel("Radius (um)")
        ax.set_ylabel("$(dn/da) a^4$ (cm$^{-2}$ um$^{3}$)")
        ax.set_xscale('log')
        ax.set_yscale('log')
        return

    def _assign_sizedist_from_string(self, dtype, amax, **kwargs):
        assert dtype in ALLOWED_SIZES
        if dtype == 'Grain':
            self.size = sizedist.Grain(rad=amax)
        if dtype == 'Powerlaw':
            self.size = sizedist.Powerlaw(amax=amax, **kwargs)
        if dtype == 'ExpCutoff':
            self.size = sizedist.ExpCutoff(acut=amax, **kwargs)
        return

    def _assign_comp_from_string(self, cmtype, rho):
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
        return

    def _assign_shape_from_string(self, shape):
        assert shape in SHAPES.keys()
        self.shape = SHAPES[shape]
        return
