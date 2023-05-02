from .grain import Grain
from .powerlaw import Powerlaw
from .exp_cutoff import ExpCutoff
from .astrodust import Astrodust

from .. import shape

"""
-------
API
-------

*Sizedist* must contain attributes

`dtype` : a string description

`a` : an array

And must contain the following two methods:

`ndens` (md, rho, shape) returns number density of dust grains [e.g. cm^-2 um^-1]

`mdens` (md, rho, shape) returns mass density of dust grains [e.g. g cm^-2 um^-1]
"""

