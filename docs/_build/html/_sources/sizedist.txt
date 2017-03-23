astrodust.graindist.sizedist
============================

.. toctree::
   :maxdepth: 1


Abstract Class: *Sizedist*
--------------------------

Abstract class *Sizedist* must contain attributes:

- `a` (an array)

- `ndens` (md, rho, shape) returns number density of dust grains [e.g. cm^-2 um^-1]

- `mdens` (md, rho, shape) returns mass density of dust grains [e.g. g cm^-2 um^-1]

Classes
-------

.. autoclass:: newdust.graindist.sizedist.Grain
.. autoclass:: newdust.graindist.sizedist.Powerlaw
.. autoclass:: newdust.graindist.sizedist.ExpCutoff
