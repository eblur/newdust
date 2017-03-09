astrodust.graindist.sizedist
============================

.. toctree::
   :maxdepth: 1


Abstract Class: *Sizedist*
--------------------------

Abstract class *Sizedist* must contain attributes:

- `a` (an array)

- `ndens` (function that takes some normalization value or set of values)

Right now `ndens` only takes M_d and \rho as inputs.

Classes
-------

.. autoclass:: newdust.graindist.sizedist.Grain
.. autoclass:: newdust.graindist.sizedist.Powerlaw
.. autoclass:: newdust.graindist.sizedist.ExpCutoff
