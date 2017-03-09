astrodust.graindist.composition
===============================

.. toctree::
   :maxdepth: 1

   minerals

Abstract Class: *CmIndex*
--------------------------

Abstract class *CmIndex* describes the complex index of refraction for a particular material.
It must contain the following attributes:

- `cmtype` (a string)

- `rho` (g cm^-3)

- `rp` (a function that takes wavelength ["angs"] or energy ["kev"] and returns the real part of the complex index of refraction)

- `ip` (a function that takes wavelength ["angs"] or energy ["kev"] and returns the imaginary part of the complex index of refraction)

- `cm` (a function that takes wavelengths ["angs"] or energy ["kev"] and returns the complex index of refraction, *dtype='complex'*)

Right now the only units accepted are "kev" and "angs"

Classes
-------

.. autoclass:: newdust.graindist.composition.CmDrude
.. autoclass:: newdust.graindist.composition.CmSilicate
.. autoclass:: newdust.graindist.composition.CmGraphite
