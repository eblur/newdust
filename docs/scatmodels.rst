astrodust.extinction.scatmodels
===============================

Abstract Class: *ScatModel*
---------------------------

Abstract class *ScatModel* defines the scattering and extinction algorithm to
use.  An empty object is created after the initial call.  To calculate the
relevant values, run `ScatModel.calculate(*args, **kwargs)` with the desired
wavelength, grain size (um), complex index of refraction (see
*astrodust.graindist.composition*), wavelength unit ("kev" or "angs"), and
scattering angles (arcsec).

**Attributes**

- `stype` -- a string describing the model

- `cite` -- a citation for the model--

- `qsca` -- unitless scattering efficiency

- `qext` -- unit-less extinction efficiency

- `qabs` -- unit-less absorption efficiency

- `diff` -- differential scattering cross-section, cm^2 ster^-1

- `calculate (lam, a, cm, unit='kev')` -- calculates the above

Right now the only units accepted are "kev" and "angs"

Classes
-------

.. autoclass:: newdust.extinction.scatmodels.RGscat
.. autoclass:: newdust.extinction.scatmodels.Mie
