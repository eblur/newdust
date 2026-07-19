.. _halos:

halos
=====

The superclass :ref:`Halo` holds all information and special functions for 
working with a simulated X-ray dust scattering halo. 

The :ref:`UniformGalHalo` object assumes that all dust is uniformly distributed 
along the line-of-sight towards the X-ray light source.

The :ref:`ScreenGalHalo` object assumes that all dust is confined within an 
infinitesimally thin sheet with the specified column density.

See Corrales & Paerels (2015) for a description of the mathematics and geometry.

.. warning::

   In the *xdust* code, the position of the scattering material (:math:`x`)
   is defined as :math:`x = 1 - d/D`, where :math:`d` is the distance to the
   scattering material and :math:`D` is the distance to the X-ray light source.
   This is the *opposite* of the conventions used in many other papers on X-ray dust
   scattering halos, e.g., from R. K. Smith and S. Heinz.

Any scattering model physics can be used in the halo calculations. For 
the semi-analytic (fast) model described in Corrales & Paerels 2015, utilizing 
the Rayleigh-Gans approximation with the Drude approximation for the complex 
index of refraction, on can use the ``UniformGalHaloCP15`` and ``ScrenGalHaloCP15`` 
objects.

**Classes**

* :ref:`UniformGalHalo`
* :ref:`ScreenGalHalo`
* :ref:`UniformGalHaloCP15`
* :ref:`ScreenGalHaloCP15`

**Functions**

* :ref:`path_diff`
* :ref:`time_delay`
* :ref:`calculate_taux`

.. _Halo:

Halo
----

.. autoclass:: xdust.halos.Halo

.. _UniformGalHalo:

UniformGalHalo
--------------

.. autoclass:: xdust.halos.galhalo.UniformGalHalo

.. _ScreenGalHalo:

ScreenGalHalo
-------------

.. autoclass:: xdust.halos.galhalo.ScreenGalHalo

.. _UniformGalHaloCP15:

UniformGalHaloCP15
------------------

.. autoclass:: xdust.halos.galhalo.UniformGalHaloCP15

.. _ScreenGalHaloCP15:

ScreenGalHaloCP15
-----------------

.. autoclass:: xdust.halos.galhalo.ScreenGalHaloCP15

.. _path_diff:

path_diff
---------

.. autofunction:: xdust.halos.galhalo.path_diff

.. _time_delay:

time_delay
----------

.. autofunction:: xdust.halos.galhalo.time_delay

.. _calculate_taux:

calculate_taux
--------------

.. autofunction:: xdust.halos.galhalo.calculate_taux
