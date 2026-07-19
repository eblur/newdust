.. _composition:

graindist.composition
=====================

The composition library is built on the superclass :ref:`CompositionClass`, which holds the optical constants and material density [g cm^-3] of a grain composition type.

**Premade composition types**

* :ref:`CmDrude` -- Drude approximation, treating the grain as a mass of free electrons
* :ref:`CmSilicate` -- astronomical silicates (Draine 2003)
* :ref:`CmGraphite` -- graphite (Draine 2003)


.. _CompositionClass:

Composition
-----------

.. autoclass:: xdust.graindist.composition.Composition

.. _CmDrude:

CmDrude
-------
.. autoclass:: xdust.graindist.composition.CmDrude

.. _CmSilicate:

CmSilicate
----------
.. autoclass:: xdust.graindist.composition.CmSilicate

.. _CmGraphite:

CmGraphite
----------
.. autoclass:: xdust.graindist.composition.CmGraphite
