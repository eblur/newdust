.. _scatmodels:

scatteringmodel
===============

The scattering model library is built on the superclass :ref:`ScatteringModelClass`, which defines the scattering physics model and holds the 
computational results once the scattering model is calculated. 

**Premade scattering models**

* :ref:`RGscattering`
* :ref:`Mie`
* :ref:`GGADT`
* :ref:`PAH`

.. _ScatteringModelClass:

ScatteringModel
----------------

.. autoclass:: xdust.scatteringmodel.ScatteringModel


.. _RGscattering:

RGscattering
------------
.. autoclass:: xdust.scatteringmodel.RGscattering

.. _Mie:

Mie
---
.. autoclass:: xdust.scatteringmodel.Mie

.. _GGADT:

GGADT
-----
.. autoclass:: xdust.scatteringmodel.GGADT

.. _PAH:

PAH
---
.. autoclass:: xdust.scatteringmodel.PAH