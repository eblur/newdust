.. _sizedist:

graindist.sizedist
==================

The size distribution library is built on the superclass :ref:`SizedistClass`, which holds the grain size distribution function and the grain size array. 

The grain size distribution function is defined as the number of grains per unit volume per unit grain size, i.e. :math:`dn/da`.

.. note:: 
   Within this library the functions are used to describe column density :math:`N(a) = \int dn/da \, dl`, where :math:`dl` is the path length along the line of sight. 
   However, one can always convert between column density and number density by dividing by the path length, i.e. :math:`n(a) = N(a)/L`.

**Premade size distributions**

* :ref:`Grain`
* :ref:`Powerlaw`
* :ref:`ExpCutoff`
* :ref:`WD01`
* :ref:`Astrodust`

.. _SizedistClass:

Sizedist
--------

.. autoclass:: xdust.graindist.sizedist.Sizedist


.. _Grain:

Grain
-----
.. autoclass:: xdust.graindist.sizedist.Grain

.. _Powerlaw:

Powerlaw
--------
.. autoclass:: xdust.graindist.sizedist.Powerlaw

.. _ExpCutoff:

ExpCutoff
---------
.. autoclass:: xdust.graindist.sizedist.ExpCutoff

.. _WD01:

WD01
----
.. autoclass:: xdust.graindist.sizedist.WD01

.. _Astrodust:

Astrodust
---------
.. autoclass:: xdust.graindist.sizedist.Astrodust


