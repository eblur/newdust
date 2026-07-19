.. _shape:

graindist.shape
===============

The superclass :ref:`ShapeClass` describes the geometric properties of the dust grains.

.. note::
   At this time, only spherical dust grains are supported (see :ref:`Sphere`).

For non-spherical dust grains, the grain size distribution 
can be specified using the effective grain radius :math:`a_{\rm eff}`, defined as
:math:`V_g = \frac{4}{3} \pi a_{\rm eff}^3` where :math:`V_g` is the volume of the grain.

.. _ShapeClass:

Shape
-----

.. autoclass:: xdust.graindist.shape.Shape

.. _Sphere:

Sphere
------
.. autoclass:: xdust.graindist.shape.Sphere
