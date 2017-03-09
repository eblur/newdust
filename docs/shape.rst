astrodust.graindist.shape
===============================

.. toctree::
   :maxdepth: 1

Abstract Class: *Shape*
--------------------------

Abstract class *Shape* describes the shape of the dust grains.
It must contain the following attributes:

- `shape` (a string)

- `vol` (a function giving the volume of the grain [cm^3])

- `cgeo` (a function giving the geometric area of the grain [cm^2])

Classes
-------

.. autoclass:: newdust.graindist.shape.Sphere
