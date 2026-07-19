.. _graindist:

graindist
=========

This module contains:

* a library of size distribution functions (see :ref:`sizedist`)
* a library of grain compositions, which defines the optical constants and material density (see :ref:`composition`)
* a shape module (see :ref:`shape`)

This is tied together by the :ref:`GrainDistClass` superclass.

.. _GrainDistClass:
   
GrainDist
----------

.. autoclass:: xdust.graindist.GrainDist

Supporting modules
------------------

.. toctree::
   :maxdepth: 2

   sizedist
   composition
   shape
