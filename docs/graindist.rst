astrodust.graindist
===================

This module provides the essential **GrainDist** class, which contains
information about the the grain size distribution, composition (e.g. Graphite vs
Silicate), and shape.  Right now only spherical dust grain shapes are supported.

GrainDist class
===============

.. autoclass:: newdust.graindist.GrainDist

Helper functions
================

.. autofunction:: newdust.graindist.make_GrainDist

Supporting modules
==================

.. toctree::
   :maxdepth: 2

   sizedist
   composition
   shape
