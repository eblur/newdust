.. _grainpop:

grainpop
========

The :ref:`SingleGrainPop` object is used to model a population of dust grains 
based on three distinct properties:

* the grain size distribution (see :ref:`sizedist`)
* composition type specifying the optical constants (see :ref:`composition`)
* the scattering physics model (see :ref:`scatmodels`)

Furthermore, the :ref:`GrainPopClass` object contains a list of :ref:`SingleGrainPop` objects, 
which can be called upon using keys, like a dictionary.  Keys have to be specified 
when the :ref:`GrainPopClass` object is initialized.

**Module provides**

* :ref:`SingleGrainPop` class
* :ref:`GrainPopClass` class
* :ref:`grainpop_helper`


.. _SingleGrainPop:

SingleGrainPop
--------------

.. autoclass:: xdust.grainpop.SingleGrainPop

.. _GrainPopClass:

GrainPop
--------

.. autoclass:: xdust.grainpop.GrainPop

.. _grainpop_helper:

Helper functions
----------------

.. autofunction:: xdust.grainpop.make_MRN
.. autofunction:: xdust.grainpop.make_MRN_RGDrude
