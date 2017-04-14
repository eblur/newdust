astrodust.grainpop
====================

This module provides the **SingleGrainPop** and **GrainPop** classes. They are
containers for ``GrainDist`` and ``Extinction`` objects.

**SingleGrainPop** contains a single ``GrainDist`` and ``Extinction`` object.

**GrainPop** contains a list of **SingleGrainPop** objects, which can be called
upon using keys, like a dictionary.  Keys have to be specified when the
**GrainPop** object is initialized.

Class structures
----------------

.. autoclass:: newdust.grainpop.SingleGrainPop
.. autoclass:: newdust.grainpop.GrainPop

Helper functions
----------------

.. autofunction:: newdust.grainpop.make_MRN
.. autofunction:: newdust.grainpop.make_MRN_drude

See also
--------

.. toctree::
   :maxdepth: 2

   graindist
   extinction
