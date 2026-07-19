.. xdust documentation master file, created by
   sphinx-quickstart on Sunday May 30 2026
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _index:

*Xdust* documentation
=====================

The *xdust* library calculates scattering and absorption efficiencies 
for dust from the infrared to the X-ray. It has special capabilities 
to calculate dust scattering halo images in the X-ray.

**First published version of this code** (released with `Corrales & Paerels, 2015 <http://adsabs.harvard.edu/abs/2015MNRAS.453.1121C>`_)
http://dx.doi.org/10.5281/zenodo.15991

**Source code:** `github.com/eblur/xdust <https://github.com/eblur/xdust>`_

**Support:** If you are having issues, please contact liac@umich.edu

Features
--------

A number of dust grain size distributions and optical constants are
provided, but they can be fully customized by the user by invoking
custom objects of the approporiate class.  Provided dust models
include:

* Grain size distributions: single grain size, power law, and a power law with an exponential cut-off
* `Weingartner & Draine (2001) <http://adsabs.harvard.edu/abs/2001ApJ...548..296W>`_
  grain size distributions for Milky Way dust
* The new `Hensley & Draine (2022) Astrodust <url>`_ size distribution (assuming spherical grains)
* Optical constants (complex index of refraction) for 0.1 um sized
  `graphite and astrosilicate grains <https://www.astro.princeton.edu/~draine/dust/dust.diel.html>`_

* Rayleigh-Gans scattering physics following:

  * `Smith & Dwek (1998) <http://adsabs.harvard.edu/abs/1998ApJ...503..831S>`_
  * `Mauche & Gorenstein (1986) <http://adsabs.harvard.edu/abs/1986ApJ...302..371M>`_

* Mie scattering physics using the algorithms of
  `Bohren & Huffman (1986) <http://adsabs.harvard.edu/abs/1983asls.book.....B>`_ converted from 
  `fortran and IDL <http://www.met.tamu.edu/class/atmo689-lc/bhmie.pro>`_ to python


Grain Populations
-----------------

At the top level, *Xdust* uses the ``SingleGrainPop`` object to 
model a population of dust grains based on three distinct properties:

* the grain size distribution (see ``xdust.graindist.sizedist``)
* composition type specifying the optical constants (see ``xdust.graindist.composition``)
* the scattering physics model (see ``xdust.scatteringmodel``)

Furthermore, the **GrainPop** object contains a list of **SingleGrainPop** objects, 
which can be called upon using keys, like a dictionary.  Keys have to be specified 
when the **GrainPop** object is initialized.

For example, when the ``xdust.make_MRN`` function is called, it will return a 
**GrainPop** object with the keys `'sil'` for the silicate grain size distribution, 
`'gra_para'` for the graphitic grains with monomers parallel to the light wave propagation, 
and `'gra_perp'` for the graphitic grains with monomers perpendicular to the light wave propagation. 
See the ``xdust.grainpop`` module for more information.

X-ray Scattering Halos
----------------------

The ``xdust.halos`` module provides tools for calculating the intensity of 
X-ray dust scattering halos under the ideal assumptions of single-scattering 
through an optically thin sight line (Corrales & Paerels 2015, Corrales et al. 2016).

Tutorials
---------

.. toctree::
   :maxdepth: 2

   tutorials/index

Modules
-------

.. toctree::
   :maxdepth: 2

   grainpop
   graindist
   scatmodels
   halos


.. Indices and tables
    ==================

    * :ref:`genindex`
    * :ref:`modindex`
    * :ref:`search`
