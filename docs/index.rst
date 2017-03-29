.. astrodust documentation master file, created by
   sphinx-quickstart on Wed Mar  8 14:57:12 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Home doc page for *eblur/dust*
==============================

The *eblur/dust* set of python modules calculate scattering absorption
and scattering efficiencies for dust from the infrared to the X-ray.
This code can also be used to calculate dust scattering halo images in
the X-ray, in both interstellar and intergalactic (cosmological)
contexts.

**First published version of this code** (released with `Corrales & Paerels, 2015 <http://adsabs.harvard.edu/abs/2015MNRAS.453.1121C>`_)
http://dx.doi.org/10.5281/zenodo.15991

**Source code:** github.com/eblur/dust

**Support:** If you are having issues, please contact lia@astro.wisc.edu

Features
--------

A number of dust grain size distributions and optical constants are
provided, but they can be fully customized by the user by invoking
custom objects of the approporiate class.  Provided dust models
include:

* Grain size distributions: single grain size, power law, and a power law with an exponential cut-off
* `Weingartner & Draine (2001) <http://adsabs.harvard.edu/abs/2001ApJ...548..296W>`_
  grain size distributions for Milky Way dust
* Optical constants (complex index of refraction) for 0.1 um sized
  `graphite and astrosilicate grains <https://www.astro.princeton.edu/~draine/dust/dust.diel.html>`_

* Rayleigh-Gans scattering physics

  * `Smith & Dwek (1998) <http://adsabs.harvard.edu/abs/1998ApJ...503..831S>`_
  * `Mauche & Gorenstein (1986) <http://adsabs.harvard.edu/abs/1986ApJ...302..371M>`_

* Mie scattering physics using the algorithms of
  `Bohren & Huffman (1986) <http://adsabs.harvard.edu/abs/1983asls.book.....B>`_

  * Converted from `fortran and IDL
    <http://www.met.tamu.edu/class/atmo689-lc/bhmie.pro>`_
    to python

Modules
-------
.. toctree::
   :maxdepth: 3

   graindist
   extinction

.. Indices and tables
    ==================

    * :ref:`genindex`
    * :ref:`modindex`
    * :ref:`search`
