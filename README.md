# newdust

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7500048.svg)](https://doi.org/10.5281/zenodo.7500048)

Major rewrite of [eblur/dust](https://github.com/eblur/dust) code. 

This package calculates extinction curves and small-angle scattering halos from a user-defined dust grain size distribution.
Calculates scattering and absorption from first principles (optical constants of the material, Mie or Rayleigh-Gans scattering).

## To install:

Clone the repo to your computer
```
git clone https://github.com/eblur/newdust.git newdust/
```

Enter the repo and run the `setup.py` script
```
cd newdust
python setup.py install
```

## To invoke:

```
import newdust
```

## How to use:

See the jupyter notebooks in **newdust/examples/** 
for examples of setting up grain populations and modeling scattering halos from Galactic dust.

**For simulating cosmological halos** 
(e.g. [Corrales & Paerels, 2012](http://adsabs.harvard.edu/abs/2012ApJ...751...93C) and 
[Corrales 2015](http://adsabs.harvard.edu/abs/2015ApJ...805...23C))
see the `cosmhalo` branch. Some of the cosmhalo tests do not pass. Use with caution.
