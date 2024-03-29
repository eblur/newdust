
import astropy.units as u
import astropy.constants as c

import numpy as np
import scipy as sp

# Allowed units for energy or wavelength arguments
ALLOWED_LAM_UNITS = ['kev','angs','keV','Angs','angstrom','Angstrom']

##----------------------------------------------------------
# Generic constants

# Speed of light
clight = c.c.to('cm/s').value

# Planck's h constant
hplanck = c.h.to('keV second').value

# Electron radius
r_e = 2.8179403227e-15 * u.m.to('cm')

# Mass of proton
m_p = c.m_p.to('g').value 

##----------------------------------------------------------
# Constants for converting things

micron2cm = 1.e-6 * 100.0  # cm/micron
pc2cm     = 3.09e18        # cm/parsec
angs2cm   = 1.e-8          # cm/angs

arcs2rad  = (2.0*np.pi) / (360.*60.*60.)  # rad/arcsec
arcm2rad  = (2.0*np.pi) / (360.*60.)      # rad/arcmin

hc        = (c.c * c.h).to('keV cm').value
hc_angs   = (c.c * c.h).to('keV angstrom').value

##----------------------------------------------------------
# Cosmology related constants

# Hubble's constant
h0 = 75.  # km/s/Mpc

# Critical density for Universe
rho_crit = np.float64(1.1e-29)

# Density in units of rho_crit
omega_d = 1.e-5  # dust
omega_m = 0.3    # matter
omega_l = 0.7    # dark energy

# c/H term in distance integral (a distance)
# c/H = Mpc, then convert to cm
cperh0  = (clight * 1.e-5 / h0) * (1.e6 * pc2cm)


##----- Very basic integration functions ------##
# I think I used these for checking something
# way back in the day

# Wrapper for scipy integration, used in cosmology integral
def intz(x, y):
    from scipy import integrate
    return sp.integrate.trapz(y,x)
# Note that scipy calls integration in reverse order as I do

# Basic trapezoidal integration function
def trapezoidal_int(x, y):
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    return np.sum(y[:-1]*dx + 0.5*dx*dy)

# Convert energy or wavelength (angs) to cm

def _lam_cm(lam, unit='kev'):
    assert unit in ALLOWED_LAM_UNITS
    if unit in ['kev','keV']:
        result  = hc / lam  # kev cm / kev
    if unit in ['angs','Angs','angstrom','Angstrom']:
        result  = angs2cm * lam  # cm/angs * angs
    return result  # cm

def _lam_kev(lam, unit='kev'):
    assert unit in ALLOWED_LAM_UNITS
    if unit in ['kev','keV']:
        result = lam
    if unit == ['angs','Angs','angstrom','Angstrom']:
        result = hc_angs / lam  # kev angs / angs
    return result

# Make sure that a scalar number is stored as an array
def _make_array(scalar):
    result = scalar
    if isinstance(scalar, list):
        result = np.array([scalar])
    else:
        try:
            len(scalar)
        except:
            result = np.array([scalar])
            assert len(result) == 1
    return result
