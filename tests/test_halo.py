import numpy as np
from scipy.integrate import trapz

from newdust.halos import *
from newdust import grainpop

NE, NTH = 10, 200
EVALS   = np.logspace(-1, 1, NE)   # keV
THVALS  = np.logspace(-1, 4, NTH)  # arcsec
GPOP    = grainpop.make_MRN_drude()['RGD']
E0, A0, TH0 = 1.0, 0.3, 10.0  # keV, um, arcsec

def test_Halo_dimensions():
    test = Halo(EVALS, THVALS, unit='kev')
    assert len(test.lam) == NE
    assert test.lam_unit == 'kev'
    assert len(test.theta) == NTH

# ---- Test Galactic Halo stuff ---- #

def test_galhalo_uniform():
    test = Halo(EVALS, THVALS, unit='kev')
    galhalo.uniformISM(test, GPOP)
    assert np.shape(test.norm_int) == (NE, NTH)
