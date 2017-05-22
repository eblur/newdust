import pytest
import numpy as np
from scipy.integrate import trapz

from newdust.halos import *
from newdust import grainpop
from . import percent_diff

NE, NTH = 10, 200
EVALS   = np.logspace(-1, 1, NE)   # keV
THVALS  = np.logspace(-1, 4, NTH)  # arcsec

GPOP    = grainpop.make_MRN_drude()['RGD']

E0, A0, TH0 = 1.0, 0.3, 10.0  # keV, um, arcsec
FABS = 1.0 * np.power(EVALS, -2.0) * np.exp(-0.1 * np.power(EVALS, -2.5))

def test_Halo_dimensions():
    test = Halo(EVALS, THVALS, unit='kev')
    assert len(test.lam) == NE
    assert test.lam_unit == 'kev'
    assert len(test.theta) == NTH

# ---- Test Galactic Halo stuff ---- #

UNI_HALO = Halo(EVALS, THVALS, unit='kev')
SCR_HALO = Halo(EVALS, THVALS, unit='kev')

# Test that calculations run
def test_cosmhalo_uniform():
    cosmhalo.uniformIGM(UNI_HALO, GPOP)
    assert isinstance(UNI_HALO.htype, cosmhalo.CosmHalo)

@pytest.mark.parametrize('x', [1.0, 0.5])
def test_cosmhalo_screen(x):
    cosmhalo.screenIGM(SCR_HALO, GPOP, x=x)
    assert isinstance(SCR_HALO.htype, cosmhalo.CosmHalo)
