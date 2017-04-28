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

def test_galhalo_uniform():
    test = Halo(EVALS, THVALS, unit='kev')

    # Test that the function runs
    galhalo.uniformISM(test, GPOP)
    assert isinstance(test.htype, galhalo.UniformGalHalo)
    assert np.shape(test.norm_int) == (NE, NTH)
    assert np.shape(test.taux) == (NE,)

    # Test that the scattering halos integrate to total scattering optical depth
    alph_grid = np.repeat(THVALS.reshape(1, NTH), NE, axis=0)
    int_halo  = trapz(test.norm_int * 2.0 * np.pi * alph_grid, THVALS, axis=1)
    assert all(percent_diff(int_halo, test.taux) <= 0.05)

    #--- Test that the flux and intensity functions work
    # Check that calculation runs
    test.calculate_intensity(FABS)
    assert test.intensity is not None
    assert np.size(test.fabs) == NE
    assert np.size(test.fext) == NE
    assert np.size(test.fhalo) == NE
    # Integrated intensity profile should add up to total halo flux
    # Calculation uses optically thin approximation, so it adds up to FABS * tau_sca
    tot_halo = trapz(test.intensity * 2.0 * np.pi * THVALS, THVALS)
    assert percent_diff(tot_halo, np.sum(test.fabs * test.taux)) <= 0.05

    # The fhalo function calculations halo flux without optically thin approximation
    # So fext = fabs - fhalo
    assert all(percent_diff(test.fext, test.fabs - test.fhalo) <= 0.01)

    # Test that distinction between extincted flux and absorbed flux is handled properly
    FEXT = FABS * np.exp(-test.taux)
    assert all(percent_diff(test.fext, FEXT) <= 0.01)
    fh1  = test.fhalo
    pfa1 = test.percent_fabs
    pfe1 = test.percent_fext
    test.calculate_intensity(FEXT, ftype='ext')
    assert all(percent_diff(test.fext, FEXT) <= 0.01)
    assert all(percent_diff(fh1, test.fhalo) <= 0.01)
    assert percent_diff(pfa1, test.percent_fabs) <= 0.01
    assert percent_diff(pfe1, test.percent_fext) <= 0.01
