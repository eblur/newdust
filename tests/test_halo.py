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
def test_galhalo_uniform():
    galhalo.uniformISM(UNI_HALO, GPOP)
    assert isinstance(UNI_HALO.htype, galhalo.UniformGalHalo)

@pytest.mark.parametrize('x', [1.0, 0.5])
def test_galhalo_screen(x):
    galhalo.screenISM(SCR_HALO, GPOP, x=x)
    assert isinstance(SCR_HALO.htype, galhalo.ScreenGalHalo)
    # Observed angle should be equal to scattering angle when x = 1,
    # so halo should match differential scattering cross section integrated over dust grain size distributions
    if x == 1.0:
        test = np.abs(SCR_HALO.norm_int - GPOP.ext.int_diff)
        assert np.all(test < 0.01)

@pytest.mark.parametrize('test', [UNI_HALO, SCR_HALO])
def test_halos_general(test):
    # Test basic shape properties of outputs
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

@pytest.mark.parametrize('test', [UNI_HALO, SCR_HALO])
def test_halo_slices(test):
    # Test the halo slice functions
    # Split the energy values in half
    EMID = (EVALS[-1] - EVALS[0]) / 2.0
    i1 = (EVALS < EMID)
    i2 = (EVALS >= EMID)
    N1, N2 = np.size(EVALS[i1]), np.size(EVALS[i2])
    h1 = test[0:EMID]
    h2 = test[EMID:]
    # Test that the slices return Halo objects
    assert isinstance(h1, Halo)
    assert isinstance(h2, Halo)
    # Test the sizes of things
    assert np.size(h1.lam) == N1
    assert np.size(h2.lam) == N2
    assert np.shape(h1.norm_int) == (N1, NTH)
    assert np.shape(h2.norm_int) == (N2, NTH)
    # Test the values of the halo slices
    assert np.all(percent_diff(np.append(h1.taux,h2.taux), test.taux) == 0.0)
    assert np.all(percent_diff(np.append(h1.lam,h2.lam), test.lam) == 0.0)
