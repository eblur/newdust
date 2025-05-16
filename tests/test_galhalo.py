import pytest
import numpy as np
from scipy.integrate import trapezoid as trapz
import astropy.units as u

from newdust.halos import *
from newdust import grainpop
from . import percent_diff

# I tested a few values of NTH to reach the 5% integration threshold in some tests
NE, NTH = 10, 80
EVALS   = np.logspace(-1, 1, NE)   # keV
THVALS  = np.logspace(-1, 4, NTH)  # arcsec

GPOP    = grainpop.make_MRN_RGDrude()

E0, A0, TH0 = 1.0, 0.3, 10.0  # keV, um, arcsec
FABS = 1.0 * np.power(EVALS, -2.0) * np.exp(-0.1 * np.power(EVALS, -2.5))

# Unit Default Input with Astropy Unit
AMIN, AMAX, P, RHO = 0.005*u.micron, 0.5*u.micron, 3.5, 3*u.Unit('g cm^-3') # micron, micron, unitless, g cm^-3
MD = 1.e-6*u.Unit('g cm^-2') # g cm^-2

# Unitless Default Input for testing unit functionality
AMIN_UL, AMAX_UL, P_UL, RHO_UL = 0.005, 0.5, 3.5, 3 # micron, micron, unitless, g cm^-3
MD_UL = 1.e-6 # g cm^-2

# Default Input with Different Unit
AMIN_DU, AMAX_DU, P_DU, RHO_DU = AMIN.to(u.cm), AMAX.to(u.cm), P, RHO.to(u.Unit('kg m^-3')) # micron, micron, unitless, g cm^-3
MD_DU = MD.to(u.Unit('kg m^-2')) # g cm^-2

def test_Halo_dimensions():
    test = Halo(EVALS, THVALS)
    assert len(test.lam) == NE
    assert test.lam.unit == 'keV'
    assert len(test.theta) == NTH

# ---- Test Galactic Halo stuff ---- #

# Tests with default values
UNI_HALO = galhalo.UniformGalHalo(EVALS, THVALS)
SCR_HALO = galhalo.ScreenGalHalo(EVALS, THVALS)
UNI_HALO_CP15 = galhalo.UniformGalHaloCP15(EVALS, THVALS)
SCR_HALO_CP15 = galhalo.ScreenGalHaloCP15(EVALS,THVALS)
# Test for different unit
UNI_HALO_CP15_UL = galhalo.UniformGalHaloCP15(EVALS,THVALS)
SCR_HALO_CP15_UL = galhalo.ScreenGalHaloCP15(EVALS,THVALS)
UNI_HALO_CP15_DU = galhalo.UniformGalHaloCP15(EVALS,THVALS)
SCR_HALO_CP15_DU = galhalo.ScreenGalHaloCP15(EVALS,THVALS)

# Test that calculations run
def test_galhalo_uniform():
    UNI_HALO.calculate(GPOP)
    assert UNI_HALO.norm_int.unit == 'arcsec^-2'
    # Test CP15 version
    UNI_HALO_CP15.calculate(MD)
    assert UNI_HALO_CP15.norm_int.unit == 'arcsec^-2'
    UNI_HALO_CP15_UL.calculate(MD_UL,amin=AMIN_UL, amax=AMAX_UL, p=P_UL, rho = RHO_UL)
    assert UNI_HALO_CP15_UL.norm_int.unit == 'arcsec^-2'
    UNI_HALO_CP15_DU.calculate(MD_DU,amin=AMIN_DU, amax=AMAX_DU, p=P_DU, rho = RHO_DU)
    assert UNI_HALO_CP15_DU.norm_int.unit == 'arcsec^-2'
    # Check consistency of the results from different unit choices
    assert all(percent_diff(np.concatenate(UNI_HALO_CP15.norm_int.value), np.concatenate(UNI_HALO_CP15_UL.norm_int.value)) <= 0.05)
    assert all(percent_diff(np.concatenate(UNI_HALO_CP15.norm_int.value), np.concatenate(UNI_HALO_CP15_DU.norm_int.value)) <= 0.05)

@pytest.mark.parametrize('x', [1.0, 0.5])
def test_galhalo_screen(x):
    SCR_HALO.calculate(GPOP, x)
    assert SCR_HALO.norm_int.unit == 'arcsec^-2'
    
    # Observed angle should be equal to scattering angle when x = 1,
    # so halo should match differential scattering cross section integrated over dust grain size distributions
    if x == 1.0:
        # have to convert int_diff to arcsec^-2
        test = np.abs(SCR_HALO.norm_int.value - GPOP.int_diff.to('arcsec^-2').value)
        assert np.all(test < 0.01)
    # Test CP15 version
    SCR_HALO_CP15.calculate(MD, x=x)
    assert SCR_HALO_CP15.norm_int.unit == 'arcsec^-2'
    SCR_HALO_CP15_UL.calculate(MD_UL,amin=AMIN_UL, amax=AMAX_UL, p=P_UL, rho = RHO_UL, x=x)
    assert SCR_HALO_CP15_UL.norm_int.unit == 'arcsec^-2'
    SCR_HALO_CP15_DU.calculate(MD_DU,amin=AMIN_DU, amax=AMAX_DU, p=P_DU, rho = RHO_DU, x=x)
    assert SCR_HALO_CP15_DU.norm_int.unit == 'arcsec^-2'
    # Check consistency of the results from different unit choices
    assert all(percent_diff(np.concatenate(SCR_HALO_CP15.norm_int.value), np.concatenate(SCR_HALO_CP15_UL.norm_int.value)) <= 0.05)
    assert all(percent_diff(np.concatenate(SCR_HALO_CP15.norm_int.value), np.concatenate(SCR_HALO_CP15_DU.norm_int.value)) <= 0.05)

@pytest.mark.parametrize('test', [UNI_HALO, SCR_HALO, UNI_HALO_CP15, SCR_HALO_CP15])
def test_halos_general(test):
    # Test basic shape properties of outputs
    assert np.shape(test.norm_int) == (NE, NTH)
    assert np.shape(test.taux) == (NE,)

    # Test that the scattering halos integrate to total scattering optical depth
    alph_grid = np.repeat(THVALS.reshape(1, NTH), NE, axis=0)
    int_halo  = trapz(test.norm_int.value * 2.0 * np.pi * alph_grid, THVALS, axis=1)
    assert all(percent_diff(int_halo, test.taux) <= 0.05)

    #--- Test that the flux and intensity functions work
    # Check that calculation runs
    test.calculate_intensity(FABS)
    assert test.intensity is not None
    assert np.size(test.fabs) == NE
    assert np.size(test.fext) == NE
    assert np.size(test.fhalo) == NE
    assert test.intensity.unit == 'arcsec^-2'
    # Integrated intensity profile should add up to total halo flux
    # Calculation uses optically thin approximation, so it adds up to FABS * tau_sca
    tot_halo = trapz(test.intensity * 2.0 * np.pi * THVALS * u.arcsec, THVALS * u.arcsec)
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

@pytest.mark.parametrize('test', [UNI_HALO, SCR_HALO, UNI_HALO_CP15, SCR_HALO_CP15])
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

@pytest.mark.parametrize('test', [UNI_HALO, SCR_HALO, UNI_HALO_CP15, SCR_HALO_CP15])
def test_halo_index(test):
    i = 0
    tt = test[0]
    assert isinstance(tt, Halo)
    assert np.size(tt.lam) == 1
    assert tt.lam == test.lam[i]
    assert tt.taux == test.taux[i]
    assert np.shape(tt.norm_int) == (1, NTH)
    assert np.shape(tt.intensity) == (NTH,)

@pytest.mark.parametrize('test', [UNI_HALO, SCR_HALO, UNI_HALO_CP15, SCR_HALO_CP15])
def test_halo_io(test):
    test.write('test_halo_io.fits')
    new_halo = Halo(from_file='test_halo_io.fits')
    assert np.all(percent_diff(new_halo.lam,test.lam) < 0.01)
    assert np.all(percent_diff(new_halo.theta,test.theta) < 0.01)
    assert np.all(percent_diff(new_halo.taux,test.taux) < 0.01)
    assert np.all(percent_diff(new_halo.norm_int.flatten(),test.norm_int.flatten()) < 0.01)
    assert new_halo.lam.unit == 'keV'
