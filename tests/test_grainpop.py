import pytest
import numpy as np
from scipy.integrate import trapz
import astropy.units as u

from newdust.grainpop import *
from newdust import graindist
from newdust import scatteringmodel
from . import percent_diff

MD = 1.e-5  # g cm^-2
RHO = 3.0   # g c^-3

ALLOWED_SCATM = ['RG','Mie']

MRN_SIL = graindist.GrainDist('Powerlaw','Silicate')
MRN_DRU = graindist.GrainDist('Powerlaw','Drude')
EXP_SIL = graindist.GrainDist('ExpCutoff','Silicate')
GRAIN   = graindist.GrainDist('Grain','Drude')

## NOTE: To test integrals over theta, need a large enough number of points on the grid
## I started with 100 and multiplied by 2 until the test_ext_calculation tests pass
NE, NA, NTH = 5, np.size(graindist.sizedist.Powerlaw().a), 400
THETA    = np.logspace(-6, np.log10(np.pi), NTH)  # 0->pi scattering angles (rad)
TH_ASEC  = (THETA * u.radian).to('arcsec')
LAMVALS  = np.linspace(1000., 5000., NE)  # angs
EVALS    = np.logspace(-1, np.log10(3.0), NE)  # kev

# test that everything runs on both kev and angs
test1 = SingleGrainPop('Powerlaw', 'Silicate', 'Mie')
test1.calculate_ext(LAMVALS * u.angstrom, theta=THETA)

test2 = SingleGrainPop('Powerlaw', 'Silicate', 'RG')
test2.calculate_ext(EVALS, theta=THETA)

def test_SingleGrainPop():
    assert len(test1.a) == NA
    assert len(test1.ndens) == NA
    assert len(test1.mdens) == NA
    assert len(test1.cgeo) == NA
    assert len(test1.vol) == NA
    assert test1.lam.unit.to_string() == 'Angstrom'
    assert test2.lam.unit.to_string() == 'keV' # test the automatic units
    assert len(test1.lam) == NE
    assert len(test1.tau_ext) == NE
    assert len(test1.tau_abs) == NE
    assert len(test1.tau_sca) == NE

def test_GrainPop_keys():
    test = GrainPop([test1, test2])
    assert test[0].description == '0'
    assert test[1].description == '1'

    test = GrainPop([test1, test2], keys=['foo','bar'])
    assert test['foo'].description == 'foo'
    assert test['bar'].description == 'bar'

def test_GrainPop_calculation():
    test = GrainPop([SingleGrainPop('Powerlaw', 'Silicate', 'Mie'),
                     SingleGrainPop('Powerlaw', 'Silicate', 'RG')])
    test.calculate_ext(LAMVALS * u.angstrom, theta=0.0)
    assert np.shape(test.tau_ext) == (NE,)
    assert np.shape(test.tau_sca) == (NE,)
    assert np.shape(test.tau_abs) == (NE,)
    assert np.all(percent_diff(test.tau_ext, test.tau_sca + test.tau_abs) <= 0.01)

@pytest.mark.parametrize('fsil', [0.0, 0.4, 1.0])
def test_make_MRN(fsil):
    test3 = make_MRN(fsil=fsil, md=MD)
    test3.calculate_ext(LAMVALS * u.angstrom, theta=0.0)
    assert isinstance(test3, GrainPop)
    assert percent_diff(test3.md, MD) <= 0.1
    if fsil == 0.0:
        assert test3['sil'].md == 0.0
    if fsil == 1.0:
        assert test3['sil'].md == MD
    # Test that doubling the mass doubles the extinction
    test4 = make_MRN(fsil=fsil, md=2.0*MD)
    test4.calculate_ext(LAMVALS * u.angstrom, theta=0.0)
    assert np.all(percent_diff(test4.tau_ext, 2.0*test3.tau_ext) <= 0.01)


def test_make_MRN_RGDrude():
    test3 = make_MRN_RGDrude(md=MD)
    test3.calculate_ext(LAMVALS * u.angstrom, theta=0.0)
    assert isinstance(test3, SingleGrainPop)
    assert percent_diff(test3.md, MD) <= 0.1
    # Test that doubling the mass doubles the extinction
    test4 = make_MRN_RGDrude(md=2.0*MD)
    test4.calculate_ext(LAMVALS * u.angstrom, theta=0.0)
    assert np.all(percent_diff(test4.tau_ext, 2.0*test3.tau_ext) <= 0.01)


#-------- Test the extinction calculations



# Test that all the computations can run
# Test RG-Drude with X-ray photons
# Test everything else with optical photons
@pytest.mark.parametrize('sd', ['Powerlaw', 'ExpCutoff','Grain'])
@pytest.mark.parametrize('cm', ['Silicate','Graphite', 'Drude'])
@pytest.mark.parametrize('sc', ['Mie', 'RG'])
def test_ext_calculations(sd, cm, sc):
    thresh = 0.05 # default threshold for agreement
    # If Rayleigh-Gans, use X-ray photons
    # NOTE - Can only get Rayleigh-Gans to work within 5%, see test_scatmodels.py
    if sc == 'RG':
        test = SingleGrainPop(sd, cm, sc)
        test.calculate_ext(EVALS, theta=TH_ASEC)
        thresh = 0.10 # Change threshold to 10% if RGscattering
    # If not Rayleigh-Gans but Drude appox is selected, skip test
    elif cm == 'Drude':
        return
    # For everything else (Mie scattering)
    else:
        test = SingleGrainPop(sd, cm, sc)
        test.calculate_ext(LAMVALS * u.angstrom, theta=TH_ASEC)

    
    assert np.shape(test.tau_ext) == (NE,)
    assert np.shape(test.tau_sca) == (NE,)
    assert np.shape(test.tau_abs) == (NE,)
    assert np.shape(test.diff) == (NE, len(test.a), NTH)
    assert np.all(percent_diff(test.tau_ext, test.tau_abs + test.tau_sca) <= 0.01)
    test.info()

    # Test that the integrated differential cross-sections match the scattering cross sections
    th_2d   = np.repeat(THETA.reshape(1, 1, NTH), NE, axis=0) # NE x 1 x NA
    th_3d   = np.repeat(th_2d.reshape(NE, 1, NTH), len(test.a.value), axis=1) # NE x NA x NTH
    integrated = trapz(test.diff * 2.0 * np.pi * np.sin(th_3d), THETA, axis=2) # NE x NA, [cm^2]
    if sd == 'Grain':
        sigma_scat = test.scatm.qsca * test.cgeo
    else:
        sigma_scat = test.scatm.qsca * np.repeat(test.cgeo.reshape(1, NA), NE, axis=0) # NE x NA [cm^2]
    result = percent_diff(integrated.flatten(), sigma_scat.flatten())
    #print(result)
    print(result[result > thresh])
    assert np.all(result <= thresh)

    # Test write and read functions
    test.write_extinction_table('test_grainpop.fits')
    new_test = SingleGrainPop(sd, cm, sc, scatm_from_file='test_grainpop.fits')
    assert np.all(percent_diff(test.tau_ext, new_test.tau_ext) <= 1.e-5)
    assert np.all(percent_diff(test.tau_abs, new_test.tau_abs) <= 1.e-5)
    assert np.all(percent_diff(test.tau_sca, new_test.tau_sca) <= 1.e-5)
    assert np.all(percent_diff(test.diff.flatten(), new_test.diff.flatten()) <= 1.e-5)
    assert np.all(percent_diff(test.int_diff.flatten(), new_test.int_diff.flatten()) <= 1.e-5) 
    
# Make sure that doubling the dust mass doubles the extinction
@pytest.mark.parametrize('estring', ALLOWED_SCATM)
def test_mass_double(estring):
    gp1 = SingleGrainPop('Powerlaw','Silicate', estring)
    gp1.calculate_ext(LAMVALS * u.angstrom)
    gp2 = SingleGrainPop('Powerlaw','Silicate', estring, md=2.0*gp1.md)
    gp2.calculate_ext(LAMVALS * u.angstrom)

    assert all(percent_diff(gp2.tau_ext, 2.0 * gp1.tau_ext) <= 0.01)
    assert all(percent_diff(gp2.tau_abs, 2.0 * gp1.tau_abs) <= 0.01)
    assert all(percent_diff(gp2.tau_sca, 2.0 * gp1.tau_sca) <= 0.01)

##---------- Test that we can customize the grain populations easily
def test_custom_SingleGrainPop():
    sdist = graindist.sizedist.Powerlaw()
    compo = graindist.composition.CmSilicate(rho=3.0)
    mscat = scatteringmodel.Mie()
    test  = SingleGrainPop(sdist, compo, mscat)
    test  = SingleGrainPop(sdist, compo, 'RG')
    test  = SingleGrainPop('Powerlaw', compo, mscat)
    test  = SingleGrainPop(sdist, 'Silicate', mscat)
