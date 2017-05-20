import pytest
import numpy as np
from scipy.integrate import trapz

from newdust.grainpop import *
from newdust.graindist import *
from newdust.scatmodels import *
from . import percent_diff

MD = 1.e-5  # g cm^-2
RHO = 3.0   # g c^-3

ALLOWED_SCATM = ['RG','Mie']
TEST_SDIST = make_GrainDist('Powerlaw','Silicate', md=MD, rho=RHO)

MRN_SIL = graindist.make_GrainDist('Powerlaw','Silicate')
MRN_DRU = graindist.make_GrainDist('Powerlaw','Drude')
EXP_SIL = graindist.make_GrainDist('ExpCutoff','Silicate')
GRAIN   = graindist.make_GrainDist('Grain','Drude')

NE, NA, NTH = 50, np.size(TEST_SDIST.a), 30
THETA    = np.logspace(-10.0, np.log10(np.pi), NTH)  # 0->pi scattering angles (rad)
ASEC2RAD = (2.0 * np.pi) / (360.0 * 60. * 60.)     # rad / arcsec
TH_asec  = THETA / ASEC2RAD  # rad * (arcsec/rad)
LAMVALS  = np.linspace(1000., 5000., NE)  # angs
EVALS    = np.logspace(-1, 1, NE)  # kev

# test that everything runs on both kev and angs
test1 = SingleGrainPop(TEST_SDIST, 'Mie')
test1.calculate_ext(LAMVALS, unit='angs', theta=THETA)

test2 = SingleGrainPop(TEST_SDIST, 'RG')
test2.calculate_ext(EVALS, unit='kev', theta=THETA)

def test_SingleGrainPop():
    assert len(test1.a) == NA
    assert len(test1.ndens) == NA
    assert len(test1.mdens) == NA
    assert len(test1.cgeo) == NA
    assert len(test1.vol) == NA
    assert test1.lam_unit == 'angs'
    assert test2.lam_unit == 'kev'
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
    test = GrainPop([SingleGrainPop(TEST_SDIST, 'Mie'), SingleGrainPop(TEST_SDIST, 'RG')])
    test.calculate_ext(LAMVALS, unit='angs', theta=THETA)
    assert np.shape(test.tau_ext) == (NE,)
    assert np.shape(test.tau_sca) == (NE,)
    assert np.shape(test.tau_abs) == (NE,)
    assert any(percent_diff(test.tau_ext, test.tau_sca + test.tau_abs) <= 0.01)

@pytest.mark.parametrize('fsil', [0.0, 0.4, 1.0])
def test_make_MRN(fsil):
    test3 = make_MRN(fsil=fsil, md=MD)
    assert isinstance(test3, GrainPop)
    assert percent_diff(test3.md, MD) <= 0.1
    if fsil == 0.0:
        assert test3['sil'].gdist.md == 0.0
    if fsil == 1.0:
        assert test3['sil'].gdist.md == MD
    # Test that doubling the mass doubles the extinction
    test4 = make_MRN(fsil=fsil, md=2.0*MD)
    assert percent_diff(test4.tau_ext, 2.0*test3.tau_ext) <= 0.01

def test_make_MRN_drude():
    test3 = make_MRN_drude(md=MD)
    assert isinstance(test3, GrainPop)
    assert percent_diff(test3.md, MD) <= 0.1
    # Test that doubling the mass doubles the extinction
    test4 = make_MRN_drude(md=2.0*MD)
    assert percent_diff(test4.tau_ext, 2.0*test3.tau_ext) <= 0.01

#-------- Test the extinction calculations

# Test that all the computations can run
@pytest.mark.parametrize(('gd','sm'),
                         [(MRN_SIL, 'RG'), (MRN_SIL, 'Mie'),
                          (MRN_DRU, 'RG'), (MRN_DRU, 'Mie'),
                          (EXP_SIL, 'RG'), (EXP_SIL, 'Mie'),
                          (GRAIN, 'RG'), (GRAIN, 'Mie')])
def test_ext_calculations(gd, sm):
    test = SingleGrainPop(gd, sm)
    test.calculate_ext(LAMVALS, unit='angs', theta=TH_asec)
    assert np.shape(test.tau_ext) == (NE,)
    assert np.shape(test.tau_sca) == (NE,)
    assert np.shape(test.tau_abs) == (NE,)
    assert np.shape(test.diff) == (NE, len(gd.a), NTH)
    assert all(percent_diff(test.tau_ext, test.tau_abs + test.tau_sca) <= 0.01)

# Make sure that doubling the dust mass doubles the extinction
@pytest.mark.parametrize('estring', ALLOWED_SCATM)
def test_mass_double(estring):
    gd1 = graindist.make_GrainDist('Powerlaw','Silicate')
    gd2 = graindist.make_GrainDist('Powerlaw', 'Silicate')
    gd2.md = 2.0 * gd1.md

    gp1 = SingleGrainPop(gd1, estring)
    gp1.calculate_ext(LAMVALS, unit='angs')
    gp2 = SingleGrainPop(gd2, estring)
    gp2.calculate_ext(LAMVALS, unit='angs')

    assert all(percent_diff(gp2.tau_ext, 2.0 * gp1.tau_ext) <= 0.01)
    assert all(percent_diff(gp2.tau_abs, 2.0 * gp1.tau_abs) <= 0.01)
    assert all(percent_diff(gp2.tau_sca, 2.0 * gp1.tau_sca) <= 0.01)
