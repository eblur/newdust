import pytest
import numpy as np
from scipy.integrate import trapz

from newdust.grainpop import *
from newdust.graindist import *
from newdust.extinction import *
from . import percent_diff

MD = 1.e-5  # g cm^-2
RHO = 3.0   # g c^-3

TEST_SDIST = make_GrainDist('Powerlaw','Silicate', md=MD, rho=RHO)
MIE  = Extinction('Mie')
RG   = Extinction('RG')

NE, NA, NTH = 50, np.size(TEST_SDIST.a), 30
THETA   = np.logspace(0, 4, NTH)
LAMVALS = np.linspace(1000., 5000., NE)  # angs
EVALS   = np.logspace(-1, 1, NE)  # kev

# test that everything runs on both kev and angs
test1 = SingleGrainPop(TEST_SDIST, MIE)
test1.calculate_ext(LAMVALS, unit='angs', theta=THETA)

test2 = SingleGrainPop(TEST_SDIST, RG)
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
    test = GrainPop([SingleGrainPop(TEST_SDIST, MIE), SingleGrainPop(TEST_SDIST, RG)])
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
