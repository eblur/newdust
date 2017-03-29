import pytest
import numpy as np
from newdust.graindist import composition

LAMBDA = (np.logspace(0, 2.5, 100), 'angs')  # angs
ENERGY = (np.logspace(-1, 1, 100), 'kev')    # kev
RHO_TEST = 2.0  # g cm^-3

@pytest.mark.parametrize('cm',
                         [composition.CmDrude(),
                          composition.CmSilicate(),
                          composition.CmGraphite()])
def test_abstract_class(cm):
    assert type(cm.cmtype) is str
    assert type(cm.rho) is float
    assert len(cm.rp(LAMBDA[0], unit=LAMBDA[1])) == len(LAMBDA[0])
    assert len(cm.rp(ENERGY[0], unit=ENERGY[1])) == len(ENERGY[0])
    assert len(cm.ip(LAMBDA[0], unit=LAMBDA[1])) == len(LAMBDA[0])
    assert len(cm.ip(ENERGY[0], unit=ENERGY[1])) == len(ENERGY[0])
    assert len(cm.cm(LAMBDA[0], unit=LAMBDA[1])) == len(LAMBDA[0])
    assert len(cm.cm(ENERGY[0], unit=ENERGY[1])) == len(ENERGY[0])
    assert type(cm.cm(1.0)) is np.complex128


@pytest.mark.parametrize(('sizes','orients'),
                         [('big','para'),
                          ('big','perp'),
                          ('small','para'),
                          ('small','perp')])
def test_cmgraphite(sizes, orients):
    test = composition.CmGraphite(rho=2.0, size=sizes, orient=orients)
    assert test.rho == RHO_TEST
    test_abstract_class(test)

# Test that super high energy values go to rp = 1.0, ip = 0.0
ELO, EHI = 1.e-10, 1.e10  # keV
@pytest.mark.parametrize('cm',
                         [composition.CmDrude(),
                          composition.CmSilicate(),
                          composition.CmGraphite()])
def test_Elims(cm):
    assert cm.rp(EHI, unit='kev') == 1.0
    assert cm.rp(ELO, unit='kev') == 1.0
    assert cm.ip(EHI, unit='kev') == 0.0
    assert cm.ip(ELO, unit='kev') == 0.0
