import pytest
import numpy as np
import astropy.units as u
from newdust.graindist import composition

WAVEL = np.logspace(0, 2.5, 100) * u.angstrom
ENERGY = np.logspace(-1, 1, 100) * u.keV
EN = np.linspace(0.1, 10.0, 50) # no units specified, assume keV
RHO_TEST = 2.0  # g cm^-3

CMS = [composition.CmDrude(), composition.CmSilicate(), composition.CmGraphite()]

# Test that every method runs
@pytest.mark.parametrize('cm', CMS)
def test_abstract_class(cm):
    assert type(cm.cmtype) is str
    assert type(cm.rho) is float
    assert len(cm.rp(WAVEL)) == len(WAVEL)
    assert len(cm.rp(ENERGY)) == len(ENERGY)
    assert len(cm.rp(EN)) == len(EN)
    assert len(cm.ip(WAVEL)) == len(WAVEL)
    assert len(cm.ip(ENERGY)) == len(ENERGY)
    assert len(cm.ip(EN)) == len(EN)
    assert len(cm.cm(WAVEL)) == len(WAVEL)
    assert len(cm.cm(ENERGY)) == len(ENERGY)
    assert len(cm.cm(EN)) == len(EN)
    assert type(cm.cm(1.0)) is np.complex128

# Test that it returns 0 (im part) or 1 (re part) when interpolating
# outside the grid. Only relevant for table models.
@pytest.mark.parametrize('cm', [composition.CmSilicate(), composition.CmGraphite()])
def test_limits(cm):
    # Make the lower and upper test values -50% and +50% on either side
    lam_low = 0.5 * np.min(cm.wavel.value) * cm.wavel.unit
    lam_high = 1.5 * np.max(cm.wavel.value) * cm.wavel.unit
    assert cm.rp(lam_low) == 1.0
    assert cm.rp(lam_high) == 1.0
    assert cm.ip(lam_low) == 0.0
    assert cm.ip(lam_high) == 0.0

@pytest.mark.parametrize(('sizes','orients'),
                         [('big','para'),
                          ('big','perp'),
                          ('small','para'),
                          ('small','perp')])
def test_cmgraphite(sizes, orients):
    test = composition.CmGraphite(rho=2.0, size=sizes, orient=orients)
    assert test.rho == RHO_TEST
    test_abstract_class(test)
