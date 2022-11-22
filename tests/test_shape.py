import numpy as np

import astropy.units as u
from newdust.graindist import shape
from . import percent_diff

AVALS = np.linspace(0.01, 1.0, 100) * u.micron  # um

def test_sphere():
    test = shape.Sphere()
    assert test.shape == 'Sphere'
    tvol = np.sum((4.0/3.0) * np.pi * (AVALS.to('cm').value)**3)
    tgeo = np.sum(np.pi * (AVALS.to('cm').value)**2)
    assert percent_diff(tvol, np.sum(test.vol(AVALS))) <= 0.01
    assert percent_diff(tgeo, np.sum(test.cgeo(AVALS))) <= 0.01
    # Test that it will accept arrays with no units
    assert percent_diff(tvol, np.sum(test.vol(AVALS.value))) <= 0.01
    assert percent_diff(tgeo, np.sum(test.cgeo(AVALS.value))) <= 0.01
