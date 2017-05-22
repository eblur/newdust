import numpy as np

from newdust import constants as c
from newdust.graindist import shape
from . import percent_diff

AVALS = np.linspace(0.01, 1.0, 100)  # um

def test_sphere():
    test = shape.Sphere()
    assert test.shape == 'Sphere'
    tvol = np.sum((4.0/3.0) * np.pi * (AVALS * c.micron2cm)**3)
    tgeo = np.sum(np.pi * (AVALS * c.micron2cm)**2)
    assert percent_diff(tvol, np.sum(test.vol(AVALS))) <= 0.01
    assert percent_diff(tgeo, np.sum(test.cgeo(AVALS))) <= 0.01
