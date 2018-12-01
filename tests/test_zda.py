import pytest
import numpy as np
from scipy.integrate import trapz

from newdust.zda import logg
from . import percent_diff

MTYPES = ['BARE-GR-S']

AVALS = np.logspace(-4, 0.0, 100)

@pytest.mark.parametrize('model', MTYPES)
def test_logg(model):
    for k in model.keys():
        result = logg(AVALS, model, k)
        assert len(result) == len(AVALS)
        test_integral = trapz(result, AVALS)
        assert percent_diff(test_integral, 1.0) < 0.01
