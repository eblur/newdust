import pytest
import numpy as np
from scipy.integrate import trapz

from newdust.graindist import *
from . import percent_diff

MD  = 1.e-5  # g cm^-2
RHO = 3.0    # g cm^-3

SDEFAULT = 'Powerlaw'
CDEFAULT = 'Silicate'

ALLOWED_SIZES = ['Grain','Powerlaw','ExpCutoff']
ALLOWED_COMPS = ['Drude','Silicate','Graphite']

# Test that the helper function runs on all types
@pytest.mark.parametrize('sstring', ALLOWED_SIZES)
def test_sstring(sstring):
    test = GrainDist(sstring, CDEFAULT)
    assert isinstance(test, GrainDist)

@pytest.mark.parametrize('cstring', ALLOWED_COMPS)
def test_cstring(cstring):
    test = GrainDist(SDEFAULT, cstring)
    assert isinstance(test, GrainDist)

# Test that the helper function does not run on weird strings
def test_catch_exception():
    ss, cc = 'foo', 'bar'
    with pytest.raises(AssertionError):
        GrainDist(ss, CDEFAULT)
        GrainDist(SDEFAULT, cc)

# Test the basic properties and functions of GrainDist
@pytest.mark.parametrize('sstring', ALLOWED_SIZES)
def test_GrainDist(sstring):
    test = GrainDist(sstring, CDEFAULT, md=MD)
    assert len(test.a) == len(test.ndens)
    assert len(test.a) == len(test.mdens)
    if isinstance(test.size, sizedist.Grain):
        mtot = test.mdens
    else:
        mtot = trapz(test.mdens, test.a.to('micron').value)
    assert percent_diff(mtot, MD) <= 0.01

# Test that doubling the dust mass column doubles the total mass
MD2 = 2.0 * MD
def test_dmass():
    for ss in ALLOWED_SIZES:
        for cc in ALLOWED_COMPS:
            test1 = GrainDist(ss, cc, md=MD)
            test2 = GrainDist(ss, cc, md=MD2)
            if isinstance(test1.size, sizedist.Grain):
                mtot1, mtot2 = test1.mdens, test2.mdens
            else:
                mtot1 = trapz(test1.mdens, test1.a.to('micron').value)
                mtot2 = trapz(test2.mdens, test2.a.to('micron').value)
            assert percent_diff(mtot2, 2.0 * mtot1) <= 0.01

# Test that doubling the dust grain material density halves the total number
RHO2 = 2.0 * RHO
def test_ndens():
    for ss in ALLOWED_SIZES:
        for cc in ALLOWED_COMPS:
            test1 = GrainDist(ss, cc, md=MD, rho=RHO)
            test2 = GrainDist(ss, cc, md=MD, rho=RHO2)
            if isinstance(test1.size, sizedist.Grain):
                nd1, nd2 = test1.ndens, test2.ndens
            else:
                nd1 = trapz(test1.ndens, test1.a.to('micron').value)
                nd2 = trapz(test2.ndens, test2.a.to('micron').value)
            assert percent_diff(nd2, 0.5 * nd1) <= 0.01
