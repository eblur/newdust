import pytest
from scipy.integrate import trapz

from newdust.graindist import sizedist
from . import percent_diff

MDTEST  = 1.e-4  # g cm^-2
RHOTEST = 3.0    # g cm^-3

def test_Grain():
    test = sizedist.Grain()
    nd   = test.ndens(MDTEST, RHOTEST)
    assert len(test.a) == 1
    assert len(nd)     == 1
    tot_mass = test.mdens(MDTEST, RHOTEST)
    assert percent_diff(tot_mass, MDTEST) <= 0.01

def test_Powerlaw():
    test = sizedist.Powerlaw()
    nd   = test.ndens(MDTEST, RHOTEST)
    assert len(test.a) == len(nd)
    md   = test.mdens(MDTEST, RHOTEST)
    tot_mass = trapz(md, test.a)
    assert percent_diff(tot_mass, MDTEST) <= 0.01

def test_ExpCutoff():
    test = sizedist.ExpCutoff()
    nd   = test.ndens(MDTEST, RHOTEST)
    assert len(test.a) == len(nd)
    md   = test.mdens(MDTEST, RHOTEST)
    tot_mass = trapz(md, test.a)
    assert percent_diff(tot_mass, MDTEST) <= 0.01

# Test that doubling the dust mass column doubles the total mass
MDTEST2 = 2.0 * MDTEST

@pytest.mark.parametrize('sd',
                         [sizedist.Grain(),
                          sizedist.Powerlaw(),
                          sizedist.ExpCutoff()])
def test_dmass(sd):
    md1 = sd.mdens(MDTEST, RHOTEST)
    md2 = sd.mdens(MDTEST2, RHOTEST)
    if isinstance(sd, sizedist.Grain):
        mtot1, mtot2 = md1, md2
    else:
        mtot1 = trapz(md1, sd.a)
        mtot2 = trapz(md2, sd.a)
    assert percent_diff(mtot2, 2.0 * mtot1) <= 0.01

# Test that doubling the dust material density halves the number density
RHOTEST2 = 2.0 * RHOTEST

@pytest.mark.parametrize('sd',
                         [sizedist.Grain(),
                          sizedist.Powerlaw(),
                          sizedist.ExpCutoff()])
def test_change_rho(sd):
    nd1 = sd.ndens(MDTEST, RHOTEST)
    nd2 = sd.ndens(MDTEST, RHOTEST2)
    if isinstance(sd, sizedist.Grain):
        ntot1, ntot2 = nd1, nd2
    else:
        ntot1 = trapz(nd1, sd.a)
        ntot2 = trapz(nd2, sd.a)
    assert percent_diff(ntot2, 0.5 * ntot1) <= 0.01
