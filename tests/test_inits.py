import pytest
from newdust import graindist
from newdust import grainpop

ALLOWED_SIZES = ['Grain','Powerlaw','ExpCutoff']
ALLOWED_COMPS = ['Drude','Silicate','Graphite']
ALLOWED_SCATM = ['RG','Mie']

custom_sdist = graindist.sizedist.ExpCutoff(acut=0.5, nfold=12)
custom_comp  = graindist.composition.CmDrude(rho=2.2)

@pytest.mark.parametrize('sd', ALLOWED_SIZES)
@pytest.mark.parametrize('cm', ALLOWED_COMPS)
def test_graindist_init(sd, cm):
    test = graindist.GrainDist(sd, cm)
    assert isinstance(test, graindist.GrainDist)
    assert test.size.dtype  == sd
    assert test.comp.cmtype == cm

def test_custom_graindist():
    test = graindist.GrainDist(custom_sdist, custom_comp, custom=True)
    assert isinstance(test, graindist.GrainDist)
    assert test.size.dtype == custom_sdist.dtype
    assert test.comp.cmtype == custom_comp.cmtype

@pytest.mark.parametrize('sd', ALLOWED_SIZES)
@pytest.mark.parametrize('cm', ALLOWED_COMPS)
@pytest.mark.parametrize('sc', ALLOWED_SCATM)
def test_grainpop_inits(sd, cm, sc):
    test = grainpop.SingleGrainPop(sd, cm, sc)
    assert isinstance(test, grainpop.SingleGrainPop)
