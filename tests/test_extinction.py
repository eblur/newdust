import numpy as np
from scipy.integrate import trapz

import pytest

from newdust import constants as c
from newdust import graindist
from newdust import extinction
from . import percent_diff

NE, NA, NTH = 2, 20, 1000
LAMVALS = np.linspace(1000.,5000.,NE)  # angs
AVALS   = np.linspace(0.1, 0.5, NA)    # um

THETA    = np.logspace(-10.0, np.log10(np.pi), NTH)  # 0->pi scattering angles (rad)
ASEC2RAD = (2.0 * np.pi) / (360.0 * 60. * 60.)     # rad / arcsec
TH_asec  = THETA / ASEC2RAD  # rad * (arcsec/rad)

MRN_SIL = graindist.make_GrainDist('Powerlaw','Silicate')
MRN_DRU = graindist.make_GrainDist('Powerlaw','Drude')
EXP_SIL = graindist.make_GrainDist('ExpCutoff','Silicate')
GRAIN   = graindist.make_GrainDist('Grain','Drude')

RG  = extinction.scatmodels.RGscat()
MIE = extinction.scatmodels.Mie()

ALLOWED_SCATM = ['RG', 'Mie']

# Test the helper function that returns and Extinction object
@pytest.mark.parametrize('estring', ALLOWED_SCATM)
def test_Extinction(estring):
    test = extinction.make_Extinction(estring)
    assert isinstance(test, extinction.Extinction)

# Test that all the computations can run
@pytest.mark.parametrize(('gd','sm'),
                         [(MRN_SIL, RG), (MRN_SIL, MIE),
                          (MRN_DRU, RG), (MRN_DRU, MIE),
                          (EXP_SIL, RG), (EXP_SIL, MIE),
                          (GRAIN, RG), (GRAIN, MIE)])
def test_calculations(gd, sm):
    test = extinction.Extinction(sm)
    test.calculate(gd, LAMVALS, unit='angs', theta=TH_asec)
    assert np.shape(test.tau_ext) == (NE,)
    assert np.shape(test.tau_sca) == (NE,)
    assert np.shape(test.tau_abs) == (NE,)
    assert all(percent_diff(test.tau_ext, test.tau_abs + test.tau_sca) <= 0.01)

# Make sure that doubling the dust mass doubles the extinction
@pytest.mark.parametrize('estring', ALLOWED_SCATM)
def test_mass_double(estring):
    gd1 = graindist.make_GrainDist('Powerlaw','Silicate')
    gd2 = graindist.make_GrainDist('Powerlaw', 'Silicate')
    gd2.md = 2.0 * gd1.md

    sm1 = extinction.make_Extinction(estring)
    sm1.calculate(gd1, LAMVALS, unit='angs')
    sm2 = extinction.make_Extinction(estring)
    sm2.calculate(gd2, LAMVALS, unit='angs')

    assert all(percent_diff(sm2.tau_ext, 2.0 * sm1.tau_ext) <= 0.01)
    assert all(percent_diff(sm2.tau_abs, 2.0 * sm1.tau_abs) <= 0.01)
    assert all(percent_diff(sm2.tau_sca, 2.0 * sm1.tau_sca) <= 0.01)
