import numpy as np
from scipy.integrate import trapz

import pytest

from newdust import constants as c
from newdust.graindist import composition
from newdust import scatmodels
from . import percent_diff

# -- Some Notes (March 27, 2017) -- Lia
# I couldn't get differential scattering cross section to integrate to total
# scattering cross-section within < 5%.  Could be the integration method?

CMD   = composition.CmDrude()
CMS   = composition.CmSilicate()
A_UM  = 0.5  # um
E_KEV = 2.0    # keV
LAM   = 4500.  # angs
THETA    = np.logspace(-10.0, np.log10(np.pi), 1000)  # 0->pi scattering angles (rad)
ASEC2RAD = (2.0 * np.pi) / (360.0 * 60. * 60.)     # rad / arcsec
TH_asec  = THETA / ASEC2RAD  # rad * (arcsec/rad)

def test_rgscat():
    test = scatmodels.RGscat()

    # total cross-section must asymptote to zero at high energy
    test.calculate(1.e10, A_UM, CMD, unit='kev')
    assert np.exp(-test.qsca) == 1.0
    # total cross-section must asymptote to infinity at low energy
    test.calculate(1.e-10, A_UM, CMD, unit='kev')
    assert np.exp(-test.qsca) == 0.0

    # differential cross-section must sum to total cross-section
    test.calculate(E_KEV, A_UM, CMD, unit='kev', theta=TH_asec)
    dtot   = trapz(test.diff * 2.0*np.pi*np.sin(THETA), THETA)  # cm^2
    sigsca = test.qsca * np.pi * (A_UM * c.micron2cm)**2  # cm^2
    assert percent_diff(dtot, sigsca) <= 0.05

    # Test that the absorption component for RG model is zero
    assert test.qsca == test.qext
    assert test.qabs == 0.0

    # Test E^-2 dependence of the total cross-section
    E2 = 2.0 * E_KEV
    qsca1 = test.qsca
    test.calculate(E2, A_UM, CMD, unit='kev')
    qsca2 = test.qsca
    assert percent_diff(qsca2, qsca1/4.0) <= 0.01

    # Test a^4 scattering cross section dependence on the grain size
    #  Qsca = sigma_sca / (pi a^2), so qsca dependence is a^2
    A2 = 2.0 * A_UM
    test.calculate(E_KEV, A2, CMD, unit='kev')
    qsca3 = test.qsca
    assert percent_diff(qsca3, 4.0*qsca1) <= 0.01

    # Test that differential cross section goes to zero at large ANGLES
    test.calculate(E_KEV, A_UM, CMD, unit='kev', theta=1.e10)
    assert test.diff == 0.0

    # Test the write function
    test.write_table('qrg.fits')
    # Test the read function
    new_test = scatmodels.ScatModel(from_file='qrg.fits')
    assert percent_diff(test.qext, new_test.qext) <= 1.e-5
    assert percent_diff(test.qabs, new_test.qabs) <= 1.e-5
    assert percent_diff(test.qsca, new_test.qsca) <= 1.e-5
    
@pytest.mark.parametrize('cm',
                         [composition.CmDrude(),
                          composition.CmSilicate(),
                          composition.CmGraphite()])
def test_mie(cm):
    test = scatmodels.Mie()
    test.calculate(LAM, A_UM, cm, unit='angs')

    # Test that cross-section asymptotes to a small number when grain size gets very small
    test.calculate(LAM, 1.e-10, cm, unit='angs')
    assert percent_diff(np.exp(test.qsca), 1.0) < 0.001
    assert percent_diff(np.exp(test.qext), 1.0) < 0.001
    assert percent_diff(np.exp(test.qabs), 1.0) < 0.001

    # Test that the differential scattering cross section integrates to total
    test.calculate(LAM, A_UM, cm, unit='angs', theta=TH_asec)
    dtot = trapz(test.diff * 2.0*np.pi*np.sin(THETA), THETA)  # cm^2
    sigsca = test.qsca * np.pi * (A_UM * c.micron2cm)**2  # cm^2
    assert percent_diff(dtot, sigsca) <= 0.01

    # Test the write function
    test.write_table('qmie.fits')
    new_test = scatmodels.ScatModel(from_file='qmie.fits')
    assert percent_diff(test.qext, new_test.qext) <= 1.e-5
    assert percent_diff(test.qabs, new_test.qabs) <= 1.e-5
    assert percent_diff(test.qsca, new_test.qsca) <= 1.e-5

@pytest.mark.parametrize('sm',
                         [scatmodels.RGscat(),
                          scatmodels.Mie()])
def test_dimensions(sm):
    NE, NA, NTH = 2, 20, len(TH_asec)
    LAMVALS = np.linspace(1000.,5000.,NE)  # angs
    AVALS   = np.linspace(0.1, 0.5, NA)    # um
    sm.calculate(LAMVALS, AVALS, composition.CmSilicate(), unit='angs', theta=TH_asec)
    assert np.shape(sm.qsca) == (NE, NA)
    assert np.shape(sm.qext) == (NE, NA)
    assert np.shape(sm.qabs) == (NE, NA)
    assert np.shape(sm.diff) == (NE, NA, NTH)

    dtot1 = trapz(sm.diff[0,0,:] * 2.0*np.pi*np.sin(THETA), THETA)
    dtot2 = trapz(sm.diff[-1,-1,:] * 2.0*np.pi*np.sin(THETA), THETA)
    ssca1 = sm.qsca[0,0] * np.pi * (AVALS[0] * 1.e-4)**2
    ssca2 = sm.qsca[-1,-1] * np.pi * (AVALS[-1] * 1.e-4)**2
    assert percent_diff(dtot1, ssca1) <= 0.05
    assert percent_diff(dtot2, ssca2) <= 0.05

from newdust.scatmodels.scatmodel import ScatModel
# Test the writing and reading functions
@pytest.mark.parametrize('sm', [ScatModel()])
#                         [scatmodels.RGscat(),
#                          scatmodels.Mie()])
def test_writing(sm):
    sm.calculate(0.0, 0.0, 0.0)
    sm.write_table('test_scatmodels.fits')
    test = ScatModel(from_file='test_scatmodels.fits')
