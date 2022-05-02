import numpy as np
from scipy.integrate import trapz
import pytest
import astropy.units as u

from newdust.graindist import composition
from newdust import scatteringmodel
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
THETA_ARCSEC = (THETA * u.radian).to('arcsec')

def test_rgscat():
    test = scatteringmodel.RGscattering()

    # total cross-section must asymptote to zero at high energy
    test.calculate(1.e10 * u.keV, A_UM, CMD)
    assert np.exp(-test.qsca) == 1.0
    # total cross-section must asymptote to infinity at low energy
    test.calculate(1.e-10 * u.keV, A_UM, CMD)
    assert np.exp(-test.qsca) == 0.0

    # differential cross-section must sum to Qsca
    test.calculate(E_KEV, A_UM, CMD, theta=THETA_ARCSEC)
    dtot   = trapz(test.diff * 2.0*np.pi*np.sin(THETA), THETA)  # unitless
    assert percent_diff(dtot, test.qsca) <= 0.05

    # Test that it still works when I use units
    test.calculate(LAM * u.angstrom, A_UM * u.micron, CMD, theta=THETA)
    dtot   = trapz(test.diff * 2.0*np.pi*np.sin(THETA), THETA)  # unitless
    assert percent_diff(dtot, test.qsca) <= 0.05

    # Test that the absorption component for RG model is zero
    assert test.qsca == test.qext
    assert test.qabs == 0.0

    # Test E^-2 dependence of the total cross-section
    test.calculate(E_KEV, A_UM, CMD)
    qsca1 = test.qsca
    test.calculate(2.0 * E_KEV, A_UM, CMD)
    qsca2 = test.qsca
    assert percent_diff(qsca2, qsca1/4.0) <= 0.01

    # Test a^4 scattering cross section dependence on the grain size
    #  Qsca = sigma_sca / (pi a^2), so qsca dependence is a^2
    A2 = 2.0 * A_UM
    test.calculate(E_KEV, A2, CMD)
    qsca3 = test.qsca
    assert percent_diff(qsca3, 4.0*qsca1) <= 0.01

    # Test that differential cross section goes to zero at large ANGLES
    test.calculate(E_KEV, A_UM, CMD, theta=1.e10)
    assert test.diff == 0.0

    # Test that the extinction values are correct
    assert percent_diff(test.qext, test.qabs + test.qsca) <= 0.01
    
    # Test the write function
    test.write_table('qrg.fits')
    # Test the read function
    new_test = scatteringmodel.ScatteringModel(from_file='qrg.fits')
    assert percent_diff(test.qext, new_test.qext) <= 1.e-5
    assert percent_diff(test.qabs, new_test.qabs) <= 1.e-5
    assert percent_diff(test.qsca, new_test.qsca) <= 1.e-5

"""
@pytest.mark.parametrize('cm',
                         [composition.CmDrude(),
                          composition.CmSilicate(),
                          composition.CmGraphite()])
def test_mie(cm):
    test = scatteringmodel.Mie()
    test.calculate(LAM, A_UM, cm, unit='angs')

    # Test that cross-section asymptotes to a small number when grain size gets very small
    test.calculate(LAM, 1.e-10, cm, unit='angs')
    assert percent_diff(np.exp(test.qsca), 1.0) < 0.001
    assert percent_diff(np.exp(test.qext), 1.0) < 0.001
    assert percent_diff(np.exp(test.qabs), 1.0) < 0.001

    # Test that the differential scattering cross section integrates to qsca
    test.calculate(LAM, A_UM, cm, unit='angs', theta=TH_asec)
    dtot = trapz(test.diff * 2.0*np.pi*np.sin(THETA), THETA)  # unitless
    assert percent_diff(dtot, test.qsca) <= 0.01

    # Test that the extinction values are correct
    assert percent_diff(test.qext, test.qabs + test.qsca) <= 0.01

    # Test the write function
    test.write_table('qmie.fits')
    # Test the read function
    new_test = scatteringmodel.ScatteringModel(from_file='qmie.fits')
    assert percent_diff(test.qext, new_test.qext) <= 1.e-5
    assert percent_diff(test.qabs, new_test.qabs) <= 1.e-5
    assert percent_diff(test.qsca, new_test.qsca) <= 1.e-5

@pytest.mark.parametrize('sm',
                         [scatteringmodel.RGscat(),
                          scatteringmodel.Mie()])
def test_dimensions(sm):
    NE, NA, NTH = 2, 20, len(TH_asec)
    LAMVALS = np.linspace(1000.,5000.,NE)  # angs
    AVALS   = np.linspace(0.1, 0.5, NA)    # um
    sm.calculate(LAMVALS, AVALS, composition.CmSilicate(), unit='angs', theta=TH_asec)
    assert np.shape(sm.qsca) == (NE, NA)
    assert np.shape(sm.qext) == (NE, NA)
    assert np.shape(sm.qabs) == (NE, NA)
    assert np.shape(sm.diff) == (NE, NA, NTH)
"""