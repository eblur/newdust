from re import A
from matplotlib.pyplot import ion
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
PAH_UM = 0.01 # um
E_KEV = 2.0    # keV
LAM   = 4500.  # angs
THETA    = np.logspace(-10.0, np.log10(np.pi), 1000)  # 0->pi scattering angles (rad)
THETA_ARCSEC = (THETA * u.radian).to('arcsec')

#WAVEL_GRID = np.linspace(3000., 4000., 10) * u.angstrom
WAVEL_GRID = np.linspace(1.0, 40., 10) * u.angstrom
A_CM = (A_UM*u.micron).to('cm')


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

@pytest.mark.parametrize('cm',
                         [composition.CmDrude(),
                          composition.CmSilicate(),
                          composition.CmGraphite()])
def test_mie(cm):
    test = scatteringmodel.Mie()
    test.calculate(LAM * u.angstrom, A_UM, cm)

    # Test that cross-section asymptotes to a small number when grain size gets very small
    test.calculate(LAM * u.angstrom, 1.e-10, cm, unit='angs')
    assert percent_diff(np.exp(test.qsca), 1.0) < 0.001
    assert percent_diff(np.exp(test.qext), 1.0) < 0.001
    assert percent_diff(np.exp(test.qabs), 1.0) < 0.001

    # Test that the differential scattering cross section integrates to qsca
    test.calculate(LAM * u.angstrom, A_UM, cm, theta=THETA_ARCSEC)
    dtot = trapz(test.diff * 2.0*np.pi*np.sin(THETA), THETA)  # unitless
    assert percent_diff(dtot, test.qsca) <= 0.01

    # Test that the extinction values are correct
    assert percent_diff(test.qext, test.qabs + test.qsca) <= 0.01


@pytest.mark.parametrize('sm',
                         [scatteringmodel.RGscattering(),
                          scatteringmodel.Mie()])
def test_dimensions(sm):
    NE, NA, NTH = 2, 20, len(THETA_ARCSEC)
    LAMVALS = np.linspace(1000.,5000.,NE) * u.angstrom
    AVALS   = np.linspace(0.1, 0.5, NA) * u.micron
    sm.calculate(LAMVALS, AVALS, composition.CmSilicate(), theta=THETA_ARCSEC)
    assert np.shape(sm.qsca) == (NE, NA)
    assert np.shape(sm.qext) == (NE, NA)
    assert np.shape(sm.qabs) == (NE, NA)
    assert np.shape(sm.diff) == (NE, NA, NTH)

def test_read_write():
    cm = composition.CmDrude()
    test = scatteringmodel.RGscattering()
    test.calculate(WAVEL_GRID, A_CM, cm, THETA)
    test.write_table('qtest.fits')

    # Test the read function
    new_test = scatteringmodel.ScatteringModel(from_file='qtest.fits')
    assert np.all(percent_diff(test.qext, new_test.qext) <= 1.e-5)
    assert np.all(percent_diff(test.qabs, new_test.qabs) <= 1.e-5)
    assert np.all(percent_diff(test.qsca, new_test.qsca) <= 1.e-5)
    assert np.shape(new_test.qext) == (np.size(WAVEL_GRID), np.size(A_CM))

    # Test that the parameters are the same
    # Astropy units didn't have enoug accuracy to use equalities
    assert np.all(percent_diff(
        new_test.pars['lam'].to('cm', equivalencies=u.spectral()).value,
        test.pars['lam'].to('cm', equivalencies=u.spectral()).value) <= 1.e-4)
    assert np.all(percent_diff(
        new_test.pars['a'].to('cm', equivalencies=u.spectral()).value,
        test.pars['a'].to('cm', equivalencies=u.spectral()).value) <= 1.e-4)
    assert np.all(percent_diff(
        new_test.pars['theta'].to('arcsec').value, 
        test.pars['theta'].to('arcsec').value) <= 1.e-4)

def test_PAHs():
    neutral_PAH = scatteringmodel.PAH('neu')
    neutral_PAH.calculate(E_KEV, PAH_UM)

    ionized_PAH = scatteringmodel.PAH('neu')
    ionized_PAH.calculate(E_KEV, PAH_UM)
    
    # Test that qext = qsca+qabs
    assert percent_diff(neutral_PAH.qext, neutral_PAH.qabs + neutral_PAH.qsca) <= 0.01
    assert percent_diff(ionized_PAH.qext, ionized_PAH.qabs + ionized_PAH.qsca) <= 0.01

    # Test that an absurdly high energy if off th grid and returns zero
    neutral_PAH.calculate(1.e10, PAH_UM)
    assert neutral_PAH.qext == 0
    ionized_PAH.calculate(1.e10, PAH_UM)
    assert ionized_PAH.qext == 0

    # Test that units work
    neutral_PAH.calculate(LAM * u.angstrom, PAH_UM)
    neutral_PAH.calculate(E_KEV, (PAH_UM * u.micron).to('cm'))
    # if it runs, assume all is well

