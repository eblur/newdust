import numpy as np
from scipy.integrate import trapz

from newdust import constants as c
from newdust import graindist
from newdust.extinction import scatmodels
from . import percent_diff

# -- Some Notes (March 27, 2017) -- Lia
# I couldn't get differential scattering cross section to integrate to total
# scattering cross-section within < 5%.  Could be the integration method?

CMD   = graindist.composition.CmDrude()
A_UM  = 0.5  # um
E_KEV = 2.0  # keV
THETA = np.logspace(-10.0, np.log10(np.pi), 1000)  # scattering angles (rad)
ASEC2RAD = (2.0 * np.pi) / (360.0 * 60. * 60.)     # rad / arcsec

def test_rgscat():
    test = scatmodels.RGscat()

    # total cross-section must asymptote to zero at high energy
    assert np.exp(-test.Qsca(1.e10, A_UM, CMD, unit='kev')) == 1.0
    # total cross-section must asymptote to infinity at low energy
    assert np.exp(-test.Qsca(1.e-10, A_UM, CMD, unit='kev')) == 0.0

    # differential cross-section must sum to total cross-section
    TH_asec = THETA / ASEC2RAD  # rad * (arcsec/rad)
    dsca = test.Diff(TH_asec, E_KEV, A_UM, CMD, unit='kev')  # cm^2 ster^-1
    dtot = trapz(dsca * 2.0*np.pi*THETA, THETA)  # cm^2
    sigsca = test.Qsca(E_KEV, A_UM, CMD, unit='kev') * np.pi * (A_UM * c.micron2cm)**2  # cm^2
    assert percent_diff(dtot, sigsca) <= 0.05

    # Test E^-2 dependence of the total cross-section
    E2 = 2.0 * E_KEV
    qsca1 = test.Qsca(E_KEV, A_UM, CMD, unit='kev')
    qsca2 = test.Qsca(E2, A_UM, CMD, unit='kev')
    assert percent_diff(qsca2, qsca1/4.0) <= 0.01

    # Test a^4 scattering cross section dependence on the grain size
    #  Qsca = sigma_sca / (pi a^2), so qsca dependence is a^2
    A2 = 2.0 * A_UM
    qsca3 = test.Qsca(E_KEV, A2, CMD, unit='kev')
    assert percent_diff(qsca3, 4.0*qsca1) <= 0.01

    # Test that the absorption component for RG model is zero
    qext = test.Qext(E_KEV, A_UM, CMD, unit='kev')
    assert qsca1 == qext
    assert test.Qabs(E_KEV, A_UM, CMD, unit='kev') == 0.0

def test_mie():
    test = scatmodels.Mie()
    test.calculate(E_KEV, A_UM, CMD, unit='kev')

    # total cross-section must asymptoted to zero at high energy
    #assert np.exp(-test.Qsca(recalc=True, lam=1.e10, a=A_UM, cm=CMD)) == 1.0
