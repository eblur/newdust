import numpy as np
from scipy.integrate import trapz

from newdust import constants as c
from newdust import graindist
from newdust.extinction import scatmodels
from . import percent_diff

CMD   = graindist.composition.CmDrude()
A_UM  = 1.0  # um
E_KEV = 1.0  # keV
THETA = np.logspace(-1.0, np.log10(2.0*np.pi), 1000)  # scattering angles (ster)
ASEC2RAD = (2.0 * np.pi) / (360.0 * 60. * 60.)  # rad / arcsec

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
    #assert percent_diff(dtot, sigsca) <= 0.01
