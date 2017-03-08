import numpy as np
from scipy.integrate import trapz

from graindist import sizedist
import constants as c

MDTEST  = 1.e-4  # g cm^-2
RHOTEST = 3.0    # g cm^-3

def percent_diff(a, b):
    # Return the absolute value of the percent difference between two values
    return np.abs(1.0 - (a/b))

def test_Grain():
    test = sizedist.Grain()
    nd   = test.ndens(MDTEST, RHOTEST)
    assert len(test.a) == 1
    assert len(nd)     == 1
    tot_mass = (4.0*np.pi/3.0) * (test.a * c.micron2cm)**3 * RHOTEST * nd
    assert percent_diff(tot_mass, MDTEST) <= 0.01

def test_Powerlaw():
    test = sizedist.Powerlaw()
    nd   = test.ndens(MDTEST, RHOTEST)
    assert len(test.a) == len(nd)
    md   = (4.0*np.pi/3.0) * (test.a * c.micron2cm)**3 * RHOTEST * nd
    tot_mass = trapz(md, test.a)
    assert percent_diff(tot_mass, MDTEST) <= 0.01
