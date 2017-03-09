import numpy as np
from scipy.integrate import trapz

from newdust.graindist import sizedist
from newdust import constants as c
from . import percent_diff

MDTEST  = 1.e-4  # g cm^-2
RHOTEST = 3.0    # g cm^-3

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

def test_ExpCutoff():
    test = sizedist.ExpCutoff()
    nd   = test.ndens(MDTEST, RHOTEST)
    assert len(test.a) == len(nd)
    md   = (4.0*np.pi/3.0) * (test.a * c.micron2cm)**3 * RHOTEST * nd
    tot_mass = trapz(md, test.a)
    assert percent_diff(tot_mass, MDTEST) <= 0.01
