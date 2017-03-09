import os
import numpy as np
from scipy.interpolate import interp1d

from ... import constants as c

__all__ = ['CmDrude', 'CmGraphite', 'CmSilicate']

#------------- Index of Refraction object comes in handy --

#class CM(object):       # Complex index of refraction
#    def __init__( self, rp=1.0, ip=0.0 ):
#        self.rp = rp    # real part
#        self.ip = ip    # imaginary part

#-------------- Complex index of refraction calcs ---------

# ALL CM OBJECTS CONTAIN
#  cmtype : string ('Drude', 'Graphite', or 'Silicate')
#  rp     : either a function or scipy.interp1d object that is callable
#         : rp(E) where E is in [keV]
#  ip     : same as above, ip(E) where E is in [keV]


#------------- A quick way to grab a single CM ------------

def getCM(E, model=CmDrude()):
    """
    | **INPUTS**
    | E     : scalar or np.array [keV]
    | model : any Cm-type object
    |
    | **RETURNS**
    | Complex index of refraction : scalar or np.array of dtype='complex'
    """
    return model.rp(E) + 1j * model.ip(E)
