import os

def _find_cmfile(name):
    root_path = os.path.dirname(__file__).rstrip('composition')
    data_path = root_path + 'tables/'
    return data_path + name

from .cmdrude import CmDrude
from .cmsilicate import CmSilicate
from .cmgraphite import CmGraphite

def _getCM(E, model):
    """
    | **INPUTS**
    | E     : scalar or np.array [keV]
    | model : any Cm-type object
    |
    | **RETURNS**
    | Complex index of refraction : scalar or np.array of dtype='complex'
    """
    return model.rp(E) + 1j * model.ip(E)
