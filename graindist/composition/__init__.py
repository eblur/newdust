import os

def _find_cmfile(name):
    root_path = os.path.dirname(__file__).rstrip('composition')
    data_path = root_path + 'tables/'
    return data_path + name

from .cmdrude import CmDrude
from .cmsilicate import CmSilicate
from .cmgraphite import CmGraphite
