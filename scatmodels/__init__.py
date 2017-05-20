from .rgscat import RGscat
from .miescat import Mie
from .. import constants as c

"""
--------------------------------------------------------------
    API
--------------------------------------------------------------
 A dust scattering model should contain functions that take
 energy value, complex index of refraction object (see cmlib),
 and grain sizes

 Qsca ( E  : scalar or np.array [keV]
        cm : cmtype object from cmi.py
        a  : scalar [grain size, micron] ) :
 returns scalar or np.array [scattering efficiency, unitless]

 Diff ( cm : cmtype object from cmi.py
        theta : scalar or np.array [angle, arcsec]
        a  : scalar [grain size, micron]
        E  : scalar or np.array [energy, keV]
        ** if len(E) > 1 and len(theta) > 1, then len(E) must equal len(theta)
           returns dsigma/dOmega of values (E0,theta0), (E1,theta1) etc...

 Some (but not all) scattering models may also contain related extinction terms

 Qext ( E  : scalar or np.array [keV]
        cm : cmtype object cmi.py
        a  : scalar [grain size, micron] ) :
 returns scalar or np.array [extinction efficiency, unitless]

 Qabs ( E  : scalar or np.array [keV]
        cm : cmtype object cmi.py
        a  : scalar [grain size, micron] ) :
 returns scalar or np.array [absorption efficiency, unitless]
"""
