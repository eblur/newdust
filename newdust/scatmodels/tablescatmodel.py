from scipy.interpolate import interp2d, RegularGridInterpolator
from .. import constants as c
from .scatmodel import ScatModel

ALLOWED_UNITS = ['kev', 'angs']

class TableScatModel(ScatModel):
    def __init__(self, fname, stype='Table', method_2d='linear', method_3d='linear'):
        ScatModel.__init__(self)
        self.stype  = stype
        self._table = ScatModel(from_file=fname)
        ## Always keep things in kev units
        ener_kev   = c.lam_kev(self._table.pars['lam'], self._table.pars['unit'])
        self._qext = interp2d(ener_kev, self._table.pars['a'], self._table.qext, kind=method_2d)
        self._qabs = interp2d(ener_kev, self._table.pars['a'], self._table.qabs, kind=method_2d)
        self._qsca = interp2d(ener_kev, self._table.pars['a'], self._table.qsca, kind=method_2d)
        #self._diff = RegularGridInterpolator((ener_kev, self._table.pars['a'], self._table.pars['theta']),
        #                                      self._table.diff, method=method_3d)

    def calculate(self, lam, a, cm, unit='kev', theta=0.0):
        self.pars = dict(zip(['lam','a','cm','theta','unit'], [lam, a, cm, theta, unit]))
        ener_kev  = c.lam_kev(lam, unit)
        self.qext = self._qext(ener_kev, a)
        self.qabs = self._qabs(ener_kev, a)
        self.qsca = self._qsca(ener_kev, a)
        #self.diff = self._diff(np.array([ener_kev, a, theta]).T)
