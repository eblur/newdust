import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapz

__all__ = ['Halo']

ALLOWED_FTYPE = ['abs','ext']
ALLOWED_FUNIT = ['cgs','phot','count','none']

class Halo(object):
    """
    | An X-ray scattering halo
    |
    | **ATTRIBUTES**
    | lam
    | lam_unit
    | theta
    | htype
    | norm_int
    | taux
    | fabs
    | funit
    | intensity
    |
    | *properties*
    | fext
    | fhalo
    | percent_fabs
    | percent_fext
    |
    | *functions*
    | ecf(th, n)
    | __slice__(lmin, lmax)
    """
    def __init__(self, lam, theta, unit='kev'):
        self.lam       = lam
        self.lam_unit  = unit
        self.theta     = theta  # arcsec
        self.htype     = None
        self.norm_int  = None   # NE x NTH, arcsec^-2
        self.taux      = None   # NE, unitless
        self.fabs      = None   # NE, bin integrated flux [e.g. phot/cm^2/s, NOT phot/cm^2/s/keV]
        self.funit     = None
        self.intensity = None   # NTH, flux x arcsec^-2

    def calculate_intensity(self, flux, ftype='abs', funit='none'):
        assert self.norm_int is not None
        assert ftype in ALLOWED_FTYPE
        assert funit in ALLOWED_FUNIT
        if ftype == 'abs':
            fabs = flux
        if ftype == 'ext':
            fabs = flux * np.exp(self.taux)
        self.fabs = fabs
        NE, NTH = np.shape(self.norm_int)
        fa_grid = np.repeat(fabs.reshape(NE,1), NTH, axis=1)
        self.fa_grid = fa_grid
        self.intensity = np.sum(self.norm_int * fa_grid, axis=0)

    @property
    def fext(self):
        assert self.fabs is not None
        return self.fabs * np.exp(-self.taux)

    @property
    def fhalo(self):
        assert self.fabs is not None
        return self.fabs * (1.0 - np.exp(-self.taux))

    @property
    def percent_fabs(self):
        assert self.fabs is not None
        return np.sum(self.fhalo) / np.sum(self.fabs)

    @property
    def percent_fext(self):
        assert self.fabs is not None
        return np.sum(self.fhalo) / np.sum(self.fext)

    def ecf(self, th, n, log=False):
        # th = angle for computing enclosed fraction [arcsec]
        # n  = number of bins to use for interpolating
        # SMALL ANGLE SCATTERING IS ASSUMED!!
        assert self.intensity is not None
        assert th > self.theta[0]
        I_interp = interp1d(self.theta, self.intensity)
        thmax    = max(th, self.theta[-1])
        if log:
            th_grid = np.logspace(np.log10(self.theta[0]), np.log10(thmax), n)
        else:
            th_grid  = np.linspace(self.theta[0], thmax, n)
        I_grid   = I_interp(th_grid)
        fh_tot   = np.sum(self.fhalo)
        enclosed = trapz(I_grid * 2.0 * np.pi * th_grid, th_grid)
        return enclosed / fh_tot

    def __getitem__(self, i):
        result = Halo(self.lam[i], self.theta, unit=self.lam_unit)
        if self.norm_int is not None:
            result.htype    = self.htype
            result.norm_int = self.norm_int[i,...]
            result.taux     = self.taux[i]
        if self.fabs is not None:
            result.calculate_intensity(self.fabs[i], ftype='abs')
        return result

    def __slice__(self, lmin, lmax):
        ii = (self.lam >= lmin) & (self.lam < lmax)
        result = Halo(self.lam[ii], self.theta, unit=self.lam_unit)
        if self.norm_int is not None:
            result.htype    = self.htype
            result.norm_int = self.norm_int[ii,...]
            result.taux     = self.taux[ii]
        if self.fabs is not None:
            result.calculate_intensity(self.fabs[ii], ftype='abs')
        return result
