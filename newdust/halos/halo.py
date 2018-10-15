import numpy as np
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.integrate import trapz
from astropy.io import fits
import astropy.units as u
from .. import constants as c

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
    | *methods*
    | ecf(th, n)
    | __getitem__(lmin, lmax)
    """
    def __init__(self, lam=1.0, theta=1.0, unit='kev', from_file=None, **kwargs):
        self.lam       = lam    # length NE
        self.lam_unit  = unit   # 'kev' or 'angs'
        self.theta     = theta  # length NTH, arcsec
        self.htype     = None   # modified by halo calculation functions
        self.norm_int  = None   # NE x NTH, arcsec^-2
        self.taux      = None   # NE, unitless
        self.fabs      = None   # NE, bin integrated flux [e.g. phot/cm^2/s, NOT phot/cm^2/s/keV]
        self.funit     = None
        self.intensity = None   # NTH, flux x arcsec^-2
        if from_file is not None:
            self._read_from_file(from_file, **kwargs)

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

    # http://stackoverflow.com/questions/2936863/python-implementing-slicing-in-getitem
    def __getitem__(self, key):
        if isinstance(key, int):
            return self._get_lam_index(key)
        if isinstance(key, slice):
            lmin = key.start
            lmax = key.stop
            if lmin is None:
                ii = (self.lam < lmax)
            elif lmax is None:
                ii = (self.lam >= lmin)
            else:
                ii = (self.lam >= lmin) & (self.lam < lmax)
            return self._get_lam_slice(ii)

    def _get_lam_slice(self, ii):
        result = Halo(self.lam[ii], self.theta, unit=self.lam_unit)
        if self.norm_int is not None:
            result.htype    = self.htype
            result.norm_int = self.norm_int[ii,...]
            result.taux     = self.taux[ii]
        if self.fabs is not None:
            result.calculate_intensity(self.fabs[ii], ftype='abs')
        return result

    def _get_lam_index(self, i):
        assert isinstance(i, int)
        result = Halo(self.lam[i], self.theta, unit=self.lam_unit)
        if self.norm_int is not None:
            result.htype    = self.htype
            result.norm_int = np.array([self.norm_int[i,...]])
            result.taux     = self.taux[i]
        if self.fabs is not None:
            flux = np.array([self.fabs[i]])
            result.calculate_intensity(flux, ftype='abs')
        return result

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

    def write(self, filename, overwrite=True):
        """
        Inputs
        ------
        filename : string
            Name for the output FITS file

        Write the scattering halo 'norm_int' attribute to a FITS file.
        The file will also store the relevant LAM, THETA, and TAUX values.
        """
        hdr = fits.Header()
        hdr['COMMENT'] = "Normalized halo intensity as a function of angle"
        hdr['COMMENT'] = "HDU 1 is LAM, HDU 2 is THETA, HDU 3 is TAUX"
        primary_hdu = fits.PrimaryHDU(self.norm_int, header=hdr)

        hdus_to_write = [primary_hdu] + self._write_halo_pars()
        hdu_list = fits.HDUList(hdus=hdus_to_write)
        hdu_list.writeto(filename, overwrite=overwrite)
        return

    ##----- Helper material
    def _write_halo_pars(self):
        """Write the FITS file"""
        # e.g. pars['lam'], pars['a']
        # should this be part of WCS?
        c1 = fits.BinTableHDU.from_columns(
             [fits.Column(name='lam', array=c._make_array(self.lam),
             format='E', unit=self.lam_unit)])
        c2 = fits.BinTableHDU.from_columns(
             [fits.Column(name='theta', array=c._make_array(self.theta),
             format='E', unit='arcsec')])
        c3 = fits.BinTableHDU.from_columns(
             [fits.Column(name='taux', array=c._make_array(self.taux),
             format='E', unit='')])
        return [c1, c2, c3]

    def _read_from_file(self, filename, htype='CustomFile'):
        """Read a Halo object from a FITS file"""
        ff = fits.open(filename)
        # Load the normalized intensity
        self.norm_int = ff[0].data
        # Load the other parameters
        self.lam = ff[1].data['lam']
        self.lam_unit = ff[1].columns['lam'].unit
        self.theta = ff[2].data['theta']
        self.taux = ff[3].data['taux']
        # Set halo type
        self.htype = htype

    ##------ Make a fake image with a telescope arf
    def fake_image(self, arf, src_flux, exposure, 
                   pix_scale=0.5, num_pix=[2400,2400],
                   imin=0, imax=-1, save_file=None, **kwargs):
        """Make a fake image file using a telescope ARF as input.

        Parameters
        ----------
        arf : string
            Filename of telescope ARF

        src_flux : numpy.ndarray [phot/cm^2/s]
            Describes the absorbed (without X-ray scattering) flux
            model for the central X-ray point source. Must correspond
            to the values in Halo.lam

        exposure : float [seconds]
            Exposure time to use

        pix_scale : float [arcsec]
            Size of simulated pixels

        num_pix : ints (nx,ny) 
            Size of pixel grid to use

        imin : int
            Energy index to start with (Default: 0)

        imax : int
            Energy index to end with, exclusive, except when using
            negative indexing (Default: -1)

        save_file : string (Default:None)
            Filename to use if you want to save the output to a .fits

        Returns
        -------
        2D numpy.ndarray of shape (nx, ny), representing the image of
        a dust scattering halo. The halo intensity at different
        energies are converted into counts using the ARF. Then a
        Poisson distribution is used to simulate the number of counts
        in each pixel.
        
        If the user supplies a file name string using the save_file
        keyword, a FITS file will be saved.

        """
        assert len(src_flux) == len(self.lam)

        xlen, ylen = num_pix
        xcen, ycen = xlen//2, ylen//2
        ccdx, ccdy = np.meshgrid(np.arange(xlen), np.arange(ylen))
        radius = np.sqrt((ccdx - xcen)**2 + (ccdy - ycen)**2)

        # Typical ARF files have columns 'ENERG_LO', 'ENERG_HI', 'SPECRESP'
        arf_data = fits.open(arf)['SPECRESP'].data
        arf_x = 0.5*(arf_data['ENERG_LO'] + arf_data['ENERG_HI'])
        arf_y = arf_data['SPECRESP']
        arf   = InterpolatedUnivariateSpline(arf_x, arf_y, k=1)

        # Source counts to use for each energy bin
        if self.lam_unit == 'angs':
            ltemp      = self.lam * u.angstrom
            ltemp_kev  = ltemp.to(u.keV, equivalencies=u.spectral()).value
            arf_temp   = arf(ltemp_kev)[::-1]
            src_counts = src_flux * arf_temp * exposure
        else:
            src_counts = src_flux * arf(self.lam) * exposure

        # Decide which energy indexes to use
        iend = imax
        if imax < 0:
            iend = np.arange(len(self.lam)+1)[imax]
            
        # Add up randomized image for each energy index value
        r_asec = radius * pix_scale
        result = np.zeros_like(radius)
        for i in np.arange(imin, iend):
            # interp object for halo grid (arcsec, counts/arcsec^2)
            h_interp = InterpolatedUnivariateSpline(
                self.theta, self.norm_int[i,:] * src_counts[i], k=1)
            # corresponding counts at each radial value in the grid
            pix_counts = h_interp(r_asec) * pix_scale**2
            # use poisson statistics to get a random value
            pix_random = np.random.poisson(pix_counts)
            # add it to the final result
            result += pix_random

        if save_file is not None:
            hdu  = fits.PrimaryHDU(result)
            hdul = fits.HDUList([hdu])
            hdul.writeto(save_file, overwrite=True)

        return result
