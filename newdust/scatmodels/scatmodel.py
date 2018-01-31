from astropy.io import fits
from .. import constants as c

## Superclass _ScatModel
## See __init__ for API
class ScatModel(object):
    def __init__(self):
        self.qsca = None
        self.qext = None
        self.qabs = None
        self.diff = None
        self.pars = None

    def calculate(self, lam, a, cm, unit='kev', theta=0.0, **kwargs):
        print("You are attempting to run calculation from ScatModel superclass.")
        print("No attributes will be updated.")
        self.pars = {'lam':lam, 'a':a, 'cm':cm, 'unit':unit, 'theta':theta}
        return

    def write_efficiency_table(self, outfile, overwrite=True):
        # some basic info
        header    = self._write_table_header()
        # wavelength (or energy) and grain radius associated with calculation
        par_table = self._write_table_pars()
        # qext, qsca, and qext will be separate image cards in the FITS file
        img_list  = [fits.ImageHDU(q) for q in [self.qext, self.qabs, self.qsca]]
        # Put everything together to write the table
        fnl_list  = [header] + par_table + img_list
        hdu_list  = fits.HDUList(hdus=fnl_list)
        hdu_list.writeto(outfile, overwrite=overwrite)
        return

    ##----- Helper material
    def _write_table_header(self):
        """ Writes FITS file header based on self.pars """
        result = fits.Header()
        result['COMMENT']  = "Extinction efficiency table"
        result['COMMENT']  = "Qext, Qsca, Qabs in wavelength (or energy) vs grain radius"
        return fits.PrimaryHDU(header=result)

    def _write_table_pars(self):
        """writes table lam and a parameters from self.pars"""
        # e.g. pars['lam'], pars['a']
        # should this be part of WCS?
        c1 = fits.BinTableHDU.from_columns(
             [fits.Column(name='lam', array=c._make_array(self.pars['lam']),
             format='E', unit=self.pars['unit'])])
        c2 = fits.BinTableHDU.from_columns(
             [fits.Column(name='a', array=c._make_array(self.pars['a']),
             format='E', unit='micron')])
        return [c1, c2]
