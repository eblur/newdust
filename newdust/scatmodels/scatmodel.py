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
        header    = self._write_table_header()
        par_table = self._write_table_pars()
        # self.qsca, self.qext, and self.qabs will be separate cards in the FITS file
        #hdu_list = fits.HDUList([fits.PrimaryHDU(q) for q in [self.qext, self.qabs, self.qsca]])
        hdu_list = fits.HDUList(hdus=[fits.PrimaryHDU(self.qext), fits.ImageHDU(self.qsca)])
        hdu_list.writeto(outfile, overwrite=overwrite)
        return

    ##----- Helper material
    def _write_table_header(self):
        """ Writes FITS file header based on self.pars """
        return

    def _write_table_pars(self):
        """writes table lam and a parameters from self.pars"""
        # e.g. pars['lam'], pars['a']
        # should this be part of WCS?
        c1 = fits.Column(name='lam', array=c._make_array(self.pars['lam']), format='E', unit=self.pars['unit'])
        c2 = fits.Column(name='a', array=c._make_array(self.pars['a']), format='E', unit='micron')
        return fits.BinTableHDU.from_columns([c1, c2])
