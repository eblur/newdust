from astropy.io import fits

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
        print("You are attempting to run calculation from _ScatModel superclass.")
        print("No attributes will be updated.")
        return

    def write_efficiency_table(self, outfile, overwrite=True):
        hdr = self.__write_table_header()
        pardata = self.__write_table_pars(['lam','a'])
        # self.qsca, self.qext, and self.qabs will be separate cards in the FITS file
        #hdu_list = fits.HDUList([fits.PrimaryHDU(q) for q in [self.qext, self.qabs, self.qsca]])
        hdu_list = fits.HDUList(hdus=[fits.PrimaryHDU(self.qext), fits.ImageHDU(self.qsca)])
        hdu_list.writeto(outfile, overwrite=overwrite)
        return

    ##----- Helper material
    def __write_table_header(self):
        """ Writes FITS file header based on self.pars """
        return

    def __write_table_pars(self, parlist):
        """writes table parameters from self.pars"""
        # e.g. pars['lam'], pars['a']
        # should this be part of WCS?
        return
