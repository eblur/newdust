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
        #self.read_from_table(from_table)

    def calculate(self, lam, a, cm, unit='kev', theta=0.0, **kwargs):
        print("You are attempting to run calculation from ScatModel superclass.")
        print("No attributes will be updated.")
        self.pars = {'lam':lam, 'a':a, 'cm':cm, 'unit':unit, 'theta':theta}
        return

    def write_table(self, outfile, overwrite=True):
        # some basic info
        header    = self._write_table_header()
        # wavelength (or energy) and grain radius associated with calculation
        par_table = self._write_table_pars()
        # qext, qsca, and qext will be separate image cards in the FITS file
        img_list = []
        for (q, h) in zip([self.qext, self.qabs, self.qsca, self.diff],
                          ['Qext', 'Qabs', 'Qsca', 'Diff-xsect (cm^2/ster)']):
            htemp = fits.Header()
            htemp['TYPE'] = h
            img_list.append(fits.ImageHDU(self.qext, header=htemp))
        # Put everything together to write the table
        fnl_list  = [header] + par_table + img_list
        hdu_list  = fits.HDUList(hdus=fnl_list)
        hdu_list.writeto(outfile, overwrite=overwrite)
        return

    def read_from_table(self, infile):
        ff = fits.open(infile)
        # Load parameteric information
        lam   = ff[1].data['lam']
        unit  = ff[1].data['lam'].unit
        a     = ff[2].data['a']
        theta = ff[3].data['theta']
        self.pars = {'lam':lam, 'a':a, 'unit':unit, 'theta':theta}

        # Load extinction information
        qtypes = {'Qext':self.qext, 'Qabs':self.qabs, 'Qsca':self.qsca, 'Diff-xsect (cm^2/ster)':self.diff}
        for i in range(4,8):  # runs on hdus 4,5,6,7
            htype = ff[i].header['TYPE']
            qtypes[htype] = ff[i].data
        return

    ##----- Helper material
    def _write_table_header(self):
        """ Writes FITS file header based on self.pars """
        result = fits.Header()
        result['COMMENT']  = "Extinction efficiency and differential cross-sections"
        result['COMMENT']  = "HDUS 4-6 are Qext, Qsca, Qabs in wavelength (or energy) vs grain radius"
        result['COMMENT']  = "HDU 7 is differential cross-section (cm^2/ster)"
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
        c3 = fits.BinTableHDU.from_columns(
             [fits.Column(name='theta', array=c._make_array(self.pars['theta']),
             format='E', unit='arcsec')])
        return [c1, c2, c3]
