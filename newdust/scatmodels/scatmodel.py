from astropy.io import fits
from .. import constants as c

## Superclass ScatModel
## See __init__ for API
class ScatModel(object):
    def __init__(self, from_file=None):
        self.qsca = None
        self.qext = None
        self.qabs = None
        self.diff = None
        self.pars = None
        self.stype = 'Empty'
        if from_file is not None:
            self.read_from_table(from_file)
            self.stype = from_file

    # Base calculate method does nothing
    def calculate(self, lam, a, cm, **kwargs):
        return

    def write_table(self, outfile, overwrite=True):
        # Don't write a table that has not been calculated
        try:
            assert self.pars is not None
        except:
            print("There are no values to store.")
            return

        # All must be well! Store information in a FITS file
        header    = self._write_table_header()
        # wavelength (or energy) and grain radius associated with calculation
        par_table = self._write_table_pars()
        # qext, qsca, and qext will be separate image cards in the FITS file
        img_list = []
        for (q, h) in zip([self.qext, self.qabs, self.qsca, self.diff],
                          ['Qext', 'Qabs', 'Qsca', 'Diff-xsect (cm^2/ster)']):
            htemp = fits.Header()
            htemp['TYPE'] = h
            img_list.append(fits.ImageHDU(q, header=htemp))
        # Put everything together to write the table
        fnl_list  = [header] + par_table + img_list
        hdu_list  = fits.HDUList(hdus=fnl_list)
        hdu_list.writeto(outfile, overwrite=overwrite)
        return

    def read_from_table(self, infile):
        ff = fits.open(infile)
        # Load parameteric information
        lam   = ff[1].data['lam']
        unit  = ff[1].data.columns['lam'].unit
        a     = ff[2].data['a']
        theta = ff[3].data['theta']
        self.pars = {'lam':lam, 'a':a, 'unit':unit, 'theta':theta}

        # Load extinction information
        qvals = dict()
        for i in range(4,8):  # runs on hdus 4,5,6,7
            htype = ff[i].header['TYPE']
            qvals[htype] = ff[i].data
        self.qext = qvals['Qext']
        self.qabs = qvals['Qabs']
        self.qsca = qvals['Qsca']
        self.diff = qvals['Diff-xsect (cm^2/ster)']
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
