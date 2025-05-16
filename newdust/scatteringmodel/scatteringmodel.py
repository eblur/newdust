import astropy.units as u
from astropy.io import fits
from .. import helpers

__all__ = ['ScatteringModel']

## See __init__ for API
class ScatteringModel(object):
    """
    Parent class for scattering models

    Attributes
    ----------
    qsca : numpy.ndarray : Scattering efficiency Q (cross-section / geometric cross-section)

    qext : numpy.ndarray : Extinction efficiency Q (cross-section / geometric cross-section)    

    qabs : numpy.ndarray : Absorption efficiency Q (cross-section / geometric cross-section)

    diff : numpy.ndarray : Differential scattering efficiency per steridian

    pars : dict : Parameters from most recent calculation are stored here

    stype : string : A label for the model

    citation : string : A description of how to cite this model
    """
    def __init__(self, from_file=None):
        """
        Inputs
        ------
        from_file : string : Optional, string that one can use to load the scattering model
        """
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
    def calculate(self, lam, a, cm, theta, **kwargs):
        """
        Calculate the extinction efficiences with the Rayleigh-Gans approximation.

        lam : astropy.units.Quantity -or- numpy.ndarray
            Wavelength or energy values for calculating the cross-sections;
            if no units specified, defaults to keV
        
        a : astropy.units.Quantity -or- numpy.ndarray
            Grain radius value(s) to use in the calculation;
            if no units specified, defaults to micron
        
        cm : newdust.graindist.composition object
            Holds the optical constants and density for the compound.
        
        theta : astropy.units.Quantity -or- numpy.ndarray -or- float
            Scattering angles for computing the differential scattering cross-section;
            if no units specified, defaults to radian

        Returns None.
        """
        return None

    def _store_parameters(self, lam, a, cm, theta):
        """
        Parses parameter units and stores them.

         lam : astropy.units.Quantity -or- numpy.ndarray
            Wavelength or energy values for calculating the cross-sections;
            if no units specified, defaults to keV
        
        a : astropy.units.Quantity -or- numpy.ndarray
            Grain radius value(s) to use in the calculation;
            if no units specified, defaults to micron
        
        cm : newdust.graindist.composition object
            Holds the optical constants and density for the compound.
        
        theta : astropy.units.Quantity -or- numpy.ndarray -or- float
            Scattering angles for computing the differential scattering cross-section;
            if no units specified, defaults to radian
        
        Returns
        -------
        A three element tuple:
        |   `lam` in units of keV, 
        |   `a` in units of cm, and 
        |   `theta` in units of radians
        """
        # Store the parameters
        self.pars = dict()

        lam_cm = None
        if isinstance(lam, u.Quantity):
            self.pars['lam'] = lam
            lam_cm = lam.to('cm', equivalencies=u.spectral()).value
        else:       
            self.pars['lam'] = lam * u.keV
            lam_cm = (lam * u.keV).to('cm', equivalencies=u.spectral()).value
        
        # Save the value as microns, but return in cgs units
        a_cm = None
        if isinstance(a, u.Quantity):
            # Store `a` value as microns
            self.pars['a'] = a
            a_cm = a.to('cm').value
        else:
            self.pars['a'] = a * u.micron
            a_cm = (a * u.micron).to('cm').value
        
        self.pars['cm'] = cm.cmtype

        theta_rad = None
        if isinstance(theta, u.Quantity):
            self.pars['theta'] = theta
            theta_rad = theta.to('radian').value
        else:
            self.pars['theta'] = theta * u.radian
            theta_rad = theta

        return lam_cm, a_cm, theta_rad
        

    def write_table(self, outfile, overwrite=True):
        """
        Write the current scattering model calculation to a FITS file

        outfile : string : Name of output file

        overwrite : bool (True) : if True, will overwrite a file of the same name
        """
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
                          ['Qext', 'Qabs', 'Qsca', 'Diff-xsect (ster^-1)']):
            htemp = fits.Header()
            htemp['TYPE'] = h
            img_list.append(fits.ImageHDU(q, header=htemp))
        # Put everything together to write the table
        fnl_list  = [header] + par_table + img_list
        hdu_list  = fits.HDUList(hdus=fnl_list)
        hdu_list.writeto(outfile, overwrite=overwrite)
        return

    def read_from_table(self, infile):
        """
        Reads in a previous scattering model calculation from a FITS file

        Inputs
        ------

        infile : string : Name of the input file
        """
        ff = fits.open(infile)
        # Load parameteric information
        lam   = ff[1].data['lam'] * u.Unit(ff[1].header['TUNIT1'])
        a     = ff[2].data['a'] * u.Unit(ff[2].header['TUNIT1'])
        theta = ff[3].data['theta'] * u.Unit(ff[3].header['TUNIT1'])
        self.pars = {'lam':lam, 'a':a, 'theta':theta}

        # Load extinction information
        qvals = dict()
        for i in range(4,8):  # runs on hdus 4,5,6,7
            htype = ff[i].header['TYPE']
            qvals[htype] = ff[i].data
        self.qext = qvals['Qext']
        self.qabs = qvals['Qabs']
        self.qsca = qvals['Qsca']
        self.diff = qvals['Diff-xsect (ster^-1)']
        return

    ##----- Helper material
    def _write_table_header(self):
        """ Writes FITS file header based on self.pars """
        result = fits.Header()
        result['COMMENT']  = "Extinction efficiency and differential cross-sections"
        result['COMMENT']  = "HDUS 4-6 are Qext, Qsca, Qabs in wavelength (or energy) vs grain radius"
        result['COMMENT']  = "HDU 7 is the differential scattering cross-section (ster^-1)"
        return fits.PrimaryHDU(header=result)

    def _write_table_pars(self):
        """writes table lam and a parameters from self.pars"""
        # e.g. pars['lam'], pars['a']
        # should this be part of WCS?
        c1 = fits.BinTableHDU.from_columns(
             [fits.Column(name='lam', array=helpers._make_array(self.pars['lam'].value),
             format='D', unit=self.pars['lam'].unit.to_string())])
        c2 = fits.BinTableHDU.from_columns(
             [fits.Column(name='a', array=helpers._make_array(self.pars['a'].value),
             format='D', unit=self.pars['a'].unit.to_string())])
        c3 = fits.BinTableHDU.from_columns(
             [fits.Column(name='theta', array=helpers._make_array(self.pars['theta'].value),
             format='D', unit=self.pars['theta'].unit.to_string())])
        return [c1, c2, c3]
