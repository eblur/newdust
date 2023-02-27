import numpy as np
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.integrate import trapz
from astropy.io import fits
import astropy.units as u

from .. import helpers

__all__ = ['Halo']

ALLOWED_FTYPE = ['abs','ext']
ALLOWED_FUNIT = ['cgs','phot','count','none']

class Halo(object):
    """
    An X-ray scattering halo.

    Sliceable object: can input a range of energy/wavelength values and it will 
    return the halo values for the range of interest. 
    Can also summon values according using integers within range(len(lam))

    Attributes
    ----------
    
    description : string
        Text description of this scattering halo.

    lam : astropy.units.Quantity -or- numpy.ndarray
        Wavelength or energy values for performing the calculation; 
        if no units specified, defaults to keV
        
    theta : astropy.units.Quantity -or- numpy.ndarray 
        Scattering angles for computing the differential scattering cross-section;
        if no units specified, defaults to ARCCSEC
    
    norm_int : astropy.units.Quantity array [arcsec^-2]
        Normalized scattering halo intensity as a function of energy (NE x NTH)
    
    taux : numpy.ndarray (NE)
        Integrated X-ray scattering opacity as a function of energy
    
    fabs : numpy.ndarray or astropy.units.Quantity (NE)
        Bin-integrated absorbed flux (photon/cm^2/s -or- erg/cm^2/s), 
        used for calculating total halo intensity
    
    intensity : astropy.units.Quantity (NTH) [arcsec^-2 x fabs.unit]
        Energy-integrated intensity calculated for the scattering halo [fabs x norm_int]
        
    fhalo : numpy.ndarray or astropy.units.Quantity (NE) [fabs.unit]
        Theoretical bin-integrated flux spectrum for the scattering
        halo. This is computed using the formula Fabs * (1 - exp(-taux))

    """
    def __init__(self, lam=1.0, theta=1.0, from_file=None):
        self.description = None
        if isinstance(lam, u.Quantity):
            self.lam = lam
        else:
            self.lam = lam * u.keV    # length NE
        if isinstance(theta, u.Quantity):
            self.theta = theta
        else:
            self.theta = theta * u.arcsec  # length NTH, arcsec
        self.norm_int  = None   # NE x NTH, arcsec^-2
        self.taux      = None   # NE, unitless
        self.fabs      = None   # NE, bin integrated flux [e.g. phot/cm^2/s, NOT phot/cm^2/s/keV]
        self.intensity = None   # NTH, flux x arcsec^-2
        if from_file is not None:
            self._read_from_file(from_file)

    def calculate_intensity(self, flux, ftype='abs'):
        """
        Calculate the scattering halo intensity from a flux spectrum.

        flux : numpy.ndarray or astropy.units.Quantity (NE)
            Bin-integrated absorbed flux (photon/cm^2/s -or- erg/cm^2/s), 
            used for calculating total halo intensity
        
        ftype : string ('abs' or 'ext')
            Describe whether the input spectrum is absorbed flux or 
            point source component flux (after including scattering component of extinction)
        """
        assert self.norm_int is not None
        assert ftype in ALLOWED_FTYPE
        if ftype == 'abs':
            fabs = flux
        if ftype == 'ext':
            fabs = flux * np.exp(self.taux) # Fps = Fabs exp(-tau_sca)
        self.fabs = fabs
        NE, NTH = np.shape(self.norm_int)
        fa_grid = np.repeat(fabs.reshape(NE,1), NTH, axis=1)
        self.intensity = np.sum(self.norm_int * fa_grid, axis=0)

    @property
    def fext(self):
        """
        Returns the point-source flux component, Fps = Fabs exp(-tau_sca)
        """
        assert self.fabs is not None
        return self.fabs * np.exp(-self.taux)

    @property
    def fhalo(self):
        """
        Returns the total scattring halo flux, Fh = Fabs (1 - exp(-tau_sca))
        """
        assert self.fabs is not None
        return self.fabs * (1.0 - np.exp(-self.taux))

    @property
    def percent_fabs(self):
        """
        Calculate fraction of absorbed flux that goes into the scattring halo,
        effectively (1 - exp(-tau))
        """
        assert self.fabs is not None
        return np.sum(self.fhalo) / np.sum(self.fabs)

    @property
    def percent_fext(self):
        """
        Calculate fraction of point source flux that goes into the scattring halo,
        effectively (1 - exp(-tau)) / exp(-tau) = exp(tau) - 1
        """
        assert self.fabs is not None
        return np.sum(self.fhalo) / np.sum(self.fext)

    # http://stackoverflow.com/questions/2936863/python-implementing-slicing-in-getitem
    def __getitem__(self, key):
        """
        Returns a Halo object that is a subset / slice of the original halo.

        Slice values must be floats that follow the same units as the original `lam` value.
        """
        if isinstance(key, int):
            return self._get_lam_index(key)
        if isinstance(key, slice):
            lmin = key.start
            lmax = key.stop
            if lmin is None:
                ii = (self.lam.value < lmax)
            elif lmax is None:
                ii = (self.lam.value >= lmin)
            else:
                ii = (self.lam.value >= lmin) & (self.lam.value < lmax)
            return self._get_lam_slice(ii)

    def _get_lam_slice(self, ii):
        result = Halo(self.lam.value[ii] * self.lam.unit, self.theta)
        if self.norm_int is not None:
            result.description = self.description
            result.norm_int = self.norm_int[ii,...]
            result.taux     = self.taux[ii]
        if self.fabs is not None:
            result.calculate_intensity(self.fabs[ii], ftype='abs')
        return result

    def _get_lam_index(self, i):
        assert isinstance(i, int)
        result = Halo(self.lam.value[i] * self.lam.unit, self.theta)
        if self.norm_int is not None:
            result.description = self.description
            result.norm_int = np.array([self.norm_int[i,...]])
            result.taux     = self.taux[i]
        if self.fabs is not None:
            flux = np.array([self.fabs[i]])
            result.calculate_intensity(flux, ftype='abs')
        return result

    def ecf(self, th, n, log=False):
        """
        Return the fraction of (energy-integrated) scattering halo flux enclosed by theta < th

        Inputs
        ------

        th0 : astropy.units.Quantity -or- float:
            Maximum theta value for calculating enclosed fraction;
            if no unit specified, ARCSEC is assumed
        
        n : int : number of theta values to use for inteprolating

        log : bool : If True, uses a log-spaced theta grid for the integral;
            If False, uses a linear spaced theta grid.

        Returns
        -------
        float : trapz(self.intensity * 2 pi theta_arcsec, theta_arcsec)

        SMALL ANGLE SCATTERING IS ASSUMED!
        """
        # Will break if we attempt to run this without an intensity calculation
        assert self.intensity is not None

        if isinstance(th, u.Quantity):
            thmax = th.to('arcsec').value
        else:
            thmax = th * u.arcsec
        
        # Halo theta values, which will serve as grid
        th_asec = self.theta.to('arcsec').value

        # Make sure that the value of interest is within the calculated grid
        assert thmax > np.min(th_asec) & thmax < np.max(th_sec)

        # Get a new grid of intensity values over which to integrate
        if log:
            th_grid = np.logspace(np.log10(np.min(th_asec)), 
                                  np.log10(np.max(th_asec)), n)
        else:
            th_grid  = np.linspace(np.min(th_asec), np.max(th_asec), n)
        
        I_grid = np.interp(th_grid, th_asec, self.intensity)
        fh_tot = np.sum(self.fhalo) # total flux in the halo
        enclosed = trapz(I_grid * 2.0 * np.pi * th_grid, th_grid)
        return enclosed / fh_tot
    
    def frac_halo(self, th, n, log=False):
        """
        Calculate fraction of halo, as a function of energy, enclosed by theta < th

        Inputs
        ------

        th0 : astropy.units.Quantity -or- float:
            Maximum theta value for calculating enclosed fraction;
            if no unit specified, ARCSEC is assumed
        
        n : int : number of theta values to use for inteprolating

        log : bool : If True, uses a log-spaced theta grid for the integral;
            If False, uses a linear spaced theta grid.

        Returns
        -------
        float : trapz(self.intensity * 2 pi theta_arcsec, theta_arcsec)

        SMALL ANGLE SCATTERING IS ASSUMED!
        """
        # Will break if we attempt to run this without an intensity calculation
        assert self.norm_intensity is not None

        if isinstance(th, u.Quantity):
            thmax = th.to('arcsec').value
        else:
            thmax = th * u.arcsec
        
        # Halo theta values, which will serve as grid
        th_asec = self.theta.to('arcsec').value

        # Make sure that the value of interest is within the calculated grid
        assert thmax > np.min(th_asec) & thmax < np.max(th_sec)

        # Get a new grid of intensity values over which to integrate
        if log:
            th_grid = np.logspace(np.log10(np.min(th_asec)), 
                                  np.log10(np.max(th_asec)), n)
        else:
            th_grid  = np.linspace(np.min(th_asec), np.max(th_asec), n)
        
        result = []
        for i in range(len(self.lam)):
            I_grid = np.interp(th_grid, th_asec, self.norm_int[i,:])
            enclosed = trapz(I_grid * 2.0 * np.pi * th_grid, th_grid)
            result.append[enclosed]
        return np.array(result).flatten() # unitless, size NE


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
        hdr['INTUNIT'] = self.norm_int.unit.to_string()
        primary_hdu = fits.PrimaryHDU(self.norm_int.value, header=hdr)

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
             [fits.Column(name='lam', array=helpers._make_array(self.lam.value),
             format='E', unit=self.lam.unit.to_string())])
        c2 = fits.BinTableHDU.from_columns(
             [fits.Column(name='theta', array=helpers._make_array(self.theta.value),
             format='E', unit=self.theta.unit.to_string())])
        c3 = fits.BinTableHDU.from_columns(
             [fits.Column(name='taux', array=helpers._make_array(self.taux),
             format='E', unit='')])
        return [c1, c2, c3]

    def _read_from_file(self, filename):
        """Read a Halo object from a FITS file"""
        ff = fits.open(filename)
        # Load the normalized intensity
        self.norm_int = ff[0].data * u.Unit(ff[0].header['INTUNIT'])
        # Load the other parameters
        self.lam = ff[1].data['lam'] * u.Unit(ff[1].columns['lam'].unit)
        self.theta = ff[2].data['theta'] * u.Unit(ff[2].columns['theta'].unit)
        self.taux = ff[3].data['taux']
        # Set halo type
        self.description = filename

    ##------ Make a fake image with a telescope arf
    def fake_image(self, arf, src_flux, exposure,
                   pix_scale=0.5, num_pix=[2400,2400],
                   lmin=None, lmax=None, save_file=None, **kwargs):
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

        lmin : float
            Minimum halo.lam value
            (Default:None uses entire range)

        lmax : float
            Maximum halo.lam value
            (Default:None uses entire range)

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
        if lmin is None:
            imin = 0
        else:
            imin = min(np.arange(len(self.lam))[self.lam >= lmin])
        if lmax is None:
            iend = len(self.lam)
        else:
            iend = max(np.arange(len(self.lam))[self.lam <= lmax])

        #iend = imax
        #if imax < 0:
        #    iend = np.arange(len(self.lam)+1)[imax]

        # Add up randomized image for each energy index value
        r_asec = radius * pix_scale
        result = np.zeros_like(radius)
        for i in np.arange(imin, iend):
            # interp object for halo grid (arcsec, counts/arcsec^2)
            h_interp = InterpolatedUnivariateSpline(
                self.theta, self.norm_int[i,:] * src_counts[i], k=1) # counts/arcsec^2
            # Get the corresponding counts at each radial value in the grid
            pix_counts = h_interp(r_asec) * pix_scale**2  # counts per pixel
            # Some of the interpolated values are below zero. This is not okay. Set those to zero.
            pix_counts[pix_counts < 0.0] = 0.0
            # Use poisson statistics to get a random value
            pix_random = np.random.poisson(pix_counts)
            # add it to the final result
            result += pix_random

        if save_file is not None:
            hdu  = fits.PrimaryHDU(result)
            hdul = fits.HDUList([hdu])
            hdul.writeto(save_file, overwrite=True)

        return result

    #----- Simulate a scattering halo and save as SIMPUT format -----#
    def make_simput(self, fabs, simput_file, spectrum_file, 
                    ra0=0.0, dec0=0.0, exposure=1.e5*u.second, effective_area=1.e4*u.cm**2, mjd=53005.0):
        """
        fabs : Astropy Quantity [ct/cm^2/s] : bin-integrated photon flux spectrum 
        
        simput_file : string : SIMPUT file name
        
        spectrum_file : string : spectrum file name
        
        ra0 : float : image center RA [deg]
        
        dec0 : float : image center DEC [deg]
        
        exposure : Astropy Quantity [time] : Exposure time for generating photon list,
        needs to be large to generate a statistically significant number of photons
        (Default: 100 ks)
        
        effective_area : Astropy Quantity [area] : Effective area for generating photon list,
        needs to be large to generate a statistically significant number of photons
        (Default: 1 m^2)
        
        mjd : float : Nominal reference point for TIME information. Default is 01/01/2004
        (Does not need to change unless concerned about time-variable halos.)

        Returns
        -------
        Writes the top-level FITS files simput_file and spectrum_file
        """
        assert len(fabs) == len(self.lam) # quick check
        
        # simulates photons for a single halo energy
        def pick_photons(nn, theta, profile):
            """
            nsrc : int : number of photons from the central point source
            theta : np.array : observation angles for surface brightness profile (arcsec)
            profile : np.array : normalized surface brightness profile [arcsec^-2]
            
            Returns
            -------
            rad : float : angular radius (arcsec) away from point source center
            phi : float : azimuthal angle (radians) for the photon
            n : int : total number of photons simulated for that scattering halo
            """
            # Cumulative distribution that will be used to select photons
            dtheta = theta[1:] - theta[:-1]
            dtheta = np.append(dtheta, dtheta[-1])
            total  = np.sum(profile * 2.0 * np.pi * theta * dtheta)
            cumulative_fraction = np.cumsum(profile * 2.0 * np.pi * theta * dtheta) / total
        
            # number of photons in the halo image
            n = int(nn * total)
            # pick a random radius (observation angle) and phi (azimuthal angle)
            phi  = np.random.uniform(low=0.0, high=1.0, size=n) * 2.0 * np.pi # radians
            rad  = np.interp(np.random.uniform(low=0.0, high=1.0, size=n), 
                         cumulative_fraction, theta) # arcsec
            return (rad, phi, n)

        ## ---- Make the photon list
        
        # How many photons are in the halo image?
        nphotons_src = (fabs * effective_area * exposure).to('ct').value
        photon_rad, photon_phi, photon_en  = [], [], []
        tot_energy_flux = 0.0 * u.erg # Total energy flux in the simulated halo
        nrunning = 0 # running total of photons simulated
        for i in range(len(self.lam)):
            #print(self.lam[i], nphotons_src[i])
            nsrc = int(nphotons_src[i])
            rad, phi, nhalo = pick_photons(nsrc, 
                                self.theta.to('arcsec').value, 
                                self.norm_int[i,:].to('arcsec^-2').value)
            photon_rad += list(rad)
            photon_phi += list(phi)
            photon_en  += ([self.lam[i].to('keV', equivalencies=u.spectral()).value] * len(rad))
            tot_energy_flux += nhalo * self.lam[i]
            nrunning += nhalo
        photon_en = np.array(photon_en)
        print("Total number of photons simulated: {}".format(nrunning))
        
        # Now we have a master list of photons with an r(theta) and phi
        # We need to translate that into sky coordinates
        ra  = ra0 + np.array(photon_rad) * u.arcsec.to('deg') * np.cos(np.array(photon_phi))
        dec = dec0 + np.array(photon_rad) * u.arcsec.to('deg') * np.sin(np.array(photon_phi))
        
        # Scale the energy flux value by exposure and effective area
        tot_energy_flux = tot_energy_flux / exposure / effective_area # erg/s/cm^2
    
        ## -- done making the photon list. Functions below are for saving to SIMPUT files
        
        def write_spectrum_file(filename, mjd=mjd):
            ## Make the header
            header = fits.Header()
            header['HDUCLASS'] = 'HEASARC/SIMPUT'
            header['HDUCLAS1'] = 'PHOTONS'
            header['HDUVERS']  = '1.1.0'
            header['RADESYS']  = 'FK5'
            header['EQUINOX']  = '2000.0'
            header['REFRA']    = ra0
            header['REFDEC']   = dec0
            # Time specification
            # https://heasarc.gsfc.nasa.gov/docs/heasarc/ofwg/docs/rates/ogip_93_003/ogip_93_003.html#tth_sEc4.2
            header['MJDREF']   = mjd
            header['TSTART']   = 0.0
            header['TSTOP']    = exposure.to('second').value
            header['TIMEZERO'] = 0.0
            header['TIMESYS']  = 'MJD'
            header['TIMEUNIT'] = 's'
            header['CLOCKCOR'] = 'NO'
            primary = fits.PrimaryHDU(header=header)
        
            ## Make the binary table
            ra_col      = fits.Column(name='RA', array=ra, format='E', unit='deg')
            dec_col     = fits.Column(name='DEC', array=dec, format='E', unit='deg')
            ener_col    = fits.Column(name='ENERGY', array=photon_en, format='E', unit='keV')
            photon_list = fits.BinTableHDU.from_columns([ra_col, dec_col, ener_col], name='PHOTONS')
            hdu_list    = fits.HDUList([primary, photon_list])
            hdu_list.writeto(filename, overwrite=True)
            print('Wrote spectrum SIMPUT file to {}'.format(filename))
            return
    
        def write_simput_file(filename, specfile):
            # Make the header
            header = fits.Header()
            header['HDUCLASS'] = 'HEASARC/SIMPUT'
            header['HDUCLAS1'] = 'SRC_CAT'
            header['HDUVERS']  = '1.1.0'
            header['RADESYS']  = 'FK5'
            header['EQUINOX']  = '2000.0'
            hdu_list = fits.BinTableHDU.from_columns([ \
                    fits.Column(name='SRC_ID', array=[1], format='K', unit=''),
                    fits.Column(name='SRC_NAME', array=['scattering halo'], format='20A', unit=''),
                    fits.Column(name='RA', array=[ra0], format='E', unit='deg'),
                    fits.Column(name='DEC', array=[dec0], format='E', unit='deg'),
                    fits.Column(name='E_MIN', array=[min(self.lam.to('keV', equivalencies=u.spectral()).value)], 
                                format='E', unit='keV'),
                    fits.Column(name='E_MAX', array=[max(self.lam.to('keV', equivalencies=u.spectral()).value)], 
                                format='E', unit='keV'),
                    fits.Column(name='FLUX', array=[tot_energy_flux.to('erg s^-1 cm^-2').value], 
                                format='E', unit='erg s^-1 cm^-2'),
                    fits.Column(name='SPECTRUM', array=[specfile], format='20A', unit='')],
                    name='SRC_CAT')
            hdu_list.writeto(filename, overwrite=True)
            print('Wrote top level SIMPUT file: {}'.format(filename))
            return
    
        ## --- Write the photon list to a SIMPUT file
        
        write_spectrum_file(spectrum_file)
        write_simput_file(simput_file, spectrum_file)
        return
