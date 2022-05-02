
## Created by Lia Corrales to parse PAH optical constant tables (PAHion_30, PAHneu_30)
## November 11, 2013 : lia@astro.columbia.edu
## November 13, 2020 : incorporated into newdust library, liac@umich.edu

import os
import numpy as np

import astropy.units as u

from newdust.graindist.composition import _find_cmfile
from .scatteringmodel import ScatteringModel

__all__ = ['PAH']

def parse_PAH( option, ignore='#', flag='>', verbose=False ):
    """
    Function for parsing the PAH files from Draine
    """
    filename = None
    if option == 'ion': filename = _find_cmfile('PAHion_30')
    if option == 'neu': filename = _find_cmfile('PAHneu_30')
    if verbose: print("Opening {}".format(filename))

    try : f = open( filename, 'r' )
    except:
        print('ERROR: file not found')
        return

    COLS = ['w(micron)', 'Q_ext', 'Q_abs', 'Q_sca', 'g=<cos>' ]
    result = {}

    end_of_file = False
    while not end_of_file:
        try:
            line = f.readline()

            # Ignore the ignore character
            if line[0] == ignore : pass

            # Characters flagged with '>' earn a dictionary entry with grain size
            elif line[0] == flag :
                if verbose: print("Found a new grain size table")
                gsize = np.float( line.split()[1] )
                if verbose : print('Reading data for grain size:', gsize)
                result[ gsize ] = {}
                # Initialize dictionaries with lists
                for i in range( len(COLS) ) : result[gsize][COLS[i]] = []

            # Sort the columns into the correct dictionary
            else:
                row_vals = line.split()
                for i in range( len(COLS) ) :
                    result[ gsize ][ COLS[i] ].append( np.float( row_vals[i] ) )
        except:
            if verbose : print(line)
            end_of_file = True

    f.close()

    return result

class PAH(ScatteringModel):
    """
    PAH properties loaded from Draine tables (public on B. Draine's website). 
    *See* Li & Draine (2001)

    Attributes
    ----------
    In addition to those inherited from ScatteringModel

    pahtype  : string : 'ion' (ionized) or 'neu' (neutral)
    
    stype : string : 'PAH' + type

    No differential scattering cross-section calculation.
    """
    def __init__(self, pahtype, **kwargs):
        ScatteringModel.__init__(self, **kwargs)
        self.citation = "PAH files from Li & Draine (2001), ApJ, 554, 778\nhttps://ui.adsabs.harvard.edu/abs/2001ApJ...554..778L"
        self.pahtype  = pahtype
        self.stype = 'PAH' + pahtype
        self.data  = None
        self.pars  = None
        self.qsca  = None
        self.qabs  = None
        self.qext  = None

        try:
            self.data = parse_PAH(pahtype)
        except:
            print('ERROR: Cannot find PAH type', self.type)

    def calculate(self, lam, a=0.01, unit='keV'):
        """
        Get the extinction efficiences by interpolating the tables from Li & Draine (2001).

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

        Updates the `qsca`, `qext`, `qabs`, `diff`, `gsca`, and `qback` attributes
        """
        # Parameters are not stored the same with this moel type
        self.pars = dict()
        
        wavel_um = None
        if isinstance(lam, u.Quantity):
            self.pars['lam'] = lam.value
            self.pars['unit'] = lam.unit.to_string()
            wavel_um = lam.to('micron', equivalencies=u.spectral()).value
        else:
            self.pars['lam'] = lam
            self.pars['unit'] = 'keV'
            wavel_um = (lam * u.keV).to('micron', equivalencies=u.spectral()).value
        
        a_um = None
        if isinstance(a, u.Quantity):
            a_um = a.to('micron').value
            self.pars['a'] = a_um
        else:
            self.pars['a'] = a
            a_um = a
        
        NE, NA = np.size(wavel_um), np.size(a_um)

        if NA == 1:
            qsca = self._get_Q(wavel_um, 'Q_sca', a_um)
            qabs = self._get_Q(wavel_um, 'Q_abs', a_um)
            qext = self._get_Q(wavel_um, 'Q_ext', a_um)
        else:
            qsca = np.zeros_like(NE, NA)
            qabs = np.zeros_like(NE, NA)
            qext = np.zeros_like(NE, NA)
            for (aa,i) in zip(a_um, range(NA)):
                qsca[:,i] = self._get_Q(wavel_um, 'Q_sca', aa)
                qabs[:,i] = self._get_Q(wavel_um, 'Q_abs', aa)
                qext[:,i] = self._get_Q(wavel_um, 'Q_ext', aa)

        self.qsca = qsca
        self.qabs = qabs
        self.qext = qext
        
    def _get_Q(self, wavel_um, qtype, a, lam_unit='keV'):
        """
        wavel_um : numpy.ndarray : wavelength in microns

        qtype : string : 'Q_abs', 'Q_ext', 'Q_sca'
        
        a : float : PAH size [micron]
        """
        try:
            qvals = np.array(self.data[a][qtype] )
            wavel = np.array(self.data[a]['w(micron)'] )
        except:
            print('ERROR: Cannot get grain size', a, 'for', self.stype)
            return
        
        # Wavelengths in the Draine files were listed in reverse order
        result = np.interp(wavel_um, wavel[::-1], qvals[::-1] )
        return result
