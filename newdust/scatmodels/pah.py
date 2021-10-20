
## Created by Lia Corrales to parse PAH optical constant tables (PAHion_30, PAHneu_30)
## November 11, 2013 : lia@astro.columbia.edu
## November 13, 2020 : incorporated into newdust library, liac@umich.edu

import os
import numpy as np

import astropy.units as u
import astropy.constants as c

from newdust.graindist.composition import _find_cmfile
from .scatmodel import ScatModel

__all__ = ['PAH']

def parse_PAH( option, ignore='#', flag='>', verbose=False ):

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

class PAH(ScatModel):
    """
    | PAH properties loaded from Draine tables (public)
    !
    | **ATTRIBUTES**
    | pahtype  : string : 'ion' (ionized) or 'neu' (neutral)
    | stype : string : 'PAH' + type
    |
    | pars  : dict   : parameters used to run the calculation
    | qsca  : array  : scattering efficiency (unitless, per geometric area)
    | qabs  : array  : absorption efficiency (unitless, per geometric area)
    | qext  : array  : extinction efficiency (unitless, per geometric area)
    |
    | **FUNCTIONS**
    | calculate(lam, a=0.01, unit='keV')
    |     Calculates qsca, qabs, and qext attributes
    """
    def __init__(self, pahtype, **kwargs):
        ScatModel.__init__(self, **kwargs)
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
        self.pars = dict(zip(['lam', 'a', 'unit'],
                             [lam, a, unit]))

        NE, NA = np.size(lam), np.size(a)

        if NA == 1:
            qsca = self._get_Q(lam, 'Q_sca', a, lam_unit=unit)
            qabs = self._get_Q(lam, 'Q_abs', a, lam_unit=unit)
            qext = self._get_Q(lam, 'Q_ext', a, lam_unit=unit)
        else:
            qsca = np.zeros_like(NE, NA)
            qabs = np.zeros_like(NE, NA)
            qext = np.zeros_like(NE, NA)
            for (aa,i) in zip(a, range(NA)):
                qsca[:,i] = self._get_Q(lam, 'Q_sca', aa, lam_unit=unit)
                qabs[:,i] = self._get_Q(lam, 'Q_abs', aa, lam_unit=unit)
                qext[:,i] = self._get_Q(lam, 'Q_ext', aa, lam_unit=unit)

        self.qsca = qsca
        self.qabs = qabs
        self.qext = qext
        
    def _get_Q(self, lam, qtype, a, lam_unit='keV'):
        """
        lam : float
        qtype : string : Q_abs, Q_ext, Q_sca
        a : float : PAH size [micron]
        lam_unit : string : unit parsable by astropy.units
        """
        try:
            qvals = np.array(self.data[a][qtype] )
            wavel = np.array(self.data[a]['w(micron)'] )
        except:
            print('ERROR: Cannot get grain size', a, 'for', self.stype)
            return

        lam_um = (lam * u.Unit(lam_unit)).to('micron', equivalencies=u.spectral()).value
        # Wavelengths were listed in reverse order
        result = np.interp(lam_um, wavel[::-1], qvals[::-1] )
        return result
