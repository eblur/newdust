
import os
from astropy.io import ascii
import astropy.units as u
from newdust.graindist.composition import Composition

## TO USE THE ASTRODUST OPTICAL CONSTANTS:
## The user must set an environment variable, ASTRODUST, to point to the directory
## where the files of optical constants from Draine & Hensley 2021
## (Filename: index_DH21Ad_Pporo_fFe_ba) are kept
astrodust_dir = os.environ['ASTRODUST']

__all__ = ['CmAstrodust']

RHO_AD = 2.74 # g cm^-3
# grain material density for best fitting astrodust model from Hensley & Draine 2022
# has a porosity P = 0.2; use RHO = 3.42 (1 - P) g cm^-3, according to their work.

# It is left to the user to select the proper density for the file they are using.
# By default, P=0.2, fFe=0, and b/a=0.62
# The choice of fFe and b/a have a <5% effect on the X-ray extinction properties,
# while porosity can change the cross-sections by a factor of 2

class CmAstrodust(Composition):
    """
    Optical constants for Astrodust from Draine & Hensley (2021)
    """
    def __init__(self, rho=RHO_AD, from_file='index_DH21Ad_P0.20_0.00_0.625'):
        """
        Read in Astrodust optical constants from Draine & Hensley (2021)

        IMPORTANT: Set an environment variable ASTRODUST to point to the directory 
        containing the optical data

        INPUTS
        ------
        rho : float : grain material density in g cm^-3

        from_file : string : file containing the optical constants you want
            
            default: 0 porosity, 0 Fe inclusions, b/a=0.625 (prolate)

            P = porosity (vacuum fraction),

            f_Fe = fraction of the Fe in metallic form,

            axial ratio b/a (b/a < 1 for prolate spheroids, b/a > 1 for oblate spheroids).

            prolate spheroids are football shaped
           
            oblate spheroids are M&M or lentil shaped

            File names are of the format index_DH21Ad_Pporo_fFe_ba

            where poro =  0.00,  0.10,  0.20,  0.30,  0.40
                  fFe  =  0.00,  0.10
                  ba   = 0.333, 0.400, 0.500, 0.625, 1.400, 1.600, 2.000, 3.000

            Note: for fFe = 0.10, there is only data for ba = 0.500 and 2.000  
        """
        Composition.__init__(self)
        self.cmtype = 'Astrodust'
        self.rho    = rho
        self.citation = "Using optical constants for Astrodust\nDraine & Hensley 2021, ApJ, 909, 94\nhttps://ui.adsabs.harvard.edu/abs/2021ApJ...909...94D/abstract"
        self.file = from_file

        DH21file = f"{astrodust_dir}/{from_file}"
        DH21data = ascii.read(DH21file, header_start=1, data_start=2)

        self.wavel = DH21data['E(eV)'] * u.eV # the data is in ascending order
        self.revals = 1.0 + DH21data['Re(n)-1']
        self.imvals = DH21data['Im(n)']