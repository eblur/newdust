import numpy as np
import astropy.units as u
from .. import helpers
from .scatteringmodel import ScatteringModel

__all__ = ['Mie']

MAX_RAM = 8.0

# Major update: March 27, 2016
# This code is slow, so to avoid running getQs over and over again, create
# "calculate" function that runs getQs and stores it, then returns the appropriate values later??  Not sure ...

# Copied from ~/code/mie/bhmie_mod.pro
# ''Subroutine BHMIE is the Bohren-Huffman Mie scattering subroutine
#    to calculate scattering and absorption by a homogenous isotropic
#    sphere.''
#
class Mie(ScatteringModel):
    """
    Mie scattering algorithms of Bohren & Hoffman.
    See their book: *Absorption and Scattering of Light by Small Particles*

    Attributes
    ----------
    In addition to those inherited from ScatteringModel

    gsca : numpy.ndarray : average cosine of scattering angle

    qback : numpy.ndarray : back-scattering efficiency
    """
    def __init__(self, **kwargs):
        ScatteringModel.__init__(self, **kwargs)
        self.stype = 'Mie'
        self.citation = 'Mie scattering algorithm from Bohren & Hoffman\n*Absorption and Scattering of Light by Small Particles*'
        self.gsca  = None
        self.qback = None

    def calculate(self, lam, a, cm, theta=0.0, memlim=MAX_RAM):
        """
        Calculate the extinction efficiences using the Mie formula for spherical particles.

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
        # Store the parameters
        lam_cm0, a_cm0, theta_rad0 = self._store_parameters(lam, a, cm, theta)
        NE, NA, NTH = np.size(lam_cm0), np.size(a_cm0), np.size(theta_rad0)

        # Deal with the 1d stuff first
        # Make sure every variable is an array
        lam_cm_1d    = helpers._make_array(lam_cm0)
        a_cm_1d      = helpers._make_array(a_cm0)
        theta_rad_1d = helpers._make_array(theta_rad0)

        # Complex index of refraction
        refrel_1d = cm.cm(lam_cm_1d * u.cm)

        # Make everything NE x NA
        a_cm   = np.repeat(a_cm_1d.reshape(1, NA), NE, axis=0)
        lam_cm = np.repeat(lam_cm_1d.reshape(NE, 1), NA, axis=1)
        refrel = np.repeat(refrel_1d.reshape(NE, 1), NA, axis=1)
        # Size parameter (grain circumference to incoming wavelength)
        x      = 2.0 * np.pi * a_cm / lam_cm

        qsca, qext, qback, gsca, Cdiff = _mie_helper(x, refrel, theta=theta_rad_1d, memlim=memlim)

        self.qsca  = qsca  # NE x NA
        self.qext  = qext
        self.qabs  = self.qext - self.qsca
        self.qback = qback
        self.gsca  = gsca
        self.diff  = Cdiff # ster^-1,  NE x NA x NTH

#---------------- Helper function that does the actual calculation

def _mie_helper(x, refrel, theta, memlim=MAX_RAM):
    """
    theta is array of length NTH, units of radians
    
    x and refrel are NE x NA
    
    need to make outputs that are NE x NA x NTH
    """
    assert np.shape(x) == np.shape(refrel)
    assert len(np.shape(x)) <= 2
    assert len(theta) >= 1

    NE, NA = np.shape(x)
    NTH    = len(theta)

    # Make 3D array for the calculations on angular dependence
    theta_rad_3d  = np.repeat(
        np.repeat(theta.reshape(1, 1, NTH), NE, axis=0),
        NA, axis=1)
    x_3d      = np.repeat(x.reshape(NE,NA,1), NTH, axis=2)

    amu       = np.abs(np.cos(theta_rad_3d))
    indl90    = (theta_rad_3d < np.pi/2.0)
    indg90    = (theta_rad_3d >= np.pi/2.0)

    s1    = np.zeros(shape=(NE, NA, NTH), dtype='complex')
    s2    = np.zeros(shape=(NE, NA, NTH), dtype='complex')
    pi    = np.zeros(shape=(NE, NA, NTH), dtype='complex')
    pi0   = np.zeros(shape=(NE, NA, NTH), dtype='complex')
    pi1   = np.zeros(shape=(NE, NA, NTH), dtype='complex') + 1.0
    tau   = np.zeros(shape=(NE, NA, NTH), dtype='complex')

    y      = x * refrel
    ymod   = np.abs(y)
    nx     = len(x.flatten())  # total number of NE x NA

    # *** Series expansion terminated after NSTOP terms
    # Logarithmic derivatives calculated from NMX on down

    xstop  = x + 4.0 * np.power(x, 0.3333) + 2.0
    test   = np.append(xstop, ymod)
    nmx    = np.max(test) + 15
    nmx    = np.int64(nmx)  # max number of iterations

    nstop  = xstop
#   nmxx   = 150000

#    if (nmx > nmxx):
#        print('error: nmx > nmxx=', nmxx, ' for |m|x=', ymod)
    # *** Logarithmic derivative D(J) calculated by downward recurrence
    # beginning with initial value (0.,0.) at J=NMX

    ## Check memory usage
    ## If it is above the memory limit, stop the calculation
    dsize  = np.int64(NE * NA * nmx)
    dspace = _test_complex_mem_usage(dsize)  # GB
    if dspace > memlim:
        print("WARNING!! Space needed (%f GB) exceeds memory limit (%.2f GB)" % (dspace, memlim))
        print("WARNING!! Try increasing the memlim keyword to match available RAM, or decrease sampling")
        return (0.0, 0.0, 0.0, 0.0, 0.0)

    d = np.zeros(shape=(NE,NA,nmx+1), dtype='complex')  # NE x NA x nmx
    dold = np.zeros(nmx+1, dtype='complex')
    # Original code set size to nmxx.
    # I see that the array only needs to be slightly larger than nmx

    for n in range(1,nmx):  # for n=1, nmx-1 do begin
        en = nmx - n + 1
        #assert isinstance(en, int)
        #assert isinstance(nmx-n, int)
        #assert isinstance(nmx-n+1, int)
        d[...,nmx-n]  = (en/y) - (1.0 / (d[...,nmx-n+1]+en/y))

    # *** Riccati-Bessel functions with real argument X
    # calculated by upward recurrence

    psi0 = np.cos(x)  # NE x NA
    psi1 = np.sin(x)
    chi0 = -np.sin(x)
    chi1 = np.cos(x)
    xi1  = psi1 - 1j * chi1

    qsca = 0.0    # scattering efficiency
    gsca = 0.0    # <cos(theta)>

    s1_ext = 0
    s2_ext = 0
    s1_back = 0
    s2_back = 0

    pi_ext  = 0
    pi0_ext = 0
    pi1_ext = 1
    tau_ext = 0

    p    = -1.0

    for n in range(1,int(np.max(nstop))+1):  # for n=1, nstop do begin
        assert isinstance(n, int)
        en = n
        fn = (2.0*en+1.0) / (en * (en+1.0))

        # for given N, PSI  = psi_n        CHI  = chi_n
        #              PSI1 = psi_{n-1}    CHI1 = chi_{n-1}
        #              PSI0 = psi_{n-2}    CHI0 = chi_{n-2}
        # Calculate psi_n and chi_n
        # *** Compute AN and BN:

        #*** Store previous values of AN and BN for use
        #    in computation of g=<cos(theta)>
        if n > 1:
            an1 = an
            bn1 = bn

        ig  = (nstop >= n)

        psi    = np.zeros(shape=(NE,NA))
        chi    = np.zeros(shape=(NE,NA))

        psi[ig] = (2.0*en-1.0) * psi1[ig]/x[ig] - psi0[ig]
        chi[ig] = (2.0*en-1.0) * chi1[ig]/x[ig] - chi0[ig]
        xi      = psi - 1j * chi

        an = np.zeros(shape=(NE,NA), dtype='complex')
        bn = np.zeros(shape=(NE,NA), dtype='complex')

        d_n = d[...,n]
        an[ig] = (d_n[ig] / refrel[ig] + en / x[ig]) * psi[ig] - psi1[ig]
        an[ig] = an[ig] / ((d_n[ig] / refrel[ig] + en / x[ig]) * xi[ig] - xi1[ig])
        bn[ig] = (refrel[ig] * d_n[ig] + en / x[ig]) * psi[ig] - psi1[ig]
        bn[ig] = bn[ig] / ((refrel[ig] * d_n[ig] + en/x[ig]) * xi[ig] - xi1[ig])

        an_3d = np.repeat(an.reshape(NE,NA,1), NTH, axis=2)
        bn_3d = np.repeat(bn.reshape(NE,NA,1), NTH, axis=2)

        # *** Augment sums for Qsca and g=<cos(theta)>
        # NOTE from LIA: In IDL version, bhmie casts double(an)
        # and double(bn).  This disgards the imaginary part.  To
        # avoid type casting errors, I use an.real and bn.real
        # Because animag and bnimag were intended to isolate the
        # real from imaginary parts, I replaced all instances of
        # double( foo * complex(0.d0,-1.d0) ) with foo.imag

        qsca   = qsca + (2.0 * en + 1.0) * (np.power(np.abs(an),2) + np.power(np.abs(bn),2))
        gsca   = gsca + ((2.0 * en + 1.0) / (en * (en + 1.0))) * (an.real * bn.real + an.imag * bn.imag)

        if n > 1:
            gsca    = gsca + ((en-1.0) * (en+1.0)/en) * \
                (an1.real*an.real + an1.imag*an.imag + bn1.real*bn.real + bn1.imag*bn.imag)

        # *** Now calculate scattering intensity pattern
        #     First do angles from 0 to 90

        # LIA : Altered the two loops below so that only the indices where ang
        # < 90 are used.  Replaced (j) with [indl90]

        # Note also: If theta is specified, and np.size(E) > 1,
        # the number of E values must match the number of theta
        # values.  Cosmological halo functions will utilize this
        # Diff this way.

        pi  = pi1
        tau = en * amu * pi - (en + 1.0) * pi0

        if np.size(indl90) != 0:
            antmp = an_3d[...,indl90]
            bntmp = bn_3d[...,indl90]  # For case where multiple E and theta are specified
            s1[...,indl90] = s1[...,indl90] + fn * (antmp * pi[...,indl90] + bntmp * tau[...,indl90])
            s2[...,indl90] = s2[...,indl90] + fn * (antmp * tau[...,indl90] + bntmp * pi[...,indl90])
        #ENDIF

        pi_ext = pi1_ext
        tau_ext = en * 1.0 * pi_ext - (en + 1.0) * pi0_ext

        s1_ext = s1_ext + fn * (an * pi_ext + bn * tau_ext)
        s2_ext = s2_ext + fn * (bn * pi_ext + an * tau_ext)

        # *** Now do angles greater than 90 using PI and TAU from
        #     angles less than 90.
        #     P=1 for N=1,3,...; P=-1 for N=2,4,...

        p = -p

        # LIA : Previous code used tau(j) from the previous loop.  How do I
        # get around this?

        if np.size(indg90) != 0:
            antmp = an_3d[...,indg90]
            bntmp = bn_3d[...,indg90]
            s1[...,indg90] = s1[...,indg90] + fn * p * (antmp * pi[...,indg90] - bntmp * tau[...,indg90])
            s2[...,indg90] = s2[...,indg90] + fn * p * (bntmp * pi[...,indg90] - antmp * tau[...,indg90])
        #ENDIF

        s1_back = s1_back + fn * p * (an * pi_ext - bn * tau_ext)
        s2_back = s2_back + fn * p * (bn * pi_ext - an * tau_ext)

        psi0 = psi1
        psi1 = psi
        chi0 = chi1
        chi1 = chi
        xi1  = psi1 - 1j * chi1

        # *** Compute pi_n for next value of n
        #     For each angle J, compute pi_n+1
        #     from PI = pi_n , PI0 = pi_n-1

        pi1  = ((2.0 * en + 1.0) * amu * pi - (en + 1.0) * pi0) / en
        pi0  = pi

        pi1_ext = ((2.0 * en + 1.0) * 1.0 * pi_ext - (en + 1.0) * pi0_ext) / en
        pi0_ext = pi_ext

        # ENDFOR

    # *** Have summed sufficient terms.
    #     Now compute QSCA,QEXT,QBACK,and GSCA
    gsca = 2.0 * gsca / qsca
    qsca = (2.0 / np.power(x,2)) * qsca

    # LIA : Changed qext to use s1(theta=0) instead of s1(1).  Why did the
    # original code use s1(1)?

    qext = (4.0 / np.power(x,2)) * s1_ext.real
    qback = np.power(np.abs(s1_back)/x, 2) / np.pi

    Cdiff = 0.0
    bad_theta = (np.abs(theta_rad_3d) > np.pi)  # Set to 0 values where theta > !pi
    s1[...,bad_theta] = 0
    s2[...,bad_theta] = 0
    Cdiff = 0.5 * (np.power(np.abs(s1), 2) + np.power(np.abs(s2), 2)) / (np.pi * np.power(x_3d,2))

    result = (qsca, qext, qback, gsca, Cdiff)
    return result

##------ GENERAL HELPER FUNCTION

def _test_complex_mem_usage(num):
    ## Test memory usage for an array of complex numbers of length numbers
    ## Returns size of complex number array in units of GB
    # 64 bit numbers are 8 bytes each
    # Assume 64 bit, times two (real and imaginary part), so each complex number is 16 bytes
    bytes_per_num = 16.   # bytes per complex number
    GB_to_bytes   = 1.e9  # bytes per GB
    total_mem_GB  = num * bytes_per_num / GB_to_bytes
    return total_mem_GB
