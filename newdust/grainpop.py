import numpy as np
from scipy.integrate import trapz
from astropy.io import fits

from . import graindist
from . import scatmodels
from . import constants as c

__all__ = ['SingleGrainPop','GrainPop','make_MRN','make_MRN_drude']

MD_DEFAULT    = 1.e-4  # g cm^-2
AMIN, AMAX, P = 0.005, 0.3, 3.5  # um, um, unitless
RHO_AVG       = 3.0  # g cm^-3
UNIT_LABELS   = {'kev':'Energy (keV)', 'angs':'Wavelength (angs)'}

# Make this a subclass of GrainDist at some point
class SingleGrainPop(graindist.GrainDist):
    """
    | **ATTRIBUTES**
    | lam, lam_unit, tau_ext, tau_sca, tau_abs, diff, int_diff
    | *Inherited from the GrainDist object*
    | a, ndens, mdens, cgeo, vol
    |
    | *functions*
    | calculate_ext(lam, unit='kev', **kwargs) runs the extinction calculation on the wavelength grid specified by lam and unit
    | plot_sizes(ax, **kwargs) plots the size distribution (see *astrodust.graindist.sizedist*)
    | plot_ext(ax, keyword, **kwargs) plots the extinction properties (see *astrodust.extinction*)
    |   - ``keyword`` options are "ext", "sca", "abs", "all"
    | info() prints information about the dust grain properties
    | write_extinction_table(outfile, **kwargs) writes extinction table
    |    (qext, qabs, qsca, and diff-xsect) for the calculated scattering properties
    """
    def __init__(self, dtype, cmtype, stype, shape='Sphere', md=MD_DEFAULT, scatm_from_file=None, **kwargs):
        graindist.GrainDist.__init__(self, dtype, cmtype, shape=shape, md=md, **kwargs)

        self.lam      = None  # NE
        self.lam_unit = None  # string
        self.tau_sca  = None  # NE
        self.tau_abs  = None  # NE
        self.tau_ext  = None  # NE
        self.diff     = None  # NE x NA x NTH [cm^2 ster^-1]
        self.int_diff = None  # NE x NTH [arcsec^2], differential xsect integrated over grain size

        # Handling scattering model FITS input, if requested
        if scatm_from_file is not None:
            self.scatm = scatmodels.ScatModel(from_file=scatm_from_file)
            assert isinstance(stype, str)
            self.scatm.stype = stype
            self.lam = self.scatm.pars['lam']
            self.lam_unit = self.scatm.pars['unit']
            self._calculate_tau()
        # Otherwise choose from existing (or custom) scattering calculators
        elif isinstance(stype, str):
            self._assign_scatm_from_string(stype)
        else:
            self.scatm = stype

    def _assign_scatm_from_string(self, stype):
        assert stype in ['RG', 'Mie']
        if stype == 'RG':
            self.scatm = scatmodels.RGscat()
        if stype == 'Mie':
            self.scatm = scatmodels.Mie()

    # Run scattering model calculation, then compute optical depths
    def calculate_ext(self, lam, unit='kev', theta=0.0, **kwargs):
        self.scatm.calculate(lam, self.a, self.comp, unit=unit, theta=theta, **kwargs)
        self.lam      = lam
        self.lam_unit = unit
        self._calculate_tau()

    # Compute optical depths only
    def _calculate_tau(self):
        NE, NA, NTH = np.shape(self.scatm.diff)
        # In single size grain case
        if len(self.a) == 1:
            self.tau_ext = self.ndens * self.scatm.qext[:,0] * self.cgeo
            self.tau_sca = self.ndens * self.scatm.qsca[:,0] * self.cgeo
            self.tau_abs = self.ndens * self.scatm.qabs[:,0] * self.cgeo
        # Otherwise, integrate over grain size (axis=1)
        else:
            geo_fac = self.ndens * self.cgeo  # array of length NA, unit is um^-1
            geo_2d  = np.repeat(geo_fac.reshape(1, NA), NE, axis=0)  # NE x NA
            self.tau_ext = trapz(geo_2d * self.scatm.qext, self.a, axis=1)
            self.tau_sca = trapz(geo_2d * self.scatm.qsca, self.a, axis=1)
            self.tau_abs = trapz(geo_2d * self.scatm.qabs, self.a, axis=1)

        # NE x NA x NTH
        area_2d = np.repeat(self.cgeo.reshape(1, NA), NE, axis=0) # cm^2
        area_3d = np.repeat(area_2d.reshape(NE, NA, 1), NTH, axis=2)
        self.diff     = self.scatm.diff * area_3d # NE x NA x NTH, [cm^2 ster^-1]

        if np.size(self.a) == 1:
            int_diff = np.sum(self.scatm.diff * self.cgeo[0] * self.ndens[0] * c.arcs2rad**2, axis=1)
        else:
            agrid        = np.repeat(
                np.repeat(self.a.reshape(1, NA, 1), NE, axis=0),
                NTH, axis=2)
            ndgrid       = np.repeat(
                np.repeat(self.ndens.reshape(1, NA, 1), NE, axis=0),
                NTH, axis=2)
            int_diff = trapz(self.scatm.diff * area_3d * ndgrid, agrid, axis=1) * c.arcs2rad**2

        self.int_diff = int_diff  # NE x NTH, [arcsec^-2]

    # Plotting things
    def plot_sdist(self, ax, **kwargs):
        self.plot(ax, **kwargs)

    def plot_ext(self, ax, keyword, **kwargs):
        assert keyword in ['ext','sca','abs','all']
        try:
            assert self.lam is not None
        except:
            print("Need to run calculate_ext")
            pass
        if keyword == 'ext':
            ax.plot(self.lam, self.tau_ext, **kwargs)
            ax.set_xlabel(UNIT_LABELS[self.lam_unit])
            ax.set_ylabel(r"$\tau_{ext}$")
        if keyword == 'sca':
            ax.plot(self.lam, self.tau_sca, **kwargs)
            ax.set_xlabel(UNIT_LABELS[self.lam_unit])
            ax.set_ylabel(r"$\tau_{sca}$")
        if keyword == 'abs':
            ax.plot(self.lam, self.tau_abs, **kwargs)
            ax.set_xlabel(UNIT_LABELS[self.lam_unit])
            ax.set_ylabel(r"$\tau_{abs}$")
        if keyword == 'all':
            ax.plot(self.lam, self.tau_ext, 'k-', lw=2, label='Extinction')
            ax.plot(self.lam, self.tau_sca, 'r--', label='Scattering')
            ax.plot(self.lam, self.tau_abs, 'r:', label='Absorption')
            ax.set_xlabel(UNIT_LABELS[self.lam_unit])
            ax.set_ylabel(r"$\tau$")
            ax.legend(**kwargs)

    # Printing information
    def info(self):
        print("Size distribution: %s" % self.size.dtype)
        print("Extinction calculated with: %s" % self.scatm.stype)
        print("Grain composition: %s" % self.comp.cmtype)
        print("rho = %.2f g cm^-3, M_d = %.2e g cm^-2" % (self.rho, self.md))

    # Write an extinction table
    def write_extinction_table(self, outfile, **kwargs):
        self.scatm.write_table(outfile, **kwargs)
        return

class GrainPop(object):
    """
    | A collection of dust grain distributions (SingeGrainPop).
    | Can add a string describing this Grain population using the `description` keyword
    |
    | **ATTRIBUTES**
    | keys     : A list of keys corresponding to each SingleGrainPop (default: list of integers starting with 0)
    | gpoplist : A list of SingleGrainPop objects
    | description : A string describing this collection
    | lam      : The energy / wavelength used for calculating extinction
    | lam_unit : The unit for energy ('kev') or wavelength ('angs') used for calculating extinction
    |
    | *properties*
    | tau_ext : Total extinction optical depth as a function of wavelength / energy
    | tau_sca : Total scattering optical depth as a function of wavelength / energy
    | tau_abs : Total absorption optical depth as a function of wavelength / energy
    |
    | *functions*
    | __getitem__(key) will return the SingleGrainPop indexed by ``key``
    | calculate_ext(lam, unit='kev', **kwargs) runs the extinction calculation on the wavelength grid specified by lam and unit
    | plot_ext(ax, keyword, **kwargs) plots the extinction properties (see *astrodust.extinction*)
    |   - ``keyword`` options are "ext", "sca", "abs", "all"
    | info(key=None) prints information about the SingleGrainPop indexed by ``key``
    |   - if ``key`` is *None*, information about every grain population will be printed to screen
    """
    def __init__(self, gpoplist, keys=None, description='Custom_GrainPopDict'):
        assert isinstance(gpoplist, list)
        if keys is None:
            self.keys = list(range(len(gpoplist)))
        else:
            self.keys = keys
        self.description = description
        self.gpoplist    = gpoplist
        for k in self.keys:
            i = self.keys.index(k)
            self.gpoplist[i].description = str(self.keys[i])
        self.lam = None
        self.lam_unit = None

    def calculate_ext(self, lam, unit='kev', **kwargs):
        for gp in self.gpoplist:
            gp.calculate_ext(lam, unit=unit, **kwargs)
        self.lam = lam
        self.lam_unit = unit

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.gpoplist[key]
        else:
            assert key in self.keys
            k = self.keys.index(key)
            return self.gpoplist[k]

    @property
    def md(self):
        result = 0.0
        for gp in self.gpoplist:
            result += gp.md
        return result

    @property
    def tau_ext(self):
        result = 0.0
        if self.lam is None:
            print("ERROR: Extinction properties need to be calculated")
        else:
            for gp in self.gpoplist:
                result += gp.tau_ext
        return result

    @property
    def tau_sca(self):
        result = 0.0
        if self.lam is None:
            print("ERROR: Extinction properties need to be calculated")
        else:
            for gp in self.gpoplist:
                result += gp.tau_sca
        return result

    @property
    def tau_abs(self):
        result = 0.0
        if self.lam is None:
            print("ERROR: Extinction properties need to be calculated")
        else:
            for gp in self.gpoplist:
                result += gp.tau_abs
        return result

    def plot_ext(self, ax, keyword, **kwargs):
        assert keyword in ['all','ext','sca','abs']
        if keyword == 'ext':
            ax.plot(self.lam, self.tau_ext, **kwargs)
            ax.set_xlabel(UNIT_LABELS[self.lam_unit])
            ax.set_ylabel(r"$\tau_{ext}$")
        if keyword == 'sca':
            ax.plot(self.lam, self.tau_sca, **kwargs)
            ax.set_xlabel(UNIT_LABELS[self.lam_unit])
            ax.set_ylabel(r"$\tau_{sca}$")
        if keyword == 'abs':
            ax.plot(self.lam, self.tau_abs, **kwargs)
            ax.set_xlabel(UNIT_LABELS[self.lam_unit])
            ax.set_ylabel(r"$\tau_{abs}$")
        if keyword == 'all':
            ax.plot(self.lam, self.tau_ext, 'k-', lw=2, label='Extinction')
            ax.plot(self.lam, self.tau_sca, 'r--', label='Scattering')
            ax.plot(self.lam, self.tau_abs, 'r:', label='Absorption')
            ax.set_xlabel(UNIT_LABELS[self.lam_unit])
            ax.set_ylabel(r"$\tau$")
            ax.set_title(self.description)
            ax.legend(**kwargs)

    def info(self, key=None):
        if key is None:
            print("General information for %s dust grain population" % self.description)
            for gp in self.gpoplist:
                print("---")
                gp.info()
        else:
            assert key in self.keys
            self[key].info()


#---------- Basic helper functions for fast production of GrainPop objects

def make_MRN(amin=AMIN, amax=AMAX, p=P, md=MD_DEFAULT, fsil=0.6, **kwargs):
    """
    | Returns a GrainPop describing an MRN dust grain size distribution, which is a mixture of silicate and graphite grains.
    | Applies the 1/3 parallel, 2/3 perpendicular assumption of graphite grain orientations.
    |
    | **INPUTS**
    | amin : minimum grain size in microns
    | amax : maximum grain size in microns
    | p    : power law slope for grain size distribution
    | md   : dust mass column [g cm^-2]
    | fsil : fraction of dust mass in silicate grains
    """
    assert isinstance(fsil, float)
    assert fsil >= 0.0 and fsil <= 1.0
    md_sil  = fsil * md
    # Graphite grain assumption: 1/3 parallel and 2/3 perpendicular
    md_gra_para = (1.0 - fsil) * md * (1.0/3.0)
    md_gra_perp = (1.0 - fsil) * md * (2.0/3.0)

    pl_sil  = graindist.sizedist.Powerlaw(amin=amin, amax=amax, p=p, **kwargs)
    pl_gra  = graindist.sizedist.Powerlaw(amin=amin, amax=amax, p=p, **kwargs)

    sil    = graindist.composition.CmSilicate()
    gra_ll = graindist.composition.CmGraphite(orient='para')
    gra_T  = graindist.composition.CmGraphite(orient='perp')

    mrn_sil = SingleGrainPop(pl_sil, sil, 'Mie', md=md_sil)
    mrn_gra_para = SingleGrainPop(pl_gra, gra_ll, 'Mie', md=md_gra_para)
    mrn_gra_perp = SingleGrainPop(pl_gra, gra_T, 'Mie', md=md_gra_perp)

    gplist = [mrn_sil, mrn_gra_para, mrn_gra_perp]
    keys   = ['sil','gra_para','gra_perp']
    return GrainPop(gplist, keys=keys, description='MRN')

def make_MRN_drude(amin=AMIN, amax=AMAX, p=P, rho=RHO_AVG, md=MD_DEFAULT, **kwargs):
    """
    | Returns a GrainPop describing an MRN dust grain size distribution, and uses the Drude approximation,
    | which approximates the dust grain as a sphere of free electrons
    |
    | **INPUTS**
    | amin : minimum grain size in microns
    | amax : maximum grain size in microns
    | p    : power law slope for grain size distribution
    | rho  : density of dust grain material [g cm^-3]
    | md   : dust mass column [g cm^-2]
    """
    pl      = graindist.sizedist.Powerlaw(amin=amin, amax=amax, p=p, **kwargs)
    dru     = graindist.composition.CmDrude()
    mrn_dru = SingleGrainPop(pl, dru, 'RG', md=md)
    gplist  = [mrn_dru]
    keys    = ['RGD']
    return GrainPop(gplist, keys=keys, description='MRN_rgd')
