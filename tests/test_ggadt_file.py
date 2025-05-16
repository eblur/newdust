import pytest
import subprocess
from astropy.io import fits
import astropy.units as u
import numpy as np

from newdust.scatteringmodel import make_ggadt as ggadt
from newdust.scatteringmodel import make_ggadt_astrodust as astro


#make FITS files
#make_fits(material, folder, indicies, outfile, overwrite=True)
ggadt.make_fits('lepidocrocite', '../newdust/scatteringmodel/tables/ggadt_rand_oblate', range(4), 'gg_1.fits')

ggadt.make_fits('fayalite', '../newdust/scatteringmodel/tables/ggadt_set_prolate', range(3), 'gg_2.fits')

#make_fits_astrodust(material, indicies, folder, outfile, overwrite=True)
astro.make_fits_astrodust('hematite', '../newdust/scatteringmodel/tables/astrodust_hematite', range(10), 'astro.fits')

def test_headers():
        assert(check_header('gg_1.fits', 'lepidocrocite',  'oblate', 1.4, 'random') == True)
        assert(check_header('gg_2.fits', 'fayalite', 'prolate', 2.0, 'set') == True)
        assert(check_header('astro.fits', 'hematite') == True)
        return

@pytest.mark.parametrize('f', ['gg_1.fits', 'gg_2.fits', 'astro.fits'])
def test_radii(f):
    assert (check_radii(f))

def test_check_astro_rec():
    file = fits.open('astro.fits')
    shapes = file[2].data.field(1)
    orientations = file[2].data.field(2)
    ratios = file[2].data.field(3)

    file.close()

    assert(np.array_equal(shapes, np.array(['sphere', 'sphere', 'sphere', 'oblate', 'oblate', 'oblate', 'oblate', 'oblate', 'oblate', 'oblate'])))

    assert(np.array_equal(orientations, np.array(['random', 'random', 'random', 'set', 'set', 'set', 'set', 'set', 'set', 'set'])))

    assert(np.array_equal(ratios, np.array([1.0, 1.0, 1.0, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4])))

@pytest.mark.parametrize('f', ['gg_1.fits', 'gg_2.fits', 'astro.fits'])
def test_ephots(f):
    assert(check_ephots(f))

@pytest.mark.parametrize('f', ['gg_1.fits', 'gg_2.fits', 'astro.fits'])
def test_data(f):
    assert(check_data(f) == True)

#This isn't actually a test, just a way to remove the test files
def test_remove_files():
    command = 'rm gg_1.fits gg_2.fits astro.fits'

    subprocess.run(command, shell=True)

    assert(True)

def check_header(fits_file, material, shape='', ax_ratio=0.0, orient=''):
        file = fits.open(fits_file)
        header = file[0].header

        material = header['MATERIAL'] == material

        if (fits_file != 'astro.fits'):
            shape = header['SHAPE'] == shape
            ax_ratio = header['AX_RATIO'] == ax_ratio
            orient = header['ORIENT'] == orient
        else:
            shape = True
            ax_ratio = True
            orient = True
        
        file.close()
        if (shape and material and ax_ratio and orient):
            return True
        else:
            return False

def check_radii(file):
    radii = [0.01, 0.12, 0.23, 0.34, 0.45, 0.56, 0.67, 0.78, 0.89, 1.0]
    if file == 'gg_1.fits':
        radii = [0.01, 0.34, 0.67, 1.0]
    elif file == 'gg_2.fits':
        radii = [0.01, 0.505, 1.0]

    file = fits.open(file)
    a = file[2].data.field(0)

    file.close()
    unit = file[2].header['TUNIT1']
    return (np.array_equal(np.array(radii), a) and unit == u.micron.to_string())

def check_ephots(file):
    f = fits.open(file)
    e = f[1].data.field(0)
    unit = f[1].header['TUNIT1']

    ephots = range(600, 800)
    evs = []
    for en in ephots:
        evs.append(round(en * 10**(-3), 3))

    f.close()
    return (np.array_equal(np.array(evs), e) and unit == u.keV.to_string())

def check_data(file):
    f = fits.open(file)
    i = 0
    result = ''

    if file == 'gg_1.fits':
        while i < 4:
            test_file = f'../newdust/scatteringmodel/tables/ggadt_rand_oblate/lepidocrocite_{i}.out'
            result = check_vals(test_file, f, i)
            i += 1

    elif file == 'gg_2.fits':
        while i < 3:
            test_file = f'../newdust/scatteringmodel/tables/ggadt_set_prolate/fayalite_{i}.out'
            result = check_vals(test_file, f, i)
            i += 1
    
    else:
        while i < 10:
            test_file = f'../newdust/scatteringmodel/tables/astrodust_hematite/hematite_{i}.out'
            result = check_vals(test_file, f, i)
            i += 1
    
    f.close()
    return result

def check_vals(test_file, fits_file, i):
    #access output files with numpy
    vals = np.loadtxt(test_file, dtype=np.dtype([('D', np.float64), ('qsca', np.float64), ('qabs', np.float64), ('qext', np.float64)]))
    qext = fits_file[4].data.transpose()
    qabs = fits_file[5].data.transpose()
    qsca = fits_file[6].data.transpose()
    ext = np.array_equal(qext[i], vals['qext'])
    abs = np.array_equal(qabs[i], vals['qabs'])
    sca = np.array_equal(qsca[i], vals['qsca'])

    if (ext and abs and sca):
        return True
    else:
        return False
