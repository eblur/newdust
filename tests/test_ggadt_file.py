import pytest
import subprocess
from astropy.io import fits
import astropy.units as u
import numpy as np

from newdust.scatteringmodel import make_ggadt as ggadt
from newdust.scatteringmodel import make_ggadt_astrodust as astro


#make FITS files
#make_fits(shape, material, orientation, folder, last_index, outfile, overwrite=True)
gg_1 = ggadt.make_fits('oblate', 'lepidocrocite', 'rand', '../newdust/scatteringmodel/tables/ggadt_rand_oblate', 4, 'gg_1.fits')

gg_2 = ggadt.make_fits('prolate', 'fayalite', 'set', '../newdust/scatteringmodel/tables/ggadt_set_prolate', 3, 'gg_2.fits')

#make_astrodust_fits(shapes, material, orientations, folder, last_index)
ast = astro.make_astrodust_fits(['sphere', 'oblate'], 'hematite', ['set', 'rand'], '../newdust/scatteringmodel/tables/astrodust_hematite', 9)

def test_headers():
        assert(check_header('gg_1.fits', 'oblate', 'lepidocrocite', 1.4, 'rand') == True)
        assert(check_header('gg_2.fits', 'prolate', 'fayalite', 2.0, 'set'))
        assert(check_header('hematite_oblate_rand.fits', 'oblate', 'hematite', 1.4, 'rand') == True)
        assert(check_header('hematite_oblate_set.fits', 'oblate', 'hematite', 1.4, 'set') == True)
        assert(check_header('hematite_sphere_rand.fits', 'sphere', 'hematite', 1.0, 'rand') == True)
        assert(check_header('hematite_sphere_set.fits', 'sphere', 'hematite', 1.0, 'set') == True)
        return

@pytest.mark.parametrize('f', ['gg_1.fits', 'gg_2.fits', 'hematite_oblate_rand.fits', 'hematite_oblate_set.fits', 'hematite_sphere_rand.fits', 'hematite_sphere_set.fits'])
def test_radii(f):
    assert (check_radii(f))

@pytest.mark.parametrize('f', ['gg_1.fits', 'gg_2.fits', 'hematite_oblate_rand.fits', 'hematite_oblate_set.fits', 'hematite_sphere_rand.fits', 'hematite_sphere_set.fits'])
def test_ephots(f):
    assert(check_ephots(f))

@pytest.mark.parametrize('f', ['gg_1.fits', 'gg_2.fits', 'hematite_oblate_rand.fits', 'hematite_oblate_set.fits', 'hematite_sphere_rand.fits', 'hematite_sphere_set.fits'])
def test_data(f):
    assert(check_data(f) == True)

#This isn't actually a test, just a way to remove the test files
def test_remove_files():
    command = 'rm gg_1.fits gg_2.fits hematite_oblate_rand.fits hematite_oblate_set.fits hematite_sphere_rand.fits hematite_sphere_set.fits'

    subprocess.run(command, shell=True)

    assert(True)

def check_header(fits_file, shape, material, ax_ratio, orient):
        file = fits.open(fits_file)
        header = file[0].header

        shape = header['SHAPE'] == shape
        material = header['MATERIAL'] == material
        ax_ratio = header['AX_RATIO'] == ax_ratio
        orient = header['ORIENT'] == orient
        
        file.close()
        if (shape and material and ax_ratio and orient):
            return True
        else:
            return False

def check_radii(file):
    radii = []
    if file == 'gg_1.fits':
        radii = [0.01, 0.34, 0.67, 1.0]
    elif file == 'gg_2.fits':
        radii = [0.01, 0.505, 1.0]
    elif file == 'hematite_oblate_rand.fits' or file == 'hematite_oblate_set.fits':
        radii = [0.34, 0.45, 0.56, 0.67, 0.78, 0.89, 1.0]
    elif file == 'hematite_sphere_rand.fits':
        radii = [0.01, 0.12, 0.23]

    file = fits.open(file)
    a = file[2].data.field(0)

    file.close()
    unit = file[2].header['TUNIT1']
    return (np.array_equal(np.array(radii), a) and unit == u.micron.to_string())

def check_ephots(file):
    f = fits.open(file)
    e = f[1].data.field(0)
    unit = f[1].header['TUNIT1']

    if (file == 'hematite_sphere_set.fits'):
        return (np.array_equal(e, np.array([])) and unit == u.keV.to_string())

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
            test_file = f'../newdust/scatteringmodel/tables/ggadt_rand_oblate/lepidocrocite_{i}_oblate_rand.out'
            result = check_vals(test_file, f, i)
            i += 1

    elif file == 'gg_2.fits':
        while i < 3:
            test_file = f'../newdust/scatteringmodel/tables/ggadt_set_prolate/fayalite_{i}_prolate_set.out'
            result = check_vals(test_file, f, i)
            i += 1
    
    elif file == 'hematite_sphere_rand.fits':
        while i < 3:
            test_file = f'../newdust/scatteringmodel/tables/astrodust_hematite/hematite_{i}_sphere_rand.out'
            result = check_vals(test_file, f, i)
            i += 1

    elif file == 'hematite_oblate_rand.fits':
        i = 3
        while i < 10:
            test_file = f'../newdust/scatteringmodel/tables/astrodust_hematite/hematite_{i}_oblate_rand.out'
            result = check_vals(test_file, f, i - 3)
            i += 1
        
    elif file == 'hematite_oblate_set.fits':
        i = 3
        while i < 10:
            test_file = f'../newdust/scatteringmodel/tables/astrodust_hematite/hematite_{i}_oblate_set.out'
            result = check_vals(test_file, f, i - 3)
            i += 1
    
    else:
        ext = np.array_equal(f[4].data, np.array([]))
        abs = np.array_equal(f[5].data, np.array([]))
        sca = np.array_equal(f[6].data, np.array([]))

        if (ext and abs and sca):
            result = True
        else:
            result = False
    
    f.close()
    return result

def check_vals(test_file, fits_file, i):
    #access output files with numpy
    vals = np.loadtxt(test_file, dtype=np.dtype([('E', np.float_), ('qsca', np.float_), ('qabs', np.float_), ('qext', np.float_)]))
    ext = np.array_equal(fits_file[4].data[i], vals['qext'])
    abs = np.array_equal(fits_file[5].data[i], vals['qabs'])
    sca = np.array_equal(fits_file[6].data[i], vals['qsca'])

    if (ext and abs and sca):
        return True
    else:
        return False