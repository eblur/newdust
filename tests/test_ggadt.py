'''
Tests are made from the files in the ggadt_test_files folder

Checks that need to be made:
1. shape is correct
2. orientation is correct
3. axis ratio is correct (if oblate)
4. radii are correct/can be accessed without worrying about tuple form
5. material is correct
6. values are correct
7. units are correct
'''

from ggadt import make_fits
import numpy as np
from astropy.io import fits
import astropy.units as u

def check_vals(test_file, fits_file, i):
    #access output files with numpy
    vals = np.loadtxt(test_file, dtype=np.dtype([('E', np.float_), ('qsca', np.float_), ('qabs', np.float_), ('qext', np.float_)]))
    assert(np.array_equal(fits_file[1].data.field(0), vals['E']))
    assert(fits_file[1].header['TUNIT1'] == u.keV.to_string())
    assert(np.array_equal(fits_file[4].data[i], vals['qext']))
    assert(np.array_equal(fits_file[5].data[i], vals['qabs']))
    assert(np.array_equal(fits_file[6].data[i], vals['qsca']))

    return

#make fits parameters (in order): shape, material, folder, num_files, outfile

#rand oblate:
make_fits('oblate', 'iron_sulfate_anhydrous', 'ggadt_test_files/rand_oblate', 2, 'rand_oblate_test')

file = fits.open('rand_oblate_test')

print('testing oblate rand... checking header data\n')
head = file[0].header
assert(head['SHAPE'] == 'oblate')
assert(head['MATERIAL' == 'iron_sulfate_anhydrous'])
assert(head['AX_RATIO'] == 1.4)
assert(head['ORIENT'] == 'Random')
print('header looks good... checking radii\n')
radii = np.array([0.01, 1])
assert(np.array_equal(file[2].data.field(0), radii))
assert(file[2].header['TUNIT1'] == u.micron.to_string())
print('radii look good... checking extinction vals\n')

i = 0
while i < 2:
    print('checking file ' + str(i) + ' extinction vals')
    test_file = 'ggadt_test_files/rand_oblate/iron_sulfate_anhydrous_' + str(i) + '_oblate.out'
    check_vals(test_file, file, i)

    i += 1
print('vals look good, rand_oblate_test passes!\n')
file.close()

#rand sphere:
make_fits('sphere', 'lepidocricite', 'ggadt_test_files/rand_sphere', 3, 'rand_sphere_test') #I realize I mistyped lepidocrocite!
file = fits.open('rand_sphere_test')
print('testing sphere rand... checking header data')

head = file[0].header
assert(head['SHAPE'] == 'sphere')
assert(head['MATERIAL' == 'lepidocrocite'])
assert(head['AX_RATIO'] == 1.0)
assert(head['ORIENT'] == 'Random')
print('header looks good... checking radii\n')
radii = np.array([0.01, 0.505, 1.0])
assert(np.array_equal(file[2].data.field(0), radii))
assert(file[2].header['TUNIT1'] == u.micron.to_string())
print('radii look good... checking extinction vals\n')

i = 0
while i < 3:
    print('checking file ' + str(i) + ' extinction vals')
    test_file = 'ggadt_test_files/rand_sphere/lepidocricite_' + str(i) + '_sphere.out'
    check_vals(test_file, file, i)

    i += 1
print('vals look good, rand_sphere_test passes!\n')
file.close()

#set_oblate
make_fits('oblate', 'hematite', 'ggadt_test_files/set_oblate', 5, 'set_oblate_test')
file = fits.open('set_oblate_test')
print('testing oblate set... checking header data')

head = file[0].header
assert(head['SHAPE'] == 'oblate')
assert(head['MATERIAL' == 'hematite'])
assert(head['AX_RATIO'] == 2.0)
assert(head['ORIENT'] == 'Set')
print('header looks good... checking radii\n')

radii = np.array([0.01, 0.2575, 0.505, 0.7525, 1.0])
assert(np.array_equal(file[2].data.field(0), radii))

assert(file[2].header['TUNIT1'] == u.micron.to_string())
print('radii look good... checking extinction vals\n')

i = 0
while i < 5:
    print('checking file ' + str(i) + ' extinction vals')
    test_file = 'ggadt_test_files/set_oblate/hematite_' + str(i) + '_oblate.out'
    check_vals(test_file, file, i)

    i += 1
print('vals look good, set_oblate_test passes!\n')
file.close()

#set sphere:
make_fits('sphere', 'fayalite', 'ggadt_test_files/set_sphere', 4, 'set_sphere_test')
file = fits.open('set_sphere_test')
print('testing sphere set... checking header data')

head = file[0].header
assert(head['SHAPE'] == 'sphere')
assert(head['MATERIAL' == 'fayalite'])
assert(head['AX_RATIO'] == 1.0)
assert(head['ORIENT'] == 'Set')
print('header looks good... checking radii\n')

radii = np.array([0.01, 0.34, 0.67, 1.0])
assert(np.array_equal(file[2].data.field(0), radii))
assert(file[2].header['TUNIT1'] == u.micron.to_string())
print('radii look good... checking extinction vals\n')

i = 0
while i < 4:
    print('checking file ' + str(i) + ' extinction vals')
    test_file = 'ggadt_test_files/set_sphere/fayalite_' + str(i) + '_sphere.out'
    check_vals(test_file, file, i)

    i += 1
print('vals look good, set_sphere_test passes!\n')
file.close()

    




