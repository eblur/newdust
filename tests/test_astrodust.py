'''
Tests for astrodust.py using the data from the astrodust model

This could probably be made more extensive, but the only thing really changing in the astrodust files is the material
'''

import numpy as np
from astrodust import make_fits
from astropy.io import fits

#make_fits(material, data_folder, min_set_index, num_radii, outfile, overwrite=True)

make_fits('fayalite', 'astrodust_test_files', 11, 32, 'astrodust_test')
file = fits.open('astrodust_test')

print('Checking header info...')
curr = file[0].header
assert(curr['MATERIAL'] == 'fayalite')
assert(curr['MAX_I'] == 54 - 1 + 3)

print('Header looks good... checking params')
curr = file[1]
rs = np.logspace(0.01, 1.0, num=32) #a_eff will be log10(vals[i])
radii = []
#Because some radii have set and random orientations, there are two radii -- this starts at index 11 in the test file
i = 0
while i < len(rs):
    if i <= 9:
        radii.append(round(np.log10(rs[i]), 5))
    else:
        radii.append(round(np.log10(rs[i]), 5))
        radii.append(round(np.log10(rs[i]), 5))
    i += 1
assert(len(radii) == 54) #check for radii list
assert(np.allclose(curr.data.field(0), np.array(radii), 10**(-2), 10**(-4)))
print('Radii look good... checking energies')

#can pull energies from just one file to check
vals = np.loadtxt('astrodust_test_files/fayalite/fayalite_1_rand.out', dtype=np.dtype([('E', np.float_), ('qsca', np.float_), ('qabs', np.float_), ('qext', np.float_)]))

curr = file[2]
#print(curr.data.field(0), vals['E'])
assert(np.allclose(curr.data.field(0), vals['E'], 10**(-2), 10**(-4)))
print('Energies look good... now testing image files')

#Need to check each image HDU
#files 1 - 10 (only random oritenation)
i = 1
while i < 11:
    ggadt_file = 'astrodust_test_files/fayalite/fayalite_' + str(i) + '_rand.out'
    fits_index = 3 + i - 1
    vals = np.loadtxt(ggadt_file, dtype=np.dtype([('E', np.float_), ('qsca', np.float_), ('qabs', np.float_), ('qext', np.float_)]))
    curr = file[fits_index]
    
    assert(not curr.header['SET'])
    assert(curr.header['SHAPE'] == 'sphere')
    assert(curr.header['AXS_RAT'] == 1.0)
    assert(np.allclose(curr.data[0], vals['qext'], 10**(-2), 10**(-4)))
    assert(np.allclose(curr.data[1], vals['qsca'], 10**(-2), 10**(-4)))
    assert(np.allclose(curr.data[2], vals['qabs'], 10**(-2), 10**(-4)))
    i += 1

print('images 1-10 pass...testing final images')
#final tests
i = 11
fits_index = 3 + i - 1
while i < 33:
    #tests for rand
    ggadt_file = 'astrodust_test_files/fayalite/fayalite_' + str(i) + '_rand.out'
    vals = np.loadtxt(ggadt_file, dtype=np.dtype([('E', np.float_), ('qsca', np.float_), ('qabs', np.float_), ('qext', np.float_)]))
    curr = file[fits_index]

    assert(not curr.header['SET'])
    assert(curr.header['SHAPE'] == 'oblate')
    assert(curr.header['AXS_RAT'] == 1.4)
    assert(np.allclose(curr.data[0], vals['qext'], 10**(-2), 10**(-4)))
    assert(np.allclose(curr.data[1], vals['qsca'], 10**(-2), 10**(-4)))
    assert(np.allclose(curr.data[2], vals['qabs'], 10**(-2), 10**(-4)))

    #tests for set
    fits_index += 1
    ggadt_file = 'astrodust_test_files/fayalite/fayalite_' + str(i) + '_set.out'
    vals = np.loadtxt(ggadt_file, dtype=np.dtype([('E', np.float_), ('qsca', np.float_), ('qabs', np.float_), ('qext', np.float_)]))
    curr = file[fits_index]

    assert(curr.header['SET'])
    assert(curr.header['SHAPE'] == 'oblate')
    assert(curr.header['AXS_RAT'] == 1.4)
    assert(np.allclose(curr.data[0], vals['qext'], 10**(-2), 10**(-4)))
    assert(np.allclose(curr.data[1], vals['qsca'], 10**(-2), 10**(-4)))
    assert(np.allclose(curr.data[2], vals['qabs'], 10**(-2), 10**(-4)))


    fits_index += 1
    i += 1

print('all tests passed!')

file.close()
#tests for only random orientations
#so min_set_index > num_radii
make_fits('metallic_iron', 'astrodust_test_files', 33, 32, 'astrodust_test')
file = fits.open('astrodust_test')

print('Checking header info...')
curr = file[0].header
assert(curr['MATERIAL'] == 'metallic_iron')
assert(curr['MAX_I'] == 32 - 1 + 3)

print('Header looks good... checking params')
curr = file[1]
rs = np.logspace(0.01, 1.0, num=32) #a_eff will be log10(vals[i])
radii = []
#Because some radii have set and random orientations, there are two radii -- this starts at index 11 in the test file
i = 0
while i < len(rs):
    radii.append(round(np.log10(rs[i]), 5))
    i += 1
assert(len(radii) == 32) #check for radii list
assert(np.allclose(curr.data.field(0), np.array(radii), 10**(-2), 10**(-4)))
print('Radii look good... checking energies')

#can pull energies from just one file to check
vals = np.loadtxt('astrodust_test_files/metallic_iron/metallic_iron_1_rand.out', dtype=np.dtype([('E', np.float_), ('qsca', np.float_), ('qabs', np.float_), ('qext', np.float_)]))

curr = file[2]
#print(curr.data.field(0), vals['E'])
assert(np.allclose(curr.data.field(0), vals['E'], 10**(-2), 10**(-4)))
print('Energies look good... now testing image files')

#all files are only random in this case
i = i
while i < 32:
    ggadt_file = 'astrodust_test_files/metallic_iron/metallic_iron_' + str(i) + '_rand.out'
    fits_index = 3 + i - 1
    vals = np.loadtxt(ggadt_file, dtype=np.dtype([('E', np.float_), ('qsca', np.float_), ('qabs', np.float_), ('qext', np.float_)]))
    curr = file[fits_index]
    
    assert(not curr.header['SET'])
    if (curr.header['RADIUS'] < 0.3):
        assert(curr.header['SHAPE'] == 'sphere')
        assert(curr.header['AXS_RAT'] == 1.0)
    else:
        assert(curr.header['SHAPE'] == 'oblate')
        assert(curr.header['AXS_RAT'] == 1.4)
    assert(np.allclose(curr.data[0], vals['qext'], 10**(-2), 10**(-4)))
    assert(np.allclose(curr.data[1], vals['qsca'], 10**(-2), 10**(-4)))
    assert(np.allclose(curr.data[2], vals['qabs'], 10**(-2), 10**(-4)))
    i += 1

file.close()
#hematite tests have rand and set files for every radius
make_fits('hematite', 'astrodust_test_files', 1, 32, 'astrodust_test')
file = fits.open('astrodust_test')

print('Checking header info...')
curr = file[0].header
assert(curr['MATERIAL'] == 'hematite')
assert(curr['MAX_I'] == 64 - 1 + 3)

print('Header looks good... checking params')
curr = file[1]
rs = np.logspace(0.01, 1.0, num=32) #a_eff will be log10(vals[i])
radii = []
#Because some radii have set and random orientations, there are two radii -- this starts at index 11 in the test file
i = 0
while i < len(rs):
    radii.append(round(np.log10(rs[i]), 5))
    radii.append(round(np.log10(rs[i]), 5))
    i += 1
assert(len(radii) == 64) #check for radii list
assert(np.allclose(curr.data.field(0), np.array(radii), 10**(-2), 10**(-4)))
print('Radii look good... checking energies')

#can pull energies from just one file to check
vals = np.loadtxt('astrodust_test_files/hematite/hematite_1_rand.out', dtype=np.dtype([('E', np.float_), ('qsca', np.float_), ('qabs', np.float_), ('qext', np.float_)]))

curr = file[2]
#print(curr.data.field(0), vals['E'])
assert(np.allclose(curr.data.field(0), vals['E'], 10**(-2), 10**(-4)))
print('Energies look good... now testing image files')

#all files have set and random in this case
#all files are only random in this case
i = 1
fits_index = 3 + i - 1
while i < 32:
    #rand check -- rand comes first
    ggadt_file = 'astrodust_test_files/hematite/hematite_' + str(i) + '_rand.out'
    vals = np.loadtxt(ggadt_file, dtype=np.dtype([('E', np.float_), ('qsca', np.float_), ('qabs', np.float_), ('qext', np.float_)]))
    curr = file[fits_index]
    
    assert(not curr.header['SET'])
    if (curr.header['RADIUS'] < 0.3):
        assert(curr.header['SHAPE'] == 'sphere')
        assert(curr.header['AXS_RAT'] == 1.0)
    else:
        assert(curr.header['SHAPE'] == 'oblate')
        assert(curr.header['AXS_RAT'] == 1.4)
    assert(np.allclose(curr.data[0], vals['qext'], 10**(-2), 10**(-4)))
    assert(np.allclose(curr.data[1], vals['qsca'], 10**(-2), 10**(-4)))
    assert(np.allclose(curr.data[2], vals['qabs'], 10**(-2), 10**(-4)))

    #set check
    fits_index += 1
    ggadt_file = 'astrodust_test_files/hematite/hematite_' + str(i) + '_set.out'
    vals = np.loadtxt(ggadt_file, dtype=np.dtype([('E', np.float_), ('qsca', np.float_), ('qabs', np.float_), ('qext', np.float_)]))
    curr = file[fits_index]
    
    assert(curr.header['SET'])
    if (curr.header['RADIUS'] < 0.3):
        assert(curr.header['SHAPE'] == 'sphere')
        assert(curr.header['AXS_RAT'] == 1.0)
    else:
        assert(curr.header['SHAPE'] == 'oblate')
        assert(curr.header['AXS_RAT'] == 1.4)
    assert(np.allclose(curr.data[0], vals['qext'], 10**(-2), 10**(-4)))
    assert(np.allclose(curr.data[1], vals['qsca'], 10**(-2), 10**(-4)))
    assert(np.allclose(curr.data[2], vals['qabs'], 10**(-2), 10**(-4)))

    fits_index += 1
    i += 1

file.close()