'''
What this file does:

1. Reads GGADT output files and puts all of it floato a FITS file
2. Creates a ggadt version of the ScatteringModel class that can be initialized with a FITS file

'''

import astropy.units as u
from astropy.io import fits
import numpy as np

'''
    Purpose: make_fits will make a fits file for a given material given its ggadt output files

    Parameters should be a binary table HDU (energy is a parameter! -- it's the same for every grain)
    qsca, qext, and qabs should be image tables

    Uses the parse_file, make_header, and make_pars helpers

    naming convention for files:

    [material name]_[radius index (starting at 1)]_[rand or set (set is only for i > 10)].out

    there are 54 files per material i goes from 1-32 inclusive (from 11 - 32 there are two files per index)

    ex: hematite_8_rand.out or metallic_iron_32_set.out
'''
#data_folder is the file path to the folder than contains the ggadt data
#num_radii is the number of different radii ggadt is run for
#min_set index is the lowest radius index that has a file for both set and random orientations (1-index based on file naming!)
#If no files are set, then min_set_index just needs to be greater than the total number of files
def make_fits(material, data_folder, min_set_index, num_radii, outfile, overwrite=True):
    '''
    Need to read in the first file for a material to get the eV range and make the params HDU
    '''
    
    '''
    The last step is to go through every file for the given material and add their specific parameters, and extinction values to an imageHDU

    The header of this will contain the radius, orientation (true = oriented), shape, and the order of q_ext, q_abs, and q_sca
    '''
    img_list = []
    num_files = 0
    radii = []
    evs = []

    i = 1 #1-indexed because of naming convention
    while i <= num_radii:
        filename = data_folder + "/" + material + "/" + material + "_" + str(i) + "_rand.out"
        if (i != 1):
            img_list.append(parse_file(filename, radii))
        else:
            img_list.append(parse_file(filename, radii, evs, get_evs=True))
        num_files += 1

        #If radius is larger than 0.3, need to parse for set orientation as well
        if i >= min_set_index:
            filename = data_folder + "/" + material + "/" + material + "_" + str(i) + "_set.out"
            img_list.append(parse_file(filename, radii))
            num_files += 1

        i += 1
    
    #The header has to be made now so we know how many files there are
    header = make_header(material, num_files)

    params = make_params(evs, radii)

    #put together the header, params, and img_list into the final fits file
    final_list = [header] + params + img_list
    hdu_list = fits.HDUList(hdus=final_list)
    hdu_list.writeto(outfile, overwrite=overwrite)
    return
'''
    Requires: filename must be a valid filename from the ggadt astrodust model
    Purpose: parse_file takes in a ggadt total cross section output file and returns a fits ImageHDU of the data
    NOTE: The header of the ggadt output files are lines 1 - 23 (0 - 22 for 0-indexing), lines 24 (23) - inf are data
'''
def parse_file(filename, radii, evs=[], get_evs=False):
    #Read the file line by line (so we have data for each energy) then put each variable floato their own respective arrays
    file = open(filename, 'r')
    data = file.readlines()

    '''
    Other parameters to record:
    1. radius
    2. shape (include 1.4:1 axis ratio)
    3. orientation
    4. Material of grain 
    '''

    radius = round(float((data[6].split())[2]), 5) #in microns
    radii.append(radius)

    #need to compare each axis to figure out shape
    #Right now if a grain's radius is > 0.3 microns then it's oblate
    shape = 'sphere'
    if (radius > 0.3):
        shape = 'oblate'
    
    
    #finally need to record the orienation
    #if angle mode = random then it's a random angle, otherwise it's set
    #FOR NOW AT LEAST set orienation means (0º,0º,0º)
    #oriented = True means not random orientation
    oriented = (data[12].split())[2]
    if (oriented == 'random'):
        oriented = False
    else:
        oriented = True
    
    #only need y and z axis to find axis ratio (these ones are always changed)
    #y-axis will also always be bigger based on how I've been running GGADT
    i = 16
    if oriented:
        i += 1

    y_axis = float((data[i].split())[2])
    z_axis = float((data[i + 1].split())[2])

    axis_ratio = round(y_axis / z_axis, 2)
    
    #split each line floato each variable
    qsca = []
    qabs = []
    qext = []

    #If the grain is oriented there's an extra line in the header so i needs to be one greater
    i = 23
    if (oriented):
        i += 1
    
    while i < len(data):
        #There should be 3 different vals, which can then be added to the lists above
        #Energy isn't taken because it's the same for every file, it'll be taken later
        vals = data[i].split()
        qsca.append(float(vals[1]))
        qabs.append(float(vals[2]))
        qext.append(float(vals[3]))
        
        if get_evs:
            evs.append(float(vals[0]))


        i += 1
    
    
    htemp = fits.Header()
    htemp['RADIUS'] = radius
    htemp['SET'] = oriented
    htemp['SHAPE'] = shape
    htemp['AXS_RAT'] = axis_ratio
    htemp['ORDER'] = 'The order of data for each table is qext, qsca, qabs'

    #need to organize the data as well into a nested np.array
    vals = np.array([np.array(qext), np.array(qsca), np.array(qabs)])
    img = fits.ImageHDU(vals, header=htemp)
    
    file.close()
    return img

'''
Makes a header for the final fits file
Credit to https://github.com/eblur/newdust/blob/master/newdust/scatteringmodel/scatteringmodel.py for this code and make_params
'''

def make_header(material, num_files):
    result = fits.Header()
    result['MATERIAL'] = material
    result['MAX_I'] = (num_files - 1 + 3, 'The last index of data in the FITS file')
    result['COMMENT'] = "Scattering, absorption, and extinction efficiencies for " + material
    result['COMMENT'] = "Rand means the grain is oriented randomly, set means the grain is oriented at 0,0,0 degrees"
    result['COMMENT'] = "HDUS 1 and 2 (0-indexed) are radii and energies, respectively"
    result['COMMENT'] = "HDUs 3 - " + str(num_files - 1 + 3) + " contain qext, qsca, and qabs data for a given grain radius"

    return fits.PrimaryHDU(header=result)

def make_params(evs, radii):
    #make 2 columns, one with the ev range for the materials, and the other with the list of radii
    #radii start at 1 for file indicies, so this will do the same

    c1 = fits.BinTableHDU.from_columns([fits.Column(name='Radii', array=np.array(radii), unit='micron', format='E')])
    c2 = fits.BinTableHDU.from_columns([fits.Column(name='Energies', array=np.array(evs), unit='keV', format='E')])

    return [c1,c2]
