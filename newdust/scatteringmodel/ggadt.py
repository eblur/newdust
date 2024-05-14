'''
These functions will turn GGADT output into a fits file following the ScatteringModel convention

This is similar to the FITS file astrodust.py produces, except this time all of the data in the file has the same shape and orientation

FITS format:

    PrimaryHDU: contains information on the material, shape (including axis ratio), and orientation of the grain as well as the same comments as     ScatteringModel

    Params: A BinaryTableHDU containing np.ndarrays for eV range, radius, and theta (for differential scattering cross section)

    ImageHDU: Contains multidimensional np.ndarrays for qext, qsca, and qabs, which act as functions of radius and energy

    NOTE: Need to add functionality for differential scattering cross sections
'''

import astropy.units as u
from astropy.io import fits
import numpy as np
import scatteringmodel

#child of ScatteringModel class
class Ggadt(scatteringmodel.ScatteringModel):
    #__init__ REQUIRES a fits file to open
    def __init__(self, fits_file=None):
        assert(fits_file is not None)

        self.read_from_table(fits_file)



#Makes a FITS table for a set of GGADT output files for grains of the same shape and material
#NOTE: num_files is 0-indexed!
def make_fits(shape, material, num_files, outfile, overwrite=True):
    #data to store in FITS
    radii = []
    evs = []
    theta = [] #NEED TO IMPLEMENT
    qext = []
    qabs = []
    qsca = []
    diff = [] #NEED TO IMPLEMENT

    #The first file will be used to get the ev values so it will not be part of the loop to get data
    #CHANGE THE START OF THIS TO MAKE SURE YOU'RE IN THE RIGHT DIRECTORY TO ACCESS THE FILE
    filename = "test_model/" + material + "_0_" + shape + ".out"
    data = parse_file(filename, get_evs=True, get_ratio=True)

    #params data
    radii.append(data['radius'])
    evs = data['evs']
    oriented = data['oriented']

    #img data
    qext.append(data['qext'])
    qabs.append(data['qabs'])
    qsca.append(data['qsca'])

    #make header
    axis_ratio = data['axis ratio']
    header = make_header(material, shape, axis_ratio, oriented)

    #go through the rest of the output files and finish poplating qsca, qext, and qabs
    i = 1
    while i < num_files:
        filename = "test_model/" + material + "_" + str(i) + "_" + shape + ".out"
        data = parse_file(filename)
        qext.append(data['qext'])
        qabs.append(data['qabs'])
        qsca.append(data['qsca'])
        radii.append(data['radius'])

        i += 1
    
    #make parameters
    pars = make_pars(evs, radii, theta)
    
    img_list = []
    for (val, head) in zip([qext, qabs, qsca, diff],
                           ["Qext", "Qabs", "Qsca", "Diff-xsect (ster^-1)"]):
        htemp = fits.Header()
        htemp['TYPE'] = head
        img_list.append(fits.ImageHDU(val, header=htemp))

    #write table
    fnl_list = [header] + pars + img_list
    hdu_list = fits.HDUList(hdus=fnl_list)
    hdu_list.writeto(outfile, overwrite=overwrite)
    return outfile

def parse_file(filename, get_evs=False, get_ratio=False):
    #Read the file line by line (so we have data for each energy) then put each variable floato their own respective arrays
    file = open(filename, 'r')
    data = file.readlines()

    '''
    Other parameters to record:
    1. radius
    3. orientation
    '''

    radius = float((data[6].split())[2]) #in microns

    #Need to record the orienation
    #if angle mode = random then it's a random angle, otherwise it's set
    #FOR NOW AT LEAST set orienation means (0º,0º,0º)
    #oriented = True means not random orientation
    oriented = (data[12].split())[2]
    if (oriented == 'random'):
        oriented = False
    else:
        oriented = True
    
    axis_ratio = 1
    #Need to find the axis ratio (should be the same for all grains in the model so it only needs to be done once)
    if get_ratio:

        i = 15
        #axis data is one line lower if the grain is oriented by an angle file
        if oriented:
            i = 16
        
        #only need y and z axis to find axis ratio (these ones are always changed)
        #y-axis will also always be bigger based on how I've been running GGADT
        y_axis = float((data[i + 1].split())[2])
        z_axis = float((data[i + 2].split())[2])

        axis_ratio = round(y_axis / z_axis, 2)
        

    #split each line floato each variable
    qsca = []
    qabs = []
    qext = []
    evs = []

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
    
    data = {
        'radius': radius,
        'axis ratio': axis_ratio,
        'oriented': oriented,
        'qsca': qsca,
        'qabs': qabs,
        'qext': qext,
        'evs': evs
    }
    
    file.close()
    return data

def make_header(material, shape, axis_ratio, oriented):
    result = fits.Header()
    result['SHAPE'] = shape
    result['MATERIAL'] = material
    result['AX_RATIO'] = axis_ratio
    
    result['ORIENT'] = "Random"
    if oriented:
        result['ORIENT'] = 'Set'

    result['COMMENT']  = "Extinction efficiency and differential cross-sections"
    result['COMMENT']  = "HDUS 4-6 are Qext, Qsca, Qabs in wavelength (or energy) vs grain radius"
    result['COMMENT']  = "HDU 7 is the differential scattering cross-section (ster^-1)"
    return fits.PrimaryHDU(header=result)

def make_pars(evs, radii, theta):
    c1 = fits.BinTableHDU.from_columns(
        [fits.Column(name='lam', array=np.array(evs), format='E', unit='kiloelectronvolt')]
    )
    c2 = fits.BinTableHDU.from_columns(
        [fits.Column(name='a', array=np.array(radii), format='E', unit='um')]
    )
    c3 = fits.BinTableHDU.from_columns(
        [fits.Column(name='theta', array=np.array(theta), format='E', unit='radian')]
    )

    return [c1, c2, c3]