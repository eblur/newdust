"""
This file turns multiple GGADT output files containing extinction data for grains of the same material, shape, and orientation into a single FITS file

"""
import os
from astropy.io import fits
import numpy as np

"""
GGADT output files are named as such: [material]_[index]_[shape]_[orientation].out
    
    [index] is 0-indexed -- grain radius increases with the index
"""
def make_fits(shape, material, orientation, folder, last_index, outfile, overwrite=True):
    """
    Creates a FITS file containing qsca, qext, and qabs data from multiple GGADT grains of the shape and material

    shape: string: shape of grain for output files as it appears in the file names -- this should be constant across all input files

    material: string: material of the grain as it appears in the file names -- this should be constant across all input files

    orientation: string: Either 'set' or 'rand' -- 'set' denotes a set orientation (0º, 0º, 0º), 'rand' denotes a random orientation -- but the code should work as long as 'rand' denotes the random orientation (i.e. There could be multiple set orientations). This should be inputted as it appears in the file names.

    folder: string: file path to the folder containing the GGADT output data

    last_index: int: the last (largest) index of a GGADT output file (ex: if there are 32 files, last_index = 31)

    outfile: string: name of the ouputted FITS file

    overwrite: bool: whether or not to overrite a preexisting FITS file with the same name

    Returns FITS file with the data described above
    """

    #data to store in FITS
    radii = []
    evs = []
    theta = [] #NEED TO IMPLEMENT
    qext = []
    qabs = []
    qsca = []
    diff = [] #NEED TO IMPLEMENT

    axis_ratio = 1.0
    have_evs_and_ratio = False

    #go through the output files and popuate above variables
    i = 0
    while i <= last_index:
        filename= f'{folder}/{material}_{i}_{shape}_{orientation}.out'

        #check file existence -- for larger astrodust models there many not be GGADT output files with certain names (ex: no random oblates from indicies 0 - 10)
        if not os.path.isfile(filename):
            i += 1
            continue
        
        #evs and axis ratio are constant through a FITS file so they only need to be collected once
        if have_evs_and_ratio:

            data = parse_file(filename, orientation)
            qext.append(data['qext'])
            qabs.append(data['qabs'])
            qsca.append(data['qsca'])
            radii.append(data['radius'])

        else:

            data = parse_file(filename, orientation, have_evs_and_ratio=False)
            qext.append(data['qext'])
            qabs.append(data['qabs'])
            qsca.append(data['qsca'])
            radii.append(data['radius'])
            evs = data['evs']
            axis_ratio = data['axis ratio']

            have_evs_and_ratio = True

        i += 1
    
    #make parameters and header
    header = make_header(material, shape, axis_ratio, orientation)
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

#Helper functions:

def parse_file(filename, orientation, have_evs_and_ratio=True):
    """
    Parses through a GGADT output file and returns header and extinction data

    filename: string: full path, including name, to a GGADT output file

    orientation: string: Either 'set' or 'rand' -- 'set' denotes a set orientation (0º, 0º, 0º), 'rand' denotes a random orientation -- but the code should work as long as 'rand' denotes the random orientation (i.e. There could be multiple set orientations). This should be inputted as it appears in the file names.

    have_evs_and_ratio: bool: if true, parse_file() not will record the axis ratio nor incident energy values of the grain in the given output file -- these should be the same for every output file so they only need to be recorded once

    Returns a dict containing all of the relevant data for the GGADT output file given by filename
    """
    #Read the file line by line (so we have data for each energy) then put each variable to their own respective arrays
    file = open(filename, 'r')
    data = file.readlines()

    radius = float((data[6].split())[2]) #in microns

    #Need to find the axis ratio (should be the same for all grains in the model so it only needs to be done once)
    axis_ratio = 1.0
    if not have_evs_and_ratio:

        i = 15
        #axis data is one line lower if the grain is set by an angle file
        if orientation != 'rand':
            i = 16
        
        x_axis = float((data[i].split())[2])
        y_axis = float((data[i + 1].split())[2])
        z_axis = float((data[i + 2].split())[2])

        #Axis ratio should be > 1 so we don't need to do every permutation
        y_x = round(y_axis / x_axis, 2) if (y_axis / x_axis) >= 1 else round(x_axis / y_axis, 2)
        y_z = round(y_axis / z_axis, 2) if (y_axis / z_axis) >= 1 else round(z_axis / y_axis, 2)
        x_z = round(x_axis / z_axis, 2) if (x_axis / z_axis) >= 1 else round(z_axis / x_axis, 2)

        #Will use largest ratio for axis ratio
        axis_ratio = max(y_x, y_z, x_z)
        

    qsca = []
    qabs = []
    qext = []
    evs = []

    #split each line to each variable
    #If the grain is oriented there's an extra line in the header so i needs to be one greater
    i = 23
    if orientation != 'rand':
        i += 1
    
    while i < len(data):
        #There should be 4 different vals, which can then be added to the lists above
        vals = data[i].split()
        qsca.append(float(vals[1]))
        qabs.append(float(vals[2]))
        qext.append(float(vals[3]))

        if not have_evs_and_ratio:
            val = (float(vals[0]))
            evs.append(val)


        i += 1
    
    data = {
        'radius': radius,
        'axis ratio': axis_ratio,
        'qsca': qsca,
        'qabs': qabs,
        'qext': qext,
        'evs': evs
    }
    
    file.close()
    return data

def make_header(material, shape, axis_ratio, orientation):
    """
    Makes the PimaryHDU for the FITS file returned by make_fits

    material: string: the material of the grains (this is the same for every file)

    shape: string: the shape of the grains (this is the same for every file)

    axis_ratio: float: the axis ratio of the grains

    orientation: string: Either 'set' or 'rand' -- 'set' denotes a set orientation, 'rand' denotes a random orientation -- but the code should work as long as 'rand' denotes the random orientation (i.e. There could be multiple set orientations). This should be inputted as it appears in the file names.

    Returns a PrimaryHDU for make_fits
    """
    result = fits.Header()
    result['SHAPE'] = shape
    result['MATERIAL'] = material
    result['AX_RATIO'] = axis_ratio
    result['ORIENT'] = orientation
    result['COMMENT']  = "Extinction efficiency and differential cross-sections"
    result['COMMENT']  = "HDUS 4-6 are Qext, Qsca, Qabs in wavelength (or energy) vs grain radius"
    result['COMMENT']  = "HDU 7 is the differential scattering cross-section (ster^-1)"
    return fits.PrimaryHDU(header=result)

def make_pars(evs, radii, theta):
    """
    Makes three BinaryTableHDUs after the PrimaryHDU for make_fits

    evs: list[float]: list of the incident energies -- this list is the same for every GGADT output file

    radii: list[float]: list of the radii for each grain

    theta: list[float]: list of angles for the differential scattering cross sections

    Returns a list of the three BinaryTableHDUs listed above
    """
    c1 = fits.BinTableHDU.from_columns(
        [fits.Column(name='lam', array=np.array(evs), format='D', unit='keV')]
    )
    c2 = fits.BinTableHDU.from_columns(
        [fits.Column(name='a', array=np.array(radii), format='D', unit='micron')]
    )
    c3 = fits.BinTableHDU.from_columns(
        [fits.Column(name='theta', array=np.array(theta), format='D', unit='rad')]
    )

    return [c1, c2, c3]