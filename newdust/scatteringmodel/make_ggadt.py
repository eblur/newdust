"""
This file turns multiple GGADT output files containing extinction data for grains of the same material, shape, and orientation into a single FITS file

FITS files produced by make_ggadt follow the same structure as those used by the ScatteringModel class
"""

from astropy.io import fits
import numpy as np

"""
GGADT output files are named as such: [material]_[index].out
    
    [index] is 0-indexed -- grain radius increases with the index
"""
def make_fits(material, folder, indicies, outfile, overwrite=True):
    """
    Creates a FITS file containing qsca, qext, and qabs data from multiple GGADT grains of the shape and material

    material: string: material of the grain as it appears in the file names -- this should be constant across all input files

    folder: string: file path to the folder containing the GGADT output data

    indicies: list[int]: a list of the indicies used in file naming -- should be consecutive integers range from 0 to the last index used in file naming (ex: if there are 32 files, indicies = range(32))

    outfile: string: name of the ouputted FITS file

    overwrite: bool: whether or not to overrite a preexisting FITS file with the same name

    Returns FITS file with the data described above
    """

    #data to store in FITS
    radii = []
    qext = []
    qabs = []
    qsca = []
    diff = [] #NEED TO IMPLEMENT

    #constant parameters
    evs = []
    theta = [0.0] #NEED TO IMPLEMENT
    axis_ratio = 0.0
    shape = ''
    orientation = ''

    #go through the output files and popuate above variables
    for i in indicies:
        filename= f'{folder}/{material}_{i}.out'
        data = _parse_file(filename)

        qext.append(data['qext'])
        qabs.append(data['qabs'])
        qsca.append(data['qsca'])
        diff.append(data['diff'])
        radii.append(data['radius'])

        #need to either assign or check consistency of constant parameters
        if not len(evs) == 0: 
            evs = data['evs']
        elif evs != data['evs']:  
            raise Exception('Error: Energy grid must be the same across all files')

        if axis_ratio == 0.0:
            axis_ratio = data['axis ratio']
        elif axis_ratio != data['axis ratio']:
            raise Exception('Error: Axis ratio must be constant across all files')
        
        if shape == '':
            shape = data['shape']
        elif shape != data['shape']:
            raise Exception('Error: Shape must be constant across all files')
        
        if orientation == '':
            orientation = data['orientation']
        elif orientation != data['orientation']:
            raise Exception('Error: Orientation must be constant across all files')
    
    #Radii and indicies should be the same length -- if they're not something is wrong
    if len(indicies) != len(radii):
        raise Exception('Error: Too many radii') if len(indicies) < len(radii) else Exception('Error: Not enough radii')

    #make parameters and header
    header = _make_header(material, shape, axis_ratio, orientation)
    pars = _make_pars(evs, radii, theta)
    
    #need to transpose the shape of qext, qabs, qsca, and diff to follow scatteringmodel
    #diff also needs the extra dimension for theta
    qext = np.array(qext).transpose()
    qabs = np.array(qabs).transpose()
    qsca = np.array(qsca).transpose()
    diff = np.expand_dims(np.array(diff).transpose(), 2)
    
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

def _parse_file(filename):
    """
    Parses through a GGADT output file and returns header and extinction data

    filename: string: full path, including name, to a GGADT output file

    Returns a dict containing all of the relevant data for the GGADT output file given by filename
    """
    #Read the file line by line (so we have data for each energy) then put each variable to their own respective arrays
    file = open(filename, 'r')
    data = file.readlines()

    radius = float((data[6].split())[2]) #in microns
    orientation = (data[12].split())[2]
    if orientation == 'file':
        orientation = 'set'

    #Need to find the axis ratio 
    axis_ratio = 1.0

    i = 15
    #axis data is one line lower if the grain is set by an angle file
    if orientation != 'random':
        i = 16
        
    x_axis = float((data[i].split())[2])
    y_axis = float((data[i + 1].split())[2])
    z_axis = float((data[i + 2].split())[2])

    shape = ''
    #can determine the shape based on axis equalities
    axes = [x_axis, y_axis, z_axis]
    axes.sort(key = float)

    if (axes[0] == axes[1] and axes[1] == axes[2]):
        shape = 'sphere'
    elif (axes[0] == axes[1] and axes[1] < axes[2]):
        shape = 'prolate'
    elif (axes[0] < axes[1] and axes[1] == axes[2]):
        shape = 'oblate'
    else:
        raise Exception('Error: Unknown Shape')

    #Axis ratio should be > 1 so we don't need to do every permutation
    y_x = round(y_axis / x_axis, 2) if (y_axis / x_axis) >= 1 else round(x_axis / y_axis, 2)
    y_z = round(y_axis / z_axis, 2) if (y_axis / z_axis) >= 1 else round(z_axis / y_axis, 2)
    x_z = round(x_axis / z_axis, 2) if (x_axis / z_axis) >= 1 else round(z_axis / x_axis, 2)

    axis_ratio = max(y_x, y_z, x_z)

    qsca = []
    qabs = []
    qext = []
    diff = []
    evs = []

    #split each line to each variable
    #If the grain is oriented there's an extra line in the header so i needs to be one greater
    i = 23
    if orientation != 'random':
        i += 1
    
    while i < len(data):
        #There should be 4 different vals, which can then be added to the lists above
        vals = data[i].split()
        evs.append(float(vals[0]))
        qsca.append(float(vals[1]))
        qabs.append(float(vals[2]))
        qext.append(float(vals[3]))
        diff.append(0.0)


        i += 1
    
    data = {
        'radius': radius,
        'orientation': orientation,
        'shape': shape,
        'axis ratio': axis_ratio,
        'qsca': qsca,
        'qabs': qabs,
        'qext': qext,
        'evs': evs,
        'diff': diff
    }
    
    file.close()
    return data

def _make_header(material, shape, axis_ratio, orientation):
    """
    Makes the PimaryHDU for the FITS file returned by make_fits

    material: string: the material of the grains (this is the same for every GGADT output file)

    shape: string: the shape of the grains (this is the same for every GGADT output file)

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

def _make_pars(evs, radii, theta):
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