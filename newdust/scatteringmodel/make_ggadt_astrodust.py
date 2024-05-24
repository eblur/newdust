"""
This file expands on make_ggadt.py, turning a GGADT output model with grains of differing shapes and orientations into multiple FITS files, each of which contains all the data for grains of the same shape and orientation
"""

from make_ggadt import make_fits

"""
Files follow the same naming structure as those for make_ggadt.py:

    [material]_[index]_[shape]_[orientation].out
    
    [index] is 0-indexed, where grain radius increasing with index

I structure my files by material so this program runs for one material at a time (The folder containing GGADT output data should be organized into subfolders by material)
"""

def make_astrodust_fits(shapes, material, orientations, folder, last_index):
    """
    Makes multiple FITS files, each containing the GGADT output data for grains of the same shape and orientation

    shapes: list[string]: a list of the shapes included in the astrodust model as they appear in the file names

    material: string: material of the grain as it appears in the file names -- this should be constant across all input files

    orientations: list[string]: A list of orientations, either 'set' or 'rand' -- 'set' denotes a set orientation, 'rand' denotes a random orientation -- but the code should work as long as 'rand' denotes the random orientation (i.e. There could be multiple set orientations). These should be inputted as they appear in the file names.

    folder: string: file path to the folder containing the GGADT output data. (ex: tables/astrodust_data/fayalite)

    last_index: int: the last (largest) index of a GGADT output file (ex: if there are 32 files, last_index = 31) (NOTE: There may be multiple files with the same index if there are multiple orientations)

    Returns a list of the names of FITS files containing the data described above
    """

    #This function largely relies on make_ggadt.make_fits() but we need to get the file names right before we can pass arguments
    files = []

    #can go through each shape and orientation and run make_ggadt.make_fits()
    for shape in shapes:
        for orientation in orientations:

            outfile = f'{material}_{shape}_{orientation}.fits'
            fits_file = make_fits(shape, material, orientation, folder, last_index, outfile)
            files.append(fits_file)
            
    return files

