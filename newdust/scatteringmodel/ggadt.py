'''
What this file does:

1. Reads GGADT output files and puts all of it floato a FITS file of the same format as ScatteringModel
2. Creates a ggadt version of the ScatteringModel class that can be initialized with a FITS file

'''

import astropy.units as u
from astropy.io import fits
import argparse
import numpy as np

'''
    Requires: filename must be a valid filename from the ggadt astrodust model
    Purpose: parse_file takes in a ggadt total cross section output file and returns a dict
    NOTE: The header of the ggadt output files are lines 1 - 23 (0 - 22 for 0-indexing), lines 24 (23) - inf are data
'''

def parse_file(filename):
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

    radius = float((data[6].split())[2]) #in microns
    material = (data[10].split())[3]

    #need to compare each axis to figure out shape
    #Right now if a grain's radius is > 0.3 microns then it's oblate
    shape = 'spherical'
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
    
    #split each line floato each variable
    nrg = []
    qsca = []
    qabs = []
    qext = []

    #If the grain is oriented there's an extra line in the header so i needs to be one greater
    i = 23
    if (oriented):
        i += 1
    
    while i < len(data):
        #There should be 4 different vals, which can then be added to the lists above
        vals = data[i].split()
        nrg.append(float(vals[0]))
        qsca.append(float(vals[1]))
        qabs.append(float(vals[2]))
        qext.append(float(vals[3]))


        i += 1
    
    #lastly return a dict with all of the info
    info = {
        "radius": radius,
        "material": material,
        "shape": shape,
        "oriented": oriented,
        "energy": nrg,
        "qsca": qsca,
        "qabs": qabs,
        "qext": qext
    }

    return info

parse_file('fayalite_2_rand.out')