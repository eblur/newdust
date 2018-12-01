## Grain size distributions (and compositions) from
## Zubko, Dwek, & Arendt (2004) [ZDA hereafter]
## http://adsabs.harvard.edu/abs/2004ApJS..152..211Z

import numpy as np

## TO DO :
## [ ] Write a test for _zda_logg integrating to 1.0
## [ ] Catch Warnings on r/a = 1.0 special case in _zda_logg

# ZDA Table 2
ZDA_TYPES = ['PAHs', 'Graphite', 'Silicate', \
         'AmorphousCarbon', 'OrganicRefractory', 'WaterIce']
ZDA_RHO = dict(zip(ZDA_TYPES,
                   [2.24, 2.24, 3.5, 1.84, 1.6, 0.92]))
ZDA_UNIT_CELLS = dict(zip(ZDA_TYPES,
    ['C', 'C', 'MgFeSiO_{4}', 'C', 'C_{25}H_{25}O_{5}N', 'H_{2}O'])) # for reference

##--- ZDA Equation 20
# Analytical model for grain size distribution, f(a)
# f(a) = A g(a)  [um^{-1} H^{-1}]
# where g(a) integrates to 1 and A is the normalization in units of H^{-1}.
# a is in units of um

def _zda_logg(r, pars):
    # r = grain radius
    rmin, rmax, c, b, a, m = pars # coefficients in eq 20, and grain limits
    assert len(m) == 5 # value of m[0] doesn't matter
    assert len(a) == 5 # value of a[0] doesn't matter
    assert len(b) == 5

    terms = [c + b[0] * np.log10(r)] # length = len(a)
    terms.append(-b[1] * np.power(np.abs(np.log10(r/a[1])), m[1]))
    terms.append(-b[2] * np.power(np.abs(np.log10(r/a[2])), m[2]))
    terms.append(-b[3] * np.power(np.abs(r - a[3]), m[3])),
    terms.append(-b[4] * np.power(np.abs(r - a[4]), m[4]))
    terms = np.array(terms)
    # note special cases: if m < 0 and r/a = 1.0,
    # there will be a divide by zero error
    terms[np.isinf(terms)] = 0.0
    #print(terms)

    result = np.sum(terms, axis=0)
    result[r < rmin] = 0.0
    result[r > rmax] = 0.0
    return result # um^-1

def _zda_dnda(a, A, pars):
    logg = _zda_logg(a, pars)
    result = A * np.power(10.0, logg)
    return result

##---- ZDA model types
# First string:
# BARE = PAHs and "bare" grains
# COMP = PAHs + bare + composite grains
#
# Second string: type of bare carbon particles
# GR = graphite
# AC = amorphous carbon
# NC = no carbon
#
# Third string: abundance type
# S = solar composition
# B = B star composition
# FG = F & G star composition

##---- ZDA Table 6
ZDA_MODEL_TYPES = ['BARE-GR-S', 'BARE-GR-FG', 'BARE-GR-B', \
                   'BARE-AC-S', 'BARE-AC-FG', 'BARE-AC-B', \
                   'COMP-GR-S', 'COMP-GR-FG', 'COMP-GR-B', \
                   'COMP-AC-S', 'COMP-AC-FG', 'COMP-AC-B', \
                   'COMP-NC-S', 'COMP-NC-FG', 'COMP-NC-B']

# fraction of each grain type in the model
ZDA_F = dict(zip(ZDA_MODEL_TYPES,
                [dict(zip(ZDA_TYPES, [4.57, 29.47, 65.96, 0.0, 0.0, 0.0])),
                 dict(zip(ZDA_TYPES, [4.88, 29.44, 65.68, 0.0, 0.0, 0.0])),
                 dict(zip(ZDA_TYPES, [5.02, 33.38, 61.60, 0.0, 0.0, 0.0])),
                 dict(zip(ZDA_TYPES, [6.89, 0.0, 64.46, 28.65, 0.0, 0.0])),
                 dict(zip(ZDA_TYPES, [6.94, 0.0, 64.91, 28.15, 0.0, 0.0])),
                 dict(zip(ZDA_TYPES, [7.60, 0.0, 59.93, 32.47, 0.0, 0.0])),
                 dict(zip(ZDA_TYPES, [4.59, 14.96, 64.78, 0.0, 14.30, 1.37])),
                 dict(zip(ZDA_TYPES, [4.94, 18.43, 64.20, 0.0, 11.34, 1.09])),
                 dict(zip(ZDA_TYPES, [4.98, 19.64, 58.86, 0.0, 15.08, 1.44])),
                 dict(zip(ZDA_TYPES, [6.81, 0.0, 64.34, 10.13, 17.09, 1.63])),
                 dict(zip(ZDA_TYPES, [6.90, 0.0, 64.60, 10.85, 16.11, 1.54])),
                 dict(zip(ZDA_TYPES, [7.63, 0.0, 59.31, 4.16, 26.37, 2.53])),
                 dict(zip(ZDA_TYPES, [6.76, 0.0, 64.69, 0.0, 26.07, 2.49])),
                 dict(zip(ZDA_TYPES, [6.81, 0.0, 64.80, 0.0, 25.91, 2.48])),
                 dict(zip(ZDA_TYPES, [7.62, 0.0, 59.64, 0.0, 29.88, 2.86]))]))
# dust-to-gas mass ratio, for reference
ZDA_D2G = dict(zip(ZDA_MODEL_TYPES,
                  [0.00619, 0.00618, 0.00568, \
                   0.00639, 0.00648, 0.00589, \
                   0.00626, 0.00620, 0.00580, \
                   0.00637, 0.00642, 0.00578, \
                   0.00635, 0.00642, 0.00579]))

# ZDA Tables 7 - 21
# Model parameters follow precedent set in _zda_logg function
# pars = A, amin, amax, c, b (length 5), a (length 5), m (length 5)
# I always set a[0]=1 and m[0]=1 because it works out in the expansion.
# Any b[n]=0 terms will have a[n]=1 (to avoid divide by 0) and m[n]=0
ZDA_MODEL = dict()

# Table 7
ZDA_MODEL['BARE-GR-S'] = dict()
ZDA_MODEL['BARE-GR-S']['PAH'] = [2.227433e-7, 3.5e-4, 5.e-3, -8.02895, \
    [-3.45764, 1.18396e3, 0.0, 1.e24, 0.0], \
    [1.0, 1.0, 1.0, -5.29496e-3, 1.0], \
    [1.0, -8.20551, 0.0, 12.0146, 0.0]]
ZDA_MODEL['BARE-GR-S']['Graphite'] = [1.905816e-7, 3.5e-4, 0.33, -9.86, \
    [-5.02082, 5.81215e-3, 0.0, 1.12502e3, 1.12602e3], \
    [1.0, 0.415861, 1.0, 0.160344, 0.160501], \
    [1.0, 4.63229, 0.0, 3.69897, 3.69967]]
ZDA_MODEL['BARE-GR-S']['Silicate'] = [1.471288e-7, 3.5e-4, 0.37, -8.47091,\
    [-3.68708, 2.37316e-5, 0.0, 2.96128e3, 0.0], \
    [1.0, 7.64943e-3, 1.0, 0.480229, 1.0], \
    [1.0, 22.5489, 0.0, 12.1717, 0.0]]

def logg(avals, modelname, gtype):
    assert modelname in ZDA_MODEL_TYPES
    assert gtype in ZDA_MODEL[modelname].keys()
    return _zda_logg(avals, ZDA_MODEL[modelname][gtype][1:])
