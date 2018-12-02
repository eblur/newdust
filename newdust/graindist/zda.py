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

# Table 8
ZDA_MODEL['BARE-GR-FG'] = dict()
ZDA_MODEL['BARE-GR-FG']['PAH'] = [2.484404e-7, 3.5e-4, 5.e-3, -8.54571, \
    [-3.60112, 1.86525e5, 0.0, 1.e24, 0.0], \
    [1.0, 1.0, 1.0, 1.98119e-3, 1.0], \
    [1.0, -13.5755, 0.0, 9.25894, 0.0]]
ZDA_MODEL['BARE-GR-FG']['Graphite'] = [1.901190e-7, 3.e5-4, 0.3, -10.1149, \
    [-5.3308, 7.54276e-2, 0.0, 1.12502e3, 1.12602e3], \
    [1.0, 8.08703e-2, 1.0, 0.145378, 0.169079], \
    [1.0, 3.37644, 0.0, 3.49042, 3.63654]]
ZDA_MODEL['BARE-GR-FG']['Silicate'] = [1.541199e-7, 3.5e-4, 0.34, -8.53081, \
    [-3.70009, 3.96003e-9, 0.0, 1.48e3, 1.481e3], \
    [1.0, 9.11246e-3, 1.0, 0.484381, 0.474035], \
    [1.0, 47.0606, 0.0, 12.3253, 12.0995]]

# Table 9
ZDA_MODEL['BARE-GR-B'] = dict()
ZDA_MODEL['BARE-GR-B']['PAH'] = [2.187355e-7, 3.5e-4, 5.5e-3, -8.84618, \
    [-3.69582, 1.23836e5, 0.0, 1.e24, 0.0], \
    [1.0, 1.0, 1.0, 2.328e-3, 1.0], \
    [1.0, -13.5577, 0.0, 9.36086, 0.0]]
ZDA_MODEL['BARE-GR-B']['Graphite'] = [1.879863e-7, 3.5e-4, 0.32, -9.92887, \
    [-5.14159, 4.68489e-3, 0.0, 1.12505e3, 1.12605e3], \
    [1.0, 0.450668, 1.0, 0.154046, 0.153688], \
    [1.0, 4.85266, 0.0, 3.56481, 3.56482]]
ZDA_MODEL['BARE-GR-B']['Silicate'] = [1.238052e-7, 3.5e-4, 0.32, -8.53419, \
    [-3.7579, 3.89361e-13, 0.0, 1.481e3, 1.48003e3], \
    [1.0, 1.27635e-3, 1.0, 0.268976, 0.836879], \
    [1.0, 34.0815, 0.0, 13.3815, 44.1634]]

# Table 10
ZDA_MODEL['COMP-GR-S'] = dict()
ZDA_MODEL['COMP-GR-S']['PAH'] = [2.243245e-7, 3.5e-4, 5.5e-3, -8.97672, \
    [-3.73654, 9.86507e10, 0.0, 1.e24, 0.0], \
    [1.0, 1.0, 1.0, 2.34542e-3, 1.0], \
    [1.0, -14.8506, 0.0, 9.33589, 0.0]]
ZDA_MODEL['COMP-GR-S']['Graphite'] = [1.965e-7, 3.5e-4, 0.5, -10.4717, \
    [-5.32268, 5.63787e-3, 0.0, 1.12504e3, 1.12597e3], \
    [1.0, 7.75892e-2, 1.0, 0.125304, 0.271622], \
    [1.0, 3.33491, 0.0, 6.04033, 4.67116]]
ZDA_MODEL['COMP-GR-S']['Silicate'] = [1.160677e-7, 3.5e-4, 0.44, -5.77068, \
    [-3.82724, 1.4815e-7, 0.0, 5.843, 0.0], \
    [1.0, 7.44945e-3, 1.0, 0.398924, 1.0], \
    [1.0, 12.3238, 0.0, 0.561698, 0.0]]
ZDA_MODEL['COMP-GR-S']['Composite'] = [6.97552e-12, 0.02, 0.9, -3.90395, \
    [-3.5354, 9.85176e-31, 0.0, 0.0, 0.0], \
    [1.0, 2.30147e-4, 1.0, 1.0, 1.0], \
    [1.0, 33.3071, 0.0, 0.0, 0.0]]

# Table 11
ZDA_MODEL['COMP-GR-FG'] = dict()
ZDA_MODEL['COMP-GR-FG']['PAH'] = [2.520814e-7, 3.5e-4, 5.e-3, -8.72489, \
    [-3.65649, 9.86507e10, 0.0, 1.e24, 0.0], \
    [1.0, 1.0, 1.0, 2.05181e-3, 1.0], \
    [1.0, -14.6651, 0.0, 9.20391, 0.0]]
ZDA_MODEL['COMP-GR-FG']['Graphite'] = [1.936847e-7, 3.5e-4, 0.39, -11.1324, \
    [-6.6148, 3.66626e-2, 0.0, 1.12501e-3, 1.126e3], \
    [1.0, 0.144398, 1.0, 0.166373, 0.400672], \
    [1.0, 2.54938, 0.0, 4.58796, 6.14619]]
ZDA_MODEL['COMP-GR-FG']['Silicate'] = [1.309292e-7, 3.5e-4, 0.39, -3.81346, \
    [-3.76412, 2.62792e-9, 0.0, 6.64727, 0.0], \
    [1.0, 7.26393e-3, 1.0, 0.344185, 1.0], \
    [1.0, 15.5036, 0.0, 0.21785, 0.0]]
ZDA_MODEL['COMP-GR-FG']['Composite'] = [5.393662e-12, 0.02, 0.75, -3.82614, \
    [-3.48373, 9.86756e-31, 0.0, 0.0, 0.0], \
    [1.0, 4.13811e-4, 1.0, 1.0, 1.0], \
    [1.0, 34.9122, 0.0, 0.0, 0.0]]

# Table 12
ZDA_MODEL['COMP-GR-B'] = dict()
ZDA_MODEL['COMP-GR-B']['PAH'] = [2.216925e-7, 3.5e-4, 5.5e-3, -9.04531, \
    [-3.75834, 9.86507e10, 0.0, 1.e24, 0.0], \
    [1.0, 1.0, 1.0, 2.38145e-3, 1.0], \
    [1.0, -14.9148, 0.0, 9.34323, 0.0]]
ZDA_MODEL['COMP-GR-B']['Graphite'] = [1.918716e-7, 3.5e-4, 0.52, -10.1159, \
    [-5.45055, 2.58749e-3, 0.0, 1.0023e2, 1.00027e2], \
    [1.0, 9.91702e-2, 1.0, 0.200689, 0.699922], \
    [1.0, 3.71707, 0.0, 3.52158, 9.86403]]
ZDA_MODEL['COMP-GR-B']['Silicate'] = [1.082933e-7, 3.5e-4, 0.33, 1.39336e2, \
    [-3.66338, 2.85829e-10, 0.0, 1.48931e2, 0.0], \
    [1.0, 5.26352e-3, 1.0, 0.341914, 1.0], \
    [1.0, 16.487, 0.0, 5.05577e-3, 0.0]]
ZDA_MODEL['COMP-GR-B']['Composite'] = [4.780856e-12, 0.02, 0.45, -3.72463, \
    [-3.4173, 2.56334e-26, 0.0, 0.0, 0.0], \
    [1.0, 2.05195e-4, 1.0, 1.0, 1.0], \
    [1.0, 29.4592, 0.0, 0.0, 0.0]]

# Table 13
ZDA_MODEL['BARE-AC-S'] = dict()
ZDA_MODEL['BARE-AC-S']['PAH'] = [4.492237e-7, 3.5e-4, 3.7e-3, -9.05931, \
    [-3.76458, 6.28593e5, 0.0, 1.e24, 0.0], \
    [1.0, 1.0, 1.0, 1.69966e-3, 1.0], \
    [1.0, -14.3443, 0.0, 8.8067, 0.0]]
ZDA_MODEL['BARE-AC-S']['ACH2'] = [8.185937e-12, 0.02, 0.26, -3.96337, \
    [-3.57444, 1.93427e-18, 0.0, 0.0, 0.0], \
    [1.0, 1.0046e-4, 1.0, 1.0, 1.0], \
    [1.0, 33.923, 0.0, 0.0, 0.0]]
ZDA_MODEL['BARE-AC-S']['Silicate1'] = [3.527574e-7, 3.5e-4, 0.025, -8.88283, \
    [-3.69508, 3.03135e-20, 0.0, 0.0, 0.0], \
    [1.0, 3.00297e-7, 1.0, 1.0, 1.0], \
    [1.0, 28.9189, 0.0, 0.0, 0.0]]
ZDA_MODEL['BARE-AC-S']['Silicate2'] = [6.134893e-13, 0.0272, 0.37, 8.93254e3, \
    [5.76792e3, 5.77029e3, 3.78160e2, 0.0, 0.0], \
    [1.0, 2.82861e-2, 9.39447e-2, 1.0, 1.0], \
    [1.0, 1.00027, 9.04197, 0.0, 0.0]]

# Table 14
ZDA_MODEL['BARE-AC-FG'] = dict()
ZDA_MODEL['BARE-AC-FG']['PAH'] = [4.727727e-7, 3.5e-4, 3.6e-3, -8.91244, \
    [-3.72015, 6.78215e5, 0.0, 1.e24, 0.0], \
    [1.0, 1.0, 1.0, 1.58225e-3, 1.0], \
    [1.0, -14.2532, 0.0, 8.71891, 0.0]]
ZDA_MODEL['BARE-AC-FG']['ACH2'] = [7.862901e-12, 0.02, 0.28, -3.92513, \
    [-3.54913, 2.13708e-17, 0.0, 0.0, 0.0], \
    [1.0, 2.03908e-4, 1.0, 1.0, 1.0], \
    [1.0, 34.7835, 0.0, 0.0, 0.0]]
ZDA_MODEL['BARE-AC-FG']['Silicate1'] = [3.680573e-7, 3.5e-4, 0.024, -8.88283, \
    [-3.69508, 2.17105e-20, 0.0, 0.0, 0.0], \
    [1.0, 3.e-7, 1.0, 1.0, 1.0], \
    [1.0, 29.2, 0.0, 0.0, 0.0]]
ZDA_MODEL['BARE-AC-FG']['Silicate2'] = [6.218762e-13, 0.026, 0.37, 9.04443e3, \
    [5.7679e3, 5.77024e3, 3.82848e2, 0.0, 0.0], \
    [1.0, 2.7051e-2, 9.39615e-2, 1.0, 1.0], \
    [1.0, 1.00024, 8.94494, 0.0, 0.0]]

## ------ Convenience function for now

def logg(avals, modelname, gtype):
    assert modelname in ZDA_MODEL_TYPES
    assert gtype in ZDA_MODEL[modelname].keys()
    return _zda_logg(avals, ZDA_MODEL[modelname][gtype][1:])
