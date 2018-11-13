## Grain size distributions (and compositions) from
## Zubko, Dwek, & Arendt (2004) [ZDA hereafter]
## http://adsabs.harvard.edu/abs/2004ApJS..152..211Z

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

def zda_logg(a, pars):
    m, a, b, c = pars # coefficients in eq 20
    assert len(m) == 5 # value of m[0] doesn't matter
    assert len(a) == 5 # value of a[0] doesn't matter
    assert len(b) == 5
    assert len(c) == 1
    terms = [c0, b[0] * np.log10(a)]
    terms.append(-b[1] * np.power(np.abs(np.log10(a/a[1])), m[1]))
    terms.append(-b[2] * np.power(np.abs(np.log10(a/a[2])), m[2]))
    terms.append(-b[3] * np.power(np.abs(a - a[3]), m[3])),
    terms.append(-b[4] * np.power(np.abs(a - a[4]), m[4]))
    result = np.array(terms)
    return np.sum(result)

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

# ZDA Table 3
ZDA_MODEL_TYPES = ['COMP-GR-S', 'COMP-AC-S', 'COMP-NC-S', \
                   'COMP-GR-FG', 'COMP-AC-FG', 'COMP-NC-FG', \
                   'COMP-GR-B', 'COMP-AC-B', 'COMP-NC-B']
