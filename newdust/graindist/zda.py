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
