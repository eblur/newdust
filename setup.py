
#! /usr/bin/env python

from distutils.core import setup

import glob

##-------------------------------------------------
## Overly basic way of automating package list

def make_pkg_string(dir_path):
    dir_list = dir_path.split('/')
    result   = dir_list[0]
    for i in dir_path.split('/')[1:-1]:
        result = result + '.' + i
    return result

PACKAGES = ['newdust']
n, keep_looking = 1, True
while keep_looking:
    pkg_paths = glob.glob('newdust/' + '*/' * n)
    if len(pkg_paths) == 0:
        keep_looking = False
    else:
        pkg_names = [make_pkg_string(p) for p in pkg_paths]
        for pn in pkg_names: PACKAGES.append(pn)
        n += 1

PACKAGES.remove('newdust.graindist.tables')
        
##-------------------------------------------------
## Package setup

setup(name='newdust',
      version='0.1',
      description='Library of dust scattering codes',
      author='Lia Corrales',
      author_email='lia@astro.wisc.edu',
      url='https://github.com/eblur/newdust',
      packages=PACKAGES,
      package_data={'newdust': ['graindist/tables/*']}
)
