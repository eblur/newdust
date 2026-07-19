# Configuration file for the Sphinx documentation builder.
import os
import sys

# Make the xdust package importable from the src layout
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# -- Project information -----------------------------------------------------
project = 'Xdust'
copyright = '2026, Lia Corrales'
author = 'Lia Corrales'
version = '1.0'
release = '1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'myst_nb',
    # 'sphinx_autodoc_typehints', # requires a different format: https://pypi.org/project/sphinx-autodoc-typehints/
]

autosummary_generate = True

autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}

templates_path = ['_templates']
exclude_patterns = ['_build']

master_doc = 'index'
source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
}
language = 'en'

# -- Options for myst-nb -------------------------------------------------
nb_execution_mode = 'auto'
nb_execution_timeout = 120
myst_enable_extensions = ['dollarmath', 'amsmath']

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'

html_theme_options = {}

html_static_path = []
