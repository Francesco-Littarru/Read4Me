# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Read4Me'
copyright = '2023, Francesco Littarru'
author = 'Francesco Littarru'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

exclude_patterns = []
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.viewcode',
              'sphinx.ext.intersphinx',
              'sphinx.ext.todo',
              'sphinxcontrib.mermaid']
intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                       'gensim': ('https://radimrehurek.com/gensim/', None),
                       'telegram': ('https://docs.python-telegram-bot.org/en/stable/', None)}
templates_path = ['_templates']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

autoclass_content = 'both'          # join class and init docstrings
autodoc_member_order = 'bysource'   # keep the sorting of classes as in the source
html_static_path = ['_static']
html_theme = 'sphinx_rtd_theme'
todo_include_todos = True
pygments_style = 'sphinx'


