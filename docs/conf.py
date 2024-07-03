# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
import STOUT
from datetime import datetime

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "STOUT - Smiles-TO-iUpac-Translator"
version = STOUT.__version__
current_year = datetime.today().year
copyright = "2021-{}, Kohulan Rajan at the Friedrich Schiller University Jena".format(
    current_year
)
author = "Kohulan Rajan"
rst_prolog = """
.. |current_year| replace:: {}
""".format(
    current_year
)
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "nbsphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"
# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# Decides the language used for syntax highlighting of code blocks.
highlight_language = "python3"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#E37B74",
        "color-brand-content": "#E37B74",
        "color-code-background": "#F8F8F8",
        "color-code-border": "#E37B74",
        "color-admonition-background": "#FEECEC",
        "color-link": "#E37B74",
        "color-pre-background": "#F8F8F8",
        "color-pre-border": "#E37B74",
    },
    "dark_css_variables": {
        "color-brand-primary": "#E37B74",
        "color-brand-content": "#E37B74",
        "color-code-background": "#222222",
        "color-code-border": "#E37B74",
        "color-admonition-background": "#331E1C",
        "color-link": "#E37B74",
        "color-pre-background": "#222222",
        "color-pre-border": "#E37B74",
    },
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
}

html_static_path = ["_static"]
html_favicon = "_static/STOUT.svg"
html_logo = "_static/STOUT.png"
