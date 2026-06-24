# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from pathlib import Path

sys.path.insert(0, str(Path("..", "src").resolve()))

version_file = Path("..", "src", "sedtrails", "__version__.py").resolve()
version_namespace = {}
exec(version_file.read_text(encoding="utf-8"), version_namespace)

project = "SedTRAILS"
copyright = "2025-2026, SedTRAILS Team"
author = "SedTRAILS Team"
release = version_namespace["__version__"]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx_rtd_theme",
    "autodoc2",
]

autodoc2_packages = ["../src/sedtrails"]
autodoc2_annotations = True
autodoc2_class_inheritance = False
autodoc2_docstrings = "direct"
autodoc2_docstring_parser_regexes = [
    (r".*", "myst"),
]
suppress_warnings = ["autodoc2.dup_item", "myst.header"]
nitpick_ignore_regex = [
    ("py:class", r"numpy\..*"),
    ("py:class", r"sedtrails\.transport_converter\.format_converter\.SedtrailsData"),
]

myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_image",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "notes"]
language = "en"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/img/mascot.png"

# Custom CSS files
html_css_files = [
    "css/custom.css",
]


html_theme_options = {
    "logo_only": False,  # Show only logo, not project name
    "collapse_navigation": False,  # Keep navigation expanded
    "navigation_depth": 4,  # How deep to show navigation
}
