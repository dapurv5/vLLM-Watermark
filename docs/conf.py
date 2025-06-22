# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "vLLM-Watermark"
copyright = "2024, Your Name"
author = "Your Name"

# The full version, including alpha/beta/rc tags
release = "0.1.0"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.githubpages",
    "sphinx_rtd_theme",
    "myst_parser",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "navigation_depth": 4,
    "titles_only": False,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "blob",
    "style_nav_header_background": "#2980B9",
    # Toc options
    "canonical_url": "https://vermaapurv.com/vLLM-Watermark/",
    "analytics_id": "",  #  Provide your Google Analytics ID
    "analytics_anonymize_ip": False,
}

# Base URL for the HTML pages.
# This is used to generate correct links to static files.
html_baseurl = "/vLLM-Watermark/"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "../resources/vLLM-WM-Logo.png"

# The name of an image file (relative to this directory) to use as a favicon of
# the docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "../resources/vLLM-WM-Logo.png"

# -- Options for LaTeX output ------------------------------------------------

# LaTeX configuration for math equations
latex_elements = {
    "preamble": r"""
        \usepackage{amsmath}
        \usepackage{amssymb}
        \usepackage{mathtools}
    """,
    "figure_align": "H",
}

# -- Options for autodoc ----------------------------------------------------

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"

# Don't show typehints in the signature
autodoc_typehints_format = "short"

# Include both class docstring and __init__ docstring
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# -- Options for intersphinx mapping ----------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "transformers": ("https://huggingface.co/docs/transformers/", None),
    "vllm": ("https://docs.vllm.ai/", None),
}

# -- Options for MathJax ----------------------------------------------------

# MathJax 3 configuration for better LaTeX rendering
mathjax3_config = {
    "tex": {
        "equationNumbers": {"autoNumber": "AMS"},
        "extensions": ["AMSmath.js", "AMSsymbols.js"],
    }
}

# -- Options for todo extension ---------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Options for napoleon extension -----------------------------------------

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_custom_sections = None

# -- Options for myst-parser ------------------------------------------------

# Enable MyST extensions for better Markdown support
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# -- Custom CSS -------------------------------------------------------------


def setup(app):
    app.add_css_file("custom.css")
