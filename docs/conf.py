"""Sphinx configuration.

Generated ARC pages are refreshed explicitly with ``arc/build_artifact_docs.py``.
Importing the documentation configuration is intentionally read-only.
"""

project = "GKM"
author = "Alexander Kolpakov"

extensions = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_title = "GKM: Free-Energy Evolution"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_extra_path = [".nojekyll"]

rst_epilog = """
.. |repo| replace:: GKM
"""
