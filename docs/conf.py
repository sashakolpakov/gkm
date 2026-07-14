import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
subprocess.run(["python3", str(ROOT / "arc" / "build_artifact_docs.py")], check=True)

project = "GKM"
author = "GKM contributors"

extensions = []
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_title = "GKM: Free-Energy Evolution"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_extra_path = [".nojekyll"]

rst_epilog = """
.. |repo| replace:: GKM
"""
