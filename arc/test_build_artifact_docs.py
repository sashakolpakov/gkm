from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_artifact_doc_builder_does_not_import_campaign_runtime() -> None:
    script = Path(__file__).with_name("build_artifact_docs.py")
    code = (
        "import runpy,sys; "
        f"ns=runpy.run_path({str(script)!r}); "
        "assert 'gkm_legs' not in sys.modules; "
        "assert ns['MARGINAL_COMPLEXITY_CONTRACT']['field']=='marginal_C'"
    )
    subprocess.run([sys.executable, "-I", "-c", code], check=True)


def test_generated_artifact_docs_are_fresh() -> None:
    """Render both publication fragments and compare their checked-in bytes."""
    script = Path(__file__).with_name("build_artifact_docs.py")
    root = script.parents[1]
    docs_output = root / "docs" / "generated" / "arc_artifacts.rst"
    tex_output = root / "arc" / "manuscript" / "generated" / "arc_artifacts.tex"
    code = (
        "import runpy,sys; "
        f"ns=runpy.run_path({str(script)!r}); "
        "artifacts=tuple(ns['load_checkpoint_artifact'](game) "
        "for game in ns['AUTHORITATIVE_LEVELS']); "
        f"assert open({str(docs_output)!r},encoding='utf-8').read()"
        "==ns['render_rst'](artifacts); "
        f"assert open({str(tex_output)!r},encoding='utf-8').read()"
        "==ns['render_tex'](artifacts); "
        "assert 'gkm_legs' not in sys.modules"
    )
    subprocess.run([sys.executable, "-I", "-c", code], check=True)
