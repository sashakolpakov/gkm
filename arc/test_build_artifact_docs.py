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
