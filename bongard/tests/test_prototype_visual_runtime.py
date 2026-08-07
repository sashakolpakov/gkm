from __future__ import annotations

from pathlib import Path
import re

from bongard.canonical import canonical_digest, canonical_json
from bongard.prototype_visual_runtime import (
    MANIFEST_SCHEMA,
    visual_runtime_dependency_digest,
    visual_runtime_dependency_manifest,
)


def test_visual_runtime_manifest_is_canonical_and_repeatable() -> None:
    first = visual_runtime_dependency_manifest()
    second = visual_runtime_dependency_manifest()
    assert first == second
    assert first["schema"] == MANIFEST_SCHEMA
    assert canonical_json(first) == canonical_json(second)
    assert visual_runtime_dependency_digest() == canonical_digest(first)
    assert re.fullmatch(r"[0-9a-f]{64}", visual_runtime_dependency_digest())


def test_visual_runtime_binds_distributions_modules_and_native_bytes() -> None:
    manifest = visual_runtime_dependency_manifest()
    assert [row["distribution"] for row in manifest["distributions"]] == [
        "numpy",
        "scipy",
        "Pillow",
    ]
    assert [row["module"] for row in manifest["modules"]] == [
        "numpy",
        "numpy._core._multiarray_umath",
        "scipy",
        "scipy.ndimage._nd_image",
        "PIL",
        "PIL._imaging",
        "zlib",
    ]
    for distribution in manifest["distributions"]:
        assert distribution["version"]
        assert distribution["record_file"]["size_bytes"] > 0
        assert distribution["native_files"]
        for file_record in distribution["native_files"]:
            assert Path(file_record["resolved_path"]).is_file()
            assert file_record["size_bytes"] > 0
            assert re.fullmatch(r"[0-9a-f]{64}", file_record["sha256"])
    for module in manifest["modules"]:
        assert Path(module["file"]["resolved_path"]).is_file()
        assert module["file"]["size_bytes"] > 0


def test_visual_runtime_declares_python_only_authority() -> None:
    authority = visual_runtime_dependency_manifest()["authority"]
    assert authority == {
        "predicate_authority_id": (
            "bongard.grounded-multimodal-predicate-authority/python-v1"
        ),
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_identity_or_decision": False,
    }
