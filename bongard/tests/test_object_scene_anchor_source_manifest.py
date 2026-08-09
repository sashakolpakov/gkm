from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from bongard.object_scene_anchor_source_manifest import (
    ObjectSceneAnchorSourceManifest,
    ObjectSceneAnchorSourceManifestError,
    build_object_scene_anchor_source_manifest,
    cold_verify_object_scene_anchor_source_manifest,
    object_scene_anchor_source_manifest_source_digest,
)


ROOT_MODULE = "bongard.object_scene_anchor_benchmark_command"


def _write(root: Path, relative_path: str, source: str) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _source_tree(tmp_path: Path) -> Path:
    _write(tmp_path, "bongard/__init__.py", "PACKAGE_VALUE = 1\n")
    _write(
        tmp_path,
        "bongard/object_scene_anchor_benchmark_command.py",
        """import bongard.alpha
from bongard import beta

raise AssertionError("the manifest builder executed its root")

def delayed_import():
    from bongard.pkg import helper
    return helper
""",
    )
    _write(
        tmp_path,
        "bongard/alpha.py",
        """from .pkg import deep
raise AssertionError("the manifest builder executed alpha")
""",
    )
    _write(tmp_path, "bongard/beta.py", "BETA = 2\n")
    _write(tmp_path, "bongard/pkg/__init__.py", "PKG = True\n")
    _write(tmp_path, "bongard/pkg/deep.py", "DEEP = 3\n")
    _write(tmp_path, "bongard/pkg/helper.py", "HELPER = 4\n")
    return tmp_path


def test_builds_recursive_exact_byte_closure_without_execution(tmp_path: Path) -> None:
    root = _source_tree(tmp_path)

    manifest = build_object_scene_anchor_source_manifest(
        root_module=ROOT_MODULE,
        repository_root=root,
    )

    assert tuple(item.module_name for item in manifest.entries) == (
        "bongard",
        "bongard.alpha",
        "bongard.beta",
        "bongard.object_scene_anchor_benchmark_command",
        "bongard.pkg",
        "bongard.pkg.deep",
        "bongard.pkg.helper",
    )
    assert all(not Path(item.relative_path).is_absolute() for item in manifest.entries)
    beta = next(item for item in manifest.entries if item.module_name == "bongard.beta")
    beta_bytes = (root / beta.relative_path).read_bytes()
    assert beta.source_sha256 == hashlib.sha256(beta_bytes).hexdigest()
    assert beta.source_byte_count == len(beta_bytes)
    assert ObjectSceneAnchorSourceManifest.from_data(manifest.to_data()) == manifest
    assert cold_verify_object_scene_anchor_source_manifest(
        manifest,
        repository_root=root,
        expected_manifest_digest=manifest.manifest_digest,
    ) == manifest


def test_cold_verify_detects_current_byte_drift(tmp_path: Path) -> None:
    root = _source_tree(tmp_path)
    manifest = build_object_scene_anchor_source_manifest(
        root_module=ROOT_MODULE,
        repository_root=root,
    )
    _write(root, "bongard/beta.py", "BETA = 9\n")

    with pytest.raises(
        ObjectSceneAnchorSourceManifestError,
        match="current exact source closure",
    ):
        cold_verify_object_scene_anchor_source_manifest(
            manifest,
            repository_root=root,
            expected_manifest_digest=manifest.manifest_digest,
        )


def test_rejects_unresolved_local_import(tmp_path: Path) -> None:
    _write(tmp_path, "bongard/__init__.py", "")
    _write(
        tmp_path,
        "bongard/object_scene_anchor_benchmark_command.py",
        "import bongard.does_not_exist\n",
    )

    with pytest.raises(ObjectSceneAnchorSourceManifestError, match="unresolved"):
        build_object_scene_anchor_source_manifest(
            root_module=ROOT_MODULE,
            repository_root=tmp_path,
        )


def test_rejects_duplicate_module_and_package_resolution(tmp_path: Path) -> None:
    _write(tmp_path, "bongard/__init__.py", "")
    _write(
        tmp_path,
        "bongard/object_scene_anchor_benchmark_command.py",
        "import bongard.duplicate\n",
    )
    _write(tmp_path, "bongard/duplicate.py", "VALUE = 1\n")
    _write(tmp_path, "bongard/duplicate/__init__.py", "VALUE = 2\n")

    with pytest.raises(ObjectSceneAnchorSourceManifestError, match="duplicate"):
        build_object_scene_anchor_source_manifest(
            root_module=ROOT_MODULE,
            repository_root=tmp_path,
        )


def test_rejects_symlinked_local_source(tmp_path: Path) -> None:
    _write(tmp_path, "bongard/__init__.py", "")
    _write(
        tmp_path,
        "bongard/object_scene_anchor_benchmark_command.py",
        "import bongard.linked\n",
    )
    target = _write(tmp_path, "linked_target.py", "VALUE = 1\n")
    link = tmp_path / "bongard/linked.py"
    try:
        link.symlink_to(target)
    except OSError as exc:  # pragma: no cover - platform permission guard
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(ObjectSceneAnchorSourceManifestError, match="unsafe"):
        build_object_scene_anchor_source_manifest(
            root_module=ROOT_MODULE,
            repository_root=tmp_path,
        )


def test_rejects_tampered_manifest_and_external_digest(tmp_path: Path) -> None:
    root = _source_tree(tmp_path)
    manifest = build_object_scene_anchor_source_manifest(
        root_module=ROOT_MODULE,
        repository_root=root,
    )
    tampered = manifest.to_data()
    tampered["entries"][0]["source_sha256"] = "0" * 64

    with pytest.raises(ObjectSceneAnchorSourceManifestError, match="digest differs"):
        ObjectSceneAnchorSourceManifest.from_data(tampered)
    with pytest.raises(ObjectSceneAnchorSourceManifestError, match="external commitment"):
        cold_verify_object_scene_anchor_source_manifest(
            manifest,
            repository_root=root,
            expected_manifest_digest="f" * 64,
        )


def test_manifest_implementation_source_digest_is_exact_current_bytes() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "object_scene_anchor_source_manifest.py"
    )
    assert object_scene_anchor_source_manifest_source_digest() == hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest()
