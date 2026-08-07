from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from bongard.canonical import canonical_digest, canonical_json


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_canonical_json_has_one_exact_encoding() -> None:
    value = {
        "unicode": "oblique ∠",
        "nested": ({"z": 2, "a": 1}, True, None),
        "number": 0.125,
    }
    expected = (
        b'{"nested":[{"a":1,"z":2},true,null],"number":0.125,'
        b'"unicode":"oblique \\u2220"}'
    )
    # ``ensure_ascii=False`` retains the UTF-8 symbol rather than the escaped
    # spelling used in the readable bytes literal above.
    expected = expected.replace(b"\\u2220", "∠".encode("utf-8"))
    assert canonical_json(value) == expected
    assert canonical_digest(value) == hashlib.sha256(expected).hexdigest()
    assert json.loads(expected) == {
        "unicode": "oblique ∠",
        "nested": [{"z": 2, "a": 1}, True, None],
        "number": 0.125,
    }


@pytest.mark.parametrize(
    "value, message",
    [
        ({1: "non-string key"}, "object keys must be strings"),
        ({"bad": float("nan")}, "non-finite float"),
        ({"bad": float("inf")}, "non-finite float"),
        ({"bad": {1, 2}}, "unsupported canonical JSON value set"),
    ],
)
def test_canonical_json_rejects_noncanonical_values(
    value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        canonical_json(value)


def test_retained_visual_stack_does_not_initialize_superseded_pipelines() -> None:
    script = f"""
import builtins
import importlib
import sys

sys.path.insert(0, {str(REPOSITORY_ROOT)!r})
forbidden = (
    "bongard.admission",
    "bongard.artifacts",
    "bongard.atomic_",
    "bongard.benchmark",
    "bongard.ir",
    "bongard.predicate_backend",
    "bongard.proposer",
    "bongard.prototype_",
    "bongard.semantic_",
    "bongard.soft_predicates",
    "bongard.support_prototypes",
    "bongard.synthesis",
)
real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if any(name == item or name.startswith(item) for item in forbidden):
        raise AssertionError("superseded import attempted: " + name)
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
for name in (
    "bongard.canonical",
    "bongard.contour_witnesses",
    "bongard.visual_witnesses",
    "bongard.visual_witness_bundle",
    "bongard.loop_geometry",
    "bongard.point_contact",
    "bongard.loop_scene_witnesses",
    "bongard.relational_scene",
    "bongard.relational_visual_query",
    "bongard.composite_visual_packet",
    "bongard.closed_visual_predicates",
    "bongard.triangle_geometry",
    "bongard.vision_tags",
):
    importlib.import_module(name)

loaded = sorted(
    name for name in sys.modules
    if any(name == item or name.startswith(item) for item in forbidden)
)
if loaded:
    raise AssertionError("superseded modules loaded: " + repr(loaded))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_legacy_facade_exports_are_lazy_but_compatible() -> None:
    script = f"""
import sys
sys.path.insert(0, {str(REPOSITORY_ROOT)!r})
import bongard
import bongard.legs
assert "bongard.soft_predicates" not in sys.modules
assert "bongard.support_prototypes" not in sys.modules
assert "bongard.prototype_episode" not in sys.modules
assert "bongard.legs.neutral_features" not in sys.modules
assert bongard.FrozenVisualScore.__module__ == "bongard.soft_predicates"
assert "bongard.soft_predicates" in sys.modules
assert bongard.legs.NEUTRAL_FEATURE_ALGORITHM_ID
assert "bongard.legs.neutral_features" in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
