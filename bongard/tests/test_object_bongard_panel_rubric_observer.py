"""Focused offline tests for the one-image whole-panel rubric observer."""

from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
from pathlib import Path
import re
import subprocess
import sys

import pytest

from bongard.canonical import canonical_digest
from bongard.object_bongard_panel_rubric_observer import (
    ObjectBongardPanelRubricArtifact,
    ObjectBongardPanelRubricObservation,
    ObjectBongardPanelRubricObserverError,
    PANEL_RUBRIC_ORDINAL_LEVEL_ANCHORS,
    PanelRubricDisposition,
    classify_panel_rubric_interval,
    object_bongard_panel_rubric_prompt,
    observe_object_bongard_panel_rubric,
    verify_object_bongard_panel_rubric_artifact,
)
from bongard.object_bongard_rubric_language import (
    ObjectBongardRubricSpec,
    OrdinalLevelInterval,
    object_bongard_rubric_language_source_digest,
)
from bongard.object_bongard_soft_cues import ObjectBongardSoftCue
from bongard.tests.test_prototype_scene_observer import (
    EFFORT,
    LAUNCHER_DIGEST,
    MODEL,
    NO_TOOLS_KWARGS,
    _png,
    _receipt,
)
from bongard.transport import CodexStructuredResult


PANEL_ID = "bd/bd_panel_rubric_fixture_0000/0/0.png"
SEMANTIC_DIGEST = "a" * 64


def _spec() -> ObjectBongardRubricSpec:
    return ObjectBongardRubricSpec.from_soft_cues(
        SEMANTIC_DIGEST,
        ObjectBongardSoftCue.create(
            "One decorated figure forms two closed loops touching at one vertex."
        ),
        ObjectBongardSoftCue.create(
            "One decorated figure forms a closed loop with a dangling branch."
        ),
        0,
    )


def _observe(
    lower: int = 3,
    upper: int = 4,
    *,
    fail: bool = False,
    malformed: bool = False,
):
    panel = _png(31)
    spec = _spec()
    calls = 0

    def transport(prompt, paths, names, schema, **kwargs):
        nonlocal calls
        calls += 1
        assert len(paths) == 1
        assert names == ("panel.png",)
        assert Path(paths[0]).read_bytes() == panel
        assert spec.rubric in prompt
        assert "Level 4 is reserved" in prompt
        assert "Level 0 is reserved" in prompt
        assert "both, neither, a tie" in prompt
        if fail:
            raise RuntimeError("synthetic transport failure")
        payload = (
            {"lower": 1, "upper": 7}
            if malformed
            else {"lower": lower, "upper": upper}
        )
        return CodexStructuredResult(
            payload, _receipt(prompt, paths, names, schema, payload)
        )

    artifact = observe_object_bongard_panel_rubric(
        panel,
        panel_id=PANEL_ID,
        rubric_spec=spec,
        expected_panel_sha256=hashlib.sha256(panel).hexdigest(),
        expected_rubric_spec_digest=spec.spec_digest,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        **NO_TOOLS_KWARGS,
        transport=transport,
    )
    assert calls == 1
    return artifact, panel, spec


def test_one_panel_call_round_trip_and_model_free_cold_replay() -> None:
    artifact, panel, spec = _observe()
    assert artifact.physical_call_count == 1
    assert tuple(item.name for item in artifact.presentation) == ("panel.png",)
    assert artifact.observation.disposition is PanelRubricDisposition.PRESENT
    assert artifact.observation.interval == OrdinalLevelInterval(3, 4)
    assert ObjectBongardPanelRubricArtifact.from_data(artifact.to_data()) == artifact
    assert verify_object_bongard_panel_rubric_artifact(
        artifact,
        panel,
        panel_id=PANEL_ID,
        rubric_spec=spec,
        expected_artifact_digest=artifact.artifact_digest,
        expected_runtime_identity_digest=artifact.runtime_identity_digest,
    ) == artifact


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    (
        (3, 3, PanelRubricDisposition.PRESENT),
        (3, 4, PanelRubricDisposition.PRESENT),
        (0, 0, PanelRubricDisposition.CERTIFIED_ABSENT),
        (0, 1, PanelRubricDisposition.CERTIFIED_ABSENT),
        (2, 2, PanelRubricDisposition.INDETERMINATE),
        (1, 3, PanelRubricDisposition.INDETERMINATE),
        (0, 4, PanelRubricDisposition.INDETERMINATE),
    ),
)
def test_fixed_python_four_disposition_projection(
    lower: int, upper: int, expected: PanelRubricDisposition
) -> None:
    interval = OrdinalLevelInterval(lower, upper)
    assert classify_panel_rubric_interval(interval) is expected
    observation = ObjectBongardPanelRubricObservation.from_interval(
        _spec().spec_digest, interval
    )
    assert observation.disposition is expected
    assert ObjectBongardPanelRubricObservation.from_data(
        observation.to_data()
    ) == observation


def test_transport_and_payload_failures_are_error_not_absence() -> None:
    transport_error, _, _ = _observe(fail=True)
    parser_error, _, _ = _observe(malformed=True)
    for artifact in (transport_error, parser_error):
        assert artifact.observation.disposition is PanelRubricDisposition.ERROR
        assert artifact.observation.interval is None
        assert artifact.failure_code == artifact.observation.error_code
    assert transport_error.receipt is None
    assert parser_error.receipt is not None


def test_receipt_prevents_resealed_payload_and_projection_tamper() -> None:
    artifact, _, _ = _observe(0, 1)
    data = deepcopy(artifact.to_data())
    data["model_payload"] = {"lower": 4, "upper": 4}
    replacement = ObjectBongardPanelRubricObservation.from_interval(
        artifact.rubric_spec_digest, OrdinalLevelInterval(4, 4)
    )
    data["observation"] = replacement.to_data()
    data["artifact_digest"] = canonical_digest(
        {key: value for key, value in data.items() if key != "artifact_digest"}
    )
    with pytest.raises(ObjectBongardPanelRubricObserverError, match="receipt"):
        ObjectBongardPanelRubricArtifact.from_data(data)


def test_prompt_is_whole_panel_only_and_has_no_experimental_role_words() -> None:
    prompt = object_bongard_panel_rubric_prompt(_spec())
    for _, meaning in PANEL_RUBRIC_ORDINAL_LEVEL_ANCHORS:
        assert meaning in prompt
    assert "complete panel" in prompt
    lowered = prompt.lower()
    for word in ("label", "query", "candidate", "formula", "predicate"):
        assert re.search(rf"\b{word}s?\b", lowered) is None


def test_panel_path_has_no_atlas_geometry_or_lean_import() -> None:
    root = Path(__file__).parents[1]
    for filename in (
        "object_bongard_rubric_language.py",
        "object_bongard_panel_rubric_observer.py",
    ):
        tree = ast.parse((root / filename).read_text(encoding="utf-8"))
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imports.append(node.module)
        lowered = tuple(item.lower() for item in imports)
        assert not any("lean" in item for item in lowered)
        assert not any("atlas" in item for item in lowered)
        assert not any("hypoth" in item or "lineage" in item for item in lowered)
        assert "bongard.object_bongard_rubric_observer" not in lowered
        assert "bongard.prototype_object_scene_observer" not in lowered

    artifact, _, _ = _observe()
    assert artifact.rubric_language_source_digest == (
        object_bongard_rubric_language_source_digest()
    )

    forbidden = (
        "bongard.object_bongard_rubric_observer",
        "bongard.prototype_object_scene_observer",
        "bongard.prototype_object_hypotheses",
        "bongard.prototype_object_lineages",
    )
    subprocess.run(
        (
            sys.executable,
            "-c",
            "import sys; "
            "import bongard.object_bongard_panel_rubric_observer; "
            f"forbidden={forbidden!r}; "
            "assert not [name for name in forbidden if name in sys.modules]",
        ),
        cwd=Path(__file__).parents[2],
        check=True,
    )
