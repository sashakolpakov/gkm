"""Focused offline tests for the raw whole-panel soft-atom proposer."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from bongard import panel_soft_proposer as proposer_module
from bongard.canonical import canonical_digest
from bongard.panel_soft_predicate import PANEL_SOFT_ORIENTATIONS
from bongard.panel_soft_proposer import (
    PANEL_SOFT_PROPOSER_PRESENTATION_NAMES,
    PanelSoftProposerArtifact,
    PanelSoftProposerDropCode,
    PanelSoftProposerError,
    PanelSoftProposerStatus,
    panel_soft_proposer_output_schema,
    panel_soft_proposer_prompt,
    propose_panel_soft_atoms,
    verify_panel_soft_proposer_artifact,
)
from bongard.tests.test_prototype_scene_observer import (
    EFFORT,
    LAUNCHER_DIGEST,
    MODEL,
    NO_TOOLS_KWARGS,
    _png,
    _receipt,
)
from bongard.transport import CodexStructuredResult


SUPPORT_IDS = tuple(
    f"bd/bd_panel_soft_fixture_0000/{side}/{index}.png"
    for side in (1, 0)
    for index in range(1, 7)
)
SUPPORT_IDS_WITH_ZERO = tuple(
    f"bd/bd_panel_soft_fixture_0000/{side}/{index}.png"
    for side in (1, 0)
    for index in (0, 1, 2, 4, 5, 6)
)


def _supports() -> tuple[bytes, ...]:
    return tuple(_png(index + 31) for index in range(12))


def _payload() -> dict[str, str]:
    rows = (
        (
            "A bird-like silhouette spans the complete drawing.",
            "Swept wings form a pointed silhouette.",
            "An arched body creates a bird-like outline.",
        ),
        (
            "The complete drawing features oblique angles.",
            "Slanted corners create a sharp zigzag.",
            "Diagonal strokes converge at acute corners.",
        ),
        (
            "A smooth-bend contour shapes the complete figure.",
            "A broad curve sweeps across the outline.",
            "Rounded turns continue through the figure.",
        ),
        (
            "Nested loops structure the complete drawing.",
            "A small loop occupies an enclosed region.",
            "Concentric outlines occupy the central region.",
        ),
        (
            "A compact spiral fills the complete drawing.",
            "A curled stroke circles the central region.",
            "A tight coil shapes the complete figure.",
        ),
        (
            "A wavy contour dominates the complete figure.",
            "Repeated bends ripple across the silhouette.",
            "Curved turns produce a flowing outline.",
        ),
        (
            "Radial spokes define the complete drawing.",
            "Several rays extend from a central hub.",
            "Straight strokes fan across the figure.",
        ),
        (
            "Touching lobes form the complete silhouette.",
            "Rounded bulges meet at a narrow contact.",
            "Paired arcs shape a lobed outline.",
        ),
    )
    result: dict[str, str] = {}
    for global_index, (phrase, witness_a, witness_b) in enumerate(rows):
        side, rank = divmod(global_index, 4)
        result[f"side{side}_atom{rank}_phrase"] = phrase
        result[f"side{side}_atom{rank}_witness_a"] = witness_a
        result[f"side{side}_atom{rank}_witness_b"] = witness_b
    return result


def _run(
    payload: dict[str, object] | None = None,
    *,
    fail: bool = False,
    support_ids: tuple[str, ...] = SUPPORT_IDS,
    expected_calls: int = 1,
):
    panels = _supports()
    output = _payload() if payload is None else payload
    calls = 0

    def transport(prompt, paths, names, schema, **kwargs):
        nonlocal calls
        calls += 1
        assert names == PANEL_SOFT_PROPOSER_PRESENTATION_NAMES
        assert len(paths) == 12
        assert tuple(Path(path).read_bytes() for path in paths) == panels
        assert "group 0" in prompt and "group 1" in prompt
        assert not any(panel_id in prompt for panel_id in support_ids)
        if fail:
            raise RuntimeError("synthetic transport failure")
        return CodexStructuredResult(
            dict(output), _receipt(prompt, paths, names, schema, output)
        )

    artifact = propose_panel_soft_atoms(
        panels,
        support_panel_ids=support_ids,
        expected_support_sha256=tuple(
            hashlib.sha256(item).hexdigest() for item in panels
        ),
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        **NO_TOOLS_KWARGS,
        transport=transport,
    )
    assert calls == expected_calls
    return artifact, panels


def test_success_round_trip_and_model_free_cold_verification() -> None:
    artifact, panels = _run()
    assert artifact.status is PanelSoftProposerStatus.SUCCESS
    assert artifact.logical_proposer_attempt_count == 1
    assert artifact.transport_invocation_count == 1
    assert artifact.receipted_call_count == 1
    assert artifact.support_panel_ids == SUPPORT_IDS
    assert artifact.vocabulary is not None
    assert tuple(item.atom_id for item in artifact.vocabulary.atoms) == tuple(
        f"atom_{index:04d}" for index in range(8)
    )
    assert {item.orientation for item in artifact.vocabulary.atoms} == set(
        PANEL_SOFT_ORIENTATIONS
    )
    assert artifact.raw_proposer_evidence_digest != artifact.artifact_digest
    assert all(
        item.proposer_artifact_digest == artifact.raw_proposer_evidence_digest
        for item in artifact.vocabulary.atoms
    )
    assert PanelSoftProposerArtifact.from_data(artifact.to_data()) == artifact
    assert verify_panel_soft_proposer_artifact(
        artifact,
        panels,
        support_panel_ids=SUPPORT_IDS,
        expected_artifact_digest=artifact.artifact_digest,
    ) == artifact


def test_fixed_schema_prompt_and_neutral_raw_presentation() -> None:
    schema = panel_soft_proposer_output_schema()
    assert set(schema["properties"]) == set(schema["required"])
    assert len(schema["properties"]) == 24
    assert "minItems" not in repr(schema) and "maxItems" not in repr(schema)
    prompt = panel_soft_proposer_prompt()
    for text in ("Bird-like", "oblique angles", "smooth-bend", "'and'", "'or'"):
        assert text in prompt
    artifact, _ = _run()
    assert tuple(item.name for item in artifact.presentation) == (
        PANEL_SOFT_PROPOSER_PRESENTATION_NAMES
    )


def test_one_bad_row_is_dropped_without_killing_its_orientation() -> None:
    payload = _payload()
    payload["side0_atom0_phrase"] = "This row has no bird shape."
    artifact, _ = _run(payload)
    assert artifact.status is PanelSoftProposerStatus.SUCCESS
    assert artifact.vocabulary is not None and len(artifact.vocabulary.atoms) == 7
    assert artifact.drops[0].code is PanelSoftProposerDropCode.SEMANTIC_ROW_REJECTED
    assert artifact.drops[0].orientation == "side0_positive"


def test_duplicate_semantic_row_preserves_first_and_drops_later() -> None:
    payload = _payload()
    for field in ("phrase", "witness_a", "witness_b"):
        payload[f"side0_atom1_{field}"] = payload[f"side0_atom0_{field}"]
    artifact, _ = _run(payload)
    assert artifact.status is PanelSoftProposerStatus.SUCCESS
    assert artifact.vocabulary is not None and len(artifact.vocabulary.atoms) == 7
    assert artifact.drops == (
        artifact.drops[0],
    )
    assert artifact.drops[0].raw_rank == 1
    assert artifact.drops[0].code is PanelSoftProposerDropCode.DUPLICATE_SEMANTIC_ROW


def test_empty_orientation_and_transport_failure_never_fabricate_vocabulary() -> None:
    payload = _payload()
    for rank in range(4):
        payload[f"side1_atom{rank}_phrase"] = "This row has no usable shape."
    parser_error, _ = _run(payload)
    transport_error, _ = _run(fail=True)
    assert parser_error.status is PanelSoftProposerStatus.PARSER_ERROR
    assert parser_error.vocabulary is None
    assert parser_error.receipt is not None
    assert parser_error.transport_invocation_count == 1
    assert parser_error.receipted_call_count == 1
    assert transport_error.status is PanelSoftProposerStatus.TRANSPORT_ERROR
    assert transport_error.vocabulary is None
    assert transport_error.receipt is None
    assert transport_error.transport_invocation_count == 1
    assert transport_error.receipted_call_count == 0


def test_support_layout_and_cold_pixel_tamper_fail_closed() -> None:
    artifact, panels = _run()
    arbitrary, _ = _run(support_ids=SUPPORT_IDS_WITH_ZERO)
    assert arbitrary.support_panel_ids == SUPPORT_IDS_WITH_ZERO
    for replacement in (
        "bd/bd_other_fixture_0000/1/1.png",
        "ff/ff_panel_soft_fixture_0000/1/1.png",
    ):
        bad_ids = list(SUPPORT_IDS)
        bad_ids[0] = replacement
        with pytest.raises(PanelSoftProposerError):
            propose_panel_soft_atoms(
                panels,
                support_panel_ids=bad_ids,
                expected_support_sha256=tuple(hashlib.sha256(item).hexdigest() for item in panels),
                model=MODEL,
                reasoning_effort=EFFORT,
                expected_launcher_digest=LAUNCHER_DIGEST,
                **NO_TOOLS_KWARGS,
                transport=lambda *args, **kwargs: None,
            )
    altered = list(panels)
    altered[0] = _png(99)
    with pytest.raises(PanelSoftProposerError):
        verify_panel_soft_proposer_artifact(
            artifact,
            altered,
            support_panel_ids=SUPPORT_IDS,
            expected_artifact_digest=artifact.artifact_digest,
        )


def test_pretransport_staging_failure_records_zero_transport_invocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_before_transport(*args, **kwargs):
        raise RuntimeError("synthetic pretransport staging failure")

    monkeypatch.setattr(proposer_module._scene_runtime, "_stage_and_call", fail_before_transport)
    artifact, _ = _run(expected_calls=0)
    assert artifact.status is PanelSoftProposerStatus.TRANSPORT_ERROR
    assert artifact.logical_proposer_attempt_count == 1
    assert artifact.transport_invocation_count == 0
    assert artifact.receipted_call_count == 0


@pytest.mark.parametrize(
    "field",
    (
        "logical_proposer_attempt_count",
        "transport_invocation_count",
        "receipted_call_count",
    ),
)
def test_boolean_call_counts_are_not_canonical_integers(field: str) -> None:
    artifact, _ = _run()
    data = deepcopy(artifact.to_data())
    data[field] = True
    data["artifact_digest"] = canonical_digest(
        {key: value for key, value in data.items() if key != "artifact_digest"}
    )
    with pytest.raises(PanelSoftProposerError, match="presentation, status, or drops"):
        PanelSoftProposerArtifact.from_data(data)
