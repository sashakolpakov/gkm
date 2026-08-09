"""Offline full-receipt tests for the typed panel production adapter."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from bongard.panel_feature_observation import FeatureAxis
from bongard.panel_feature_observer_protocol import FeatureAxisObservationView
from bongard.panel_soft_ontology import (
    ComponentCountParameters,
    FeatureFamily,
    GestaltResemblanceParameters,
    NativeOrientation,
    PanelFeatureSpec,
    ReferenceFrame,
    SubjectScope,
    ClosedCount,
    GestaltKind,
)
from bongard.panel_typed_codex_observer import (
    HeadlessCodexPanelFeatureReceiptedCall,
    PanelTypedCodexObserverError,
    TypedAxisCodexArtifact,
    TypedOwnerCodexArtifact,
    TypedProposerCodexCallArtifact,
    build_panel_only_observation_context,
    invoke_receipted_panel_feature_proposer,
    observe_typed_panel_axis,
    observe_typed_panel_owners,
    verify_typed_axis_codex_artifact,
    verify_typed_owner_codex_artifact,
    verify_typed_proposer_codex_artifact,
)
from bongard.panel_owner_inventory import PANEL_OWNER_SLOT_NAMES
from bongard.tests.test_panel_feature_proposer import _payload as _proposer_payload
from bongard.tests.test_prototype_scene_observer import (
    EFFORT,
    LAUNCHER_DIGEST,
    MODEL,
    NO_TOOLS_KWARGS,
    _png,
    _receipt,
)
from bongard.transport import CodexStructuredResult


TASK_CONTEXT = hashlib.sha256(b"typed-proposer-task-context").hexdigest()


def _unused() -> dict[str, object]:
    return {
        "slot_state": "unused",
        "owner_kind": "not_applicable",
        "parent_slot": "not_applicable",
        "x_min": -1,
        "y_min": -1,
        "x_max": -1,
        "y_max": -1,
    }


def _owner_payload() -> dict[str, object]:
    slots = {name: _unused() for name in PANEL_OWNER_SLOT_NAMES}
    slots["slot_00"] = {
        "slot_state": "owner",
        "owner_kind": "figure",
        "parent_slot": "root",
        "x_min": 2,
        "y_min": 2,
        "x_max": 13,
        "y_max": 13,
    }
    return {"inventory_status": "complete", "slots": slots}


def _transport(payload, *, expected_images: tuple[bytes, ...]):
    def call(prompt, paths, names, schema, **kwargs):
        assert tuple(names) == tuple(Path(path).name for path in paths)
        assert tuple(Path(path).read_bytes() for path in paths) == expected_images
        return CodexStructuredResult(
            deepcopy(payload),
            _receipt(prompt, paths, names, schema, payload),
        )

    return call


def _owner_artifact(panel: bytes) -> TypedOwnerCodexArtifact:
    payload = _owner_payload()
    return observe_typed_panel_owners(
        panel,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        **NO_TOOLS_KWARGS,
        transport=_transport(payload, expected_images=(panel,)),
    )


def _axis_payload(
    view: FeatureAxisObservationView,
    *,
    variant_index: int = 0,
) -> dict[str, object]:
    assert view.bindings
    variant = view.variants[variant_index].alias
    return {
        binding.alias: {
            "resolution": "complete",
            "variant_aliases": [variant],
            "evidence_x": binding.search_region.minimum.x,
            "evidence_y": binding.search_region.minimum.y,
            "issue": "none",
        }
        for binding in view.bindings
    }


def test_owner_full_receipt_round_trip_and_exact_png_cold_replay() -> None:
    panel = _png(31)
    artifact = _owner_artifact(panel)
    assert artifact.inventory_artifact.receipt.transport_receipt_digest == (
        artifact.codex_receipt.receipt_digest
    )
    assert artifact.inventory_artifact.receipt.receipt_digest != (
        artifact.codex_receipt.receipt_digest
    )
    assert TypedOwnerCodexArtifact.from_data(artifact.to_data()) == artifact
    assert verify_typed_owner_codex_artifact(
        artifact,
        panel,
        expected_artifact_digest=artifact.artifact_digest,
    ) == artifact
    with pytest.raises(PanelTypedCodexObserverError):
        verify_typed_owner_codex_artifact(
            artifact,
            _png(32),
            expected_artifact_digest=artifact.artifact_digest,
        )


def test_owner_local_axis_binds_view_payload_rows_observation_and_owner_source() -> None:
    panel = _png(33)
    owner = _owner_artifact(panel)
    spec = PanelFeatureSpec(
        FeatureFamily.GESTALT_RESEMBLANCE,
        SubjectScope.ONE_COHERENT_FIGURE,
        ReferenceFrame.NONE,
        GestaltResemblanceParameters(GestaltKind.BIRD_LIKE),
    )
    axis = FeatureAxis.for_spec(spec)
    view = FeatureAxisObservationView.build(owner.to_owner_inventory(), axis)
    payload = _axis_payload(view)
    artifact = observe_typed_panel_axis(
        panel,
        axis=axis,
        owner_artifact=owner,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        **NO_TOOLS_KWARGS,
        transport=_transport(payload, expected_images=(panel,)),
    )
    assert artifact.source_artifact_digest == owner.artifact_digest
    assert artifact.view == view
    assert artifact.row_receipt_digests == (
        artifact.codex_receipt.receipt_digest,
    )
    assert TypedAxisCodexArtifact.from_data(artifact.to_data()) == artifact
    assert verify_typed_axis_codex_artifact(
        artifact,
        panel,
        owner_artifact=owner,
        expected_artifact_digest=artifact.artifact_digest,
    ) == artifact
    with pytest.raises(PanelTypedCodexObserverError):
        verify_typed_axis_codex_artifact(
            artifact,
            panel,
            expected_artifact_digest=artifact.artifact_digest,
        )


def test_whole_panel_axis_uses_no_owner_enumeration_call() -> None:
    panel = _png(34)
    context = build_panel_only_observation_context(
        panel,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        **NO_TOOLS_KWARGS,
    )
    spec = PanelFeatureSpec(
        FeatureFamily.COMPONENT_COUNT,
        SubjectScope.WHOLE_PANEL,
        ReferenceFrame.NONE,
        ComponentCountParameters(ClosedCount.ONE),
    )
    axis = FeatureAxis.for_spec(spec)
    view = FeatureAxisObservationView.build(context.to_owner_inventory(), axis)
    payload = _axis_payload(view)
    calls = 0

    def transport(prompt, paths, names, schema, **kwargs):
        nonlocal calls
        calls += 1
        return _transport(payload, expected_images=(panel,))(
            prompt, paths, names, schema, **kwargs
        )

    artifact = observe_typed_panel_axis(
        panel,
        axis=axis,
        panel_only_context=context,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        **NO_TOOLS_KWARGS,
        transport=transport,
    )
    assert calls == 1
    assert artifact.source_kind == "panel_only"
    assert artifact.panel_only_context == context
    assert artifact.view.inventory.enumeration_complete is False
    assert verify_typed_axis_codex_artifact(
        artifact,
        panel,
        expected_artifact_digest=artifact.artifact_digest,
    ) == artifact


def test_proposer_one_shot_full_receipt_manifest_and_cold_replay() -> None:
    panels = tuple(_png(40 + index) for index in range(12))
    payload = _proposer_payload()
    call = HeadlessCodexPanelFeatureReceiptedCall(
        task_context_digest=TASK_CONTEXT,
        block_orientations=(
            NativeOrientation.SIDE0_POSITIVE,
            NativeOrientation.SIDE1_POSITIVE,
        ),
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        **NO_TOOLS_KWARGS,
        transport=_transport(payload, expected_images=panels),
    )
    result = invoke_receipted_panel_feature_proposer(panels, call=call)
    artifact = call.artifact
    assert artifact.codex_receipt.structured_output_digest == artifact.payload_digest
    assert all(
        item.proposal.provenance.proposer_receipt_digest == artifact.artifact_digest
        for item in result.nominations
    )
    assert TypedProposerCodexCallArtifact.from_data(artifact.to_data()) == artifact
    assert verify_typed_proposer_codex_artifact(
        artifact,
        panels,
        expected_artifact_digest=artifact.artifact_digest,
    ) == artifact
    with pytest.raises(PanelTypedCodexObserverError, match="exactly-once"):
        invoke_receipted_panel_feature_proposer(panels, call=call)

    tampered = deepcopy(artifact.to_data())
    tampered["block_orientation_manifest"]["blocks"][0][
        "native_orientation"
    ] = NativeOrientation.SIDE1_POSITIVE.value
    with pytest.raises(PanelTypedCodexObserverError):
        TypedProposerCodexCallArtifact.from_data(tampered)
