"""Offline custody tests for the panel-feature full-receipt evidence bundle."""

from __future__ import annotations

from copy import deepcopy
import json

import pytest

from bongard import panel_batched_typed_codex_observer as _batch_observer
from bongard.panel_batched_typed_codex_observer import (
    BatchedFeatureAxisRequest,
    observe_typed_panel_axes_batched,
)
from bongard.panel_feature_evidence_bundle import (
    PanelFeatureEvidenceBundle,
    PanelFeatureEvidenceBundleError,
    PanelFeatureEvidencePanel,
    PanelFeatureEvidencePhase,
    cold_replay_panel_feature_evidence_bundle,
    _vocabulary_axes,
)
from bongard.panel_feature_observation import FeatureAxis, PanelFeatureObservationSet
from bongard.panel_feature_observer_protocol import FeatureAxisObservationView
from bongard.panel_feature_proposer import (
    PANEL_FEATURE_BLOCKS,
    PANEL_FEATURE_SLOTS_PER_DIRECTION,
    PanelFeatureProposerResult,
)
from bongard.panel_soft_ontology import (
    FeatureFamily,
    GestaltKind,
    GestaltResemblanceParameters,
    NativeOrientation,
    PanelFeatureSpec,
    ReferenceFrame,
    SubjectScope,
)
from bongard.panel_typed_codex_observer import (
    HeadlessCodexPanelFeatureReceiptedCall,
    build_panel_only_observation_context,
    invoke_receipted_panel_feature_proposer,
    observe_typed_panel_axis,
)
from bongard.tests.test_panel_feature_proposer import (
    _payload as _proposer_payload,
    _row as _proposer_row,
)
from bongard.tests.test_panel_batched_typed_codex_observer import (
    _payload as _batched_payload,
    _transport as _batched_transport,
)
from bongard.tests.test_panel_typed_codex_observer import (
    TASK_CONTEXT,
    _axis_payload,
    _owner_artifact,
    _transport,
)
from bongard.tests.test_prototype_scene_observer import (
    EFFORT,
    LAUNCHER_DIGEST,
    MODEL,
    NO_TOOLS_KWARGS,
    _png,
)


def _proposer(panels: tuple[bytes, ...], payload: dict[str, object]):
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
    return call.artifact, result


def test_typed_empty_proposer_outcome_does_not_erase_fixed_observer_evidence() -> None:
    result = PanelFeatureProposerResult(
        "0" * 64,
        "1" * 64,
        (),
        (),
        (),
        None,
    )
    assert _vocabulary_axes(result) == ()


def _whole_panel_bundle() -> PanelFeatureEvidenceBundle:
    support_pngs = tuple(_png(500 + index) for index in range(12))
    query_pngs = tuple(_png(700 + index) for index in range(2))
    proposer, result = _proposer(support_pngs, _proposer_payload())
    assert result.observer_vocabulary is not None
    axes = tuple(
        {
            FeatureAxis.for_spec(spec).axis_digest: FeatureAxis.for_spec(spec)
            for spec in result.observer_vocabulary.specs
        }[key]
        for key in sorted(
            {
                FeatureAxis.for_spec(spec).axis_digest
                for spec in result.observer_vocabulary.specs
            }
        )
    )
    assert len(axes) == 1
    rows: list[PanelFeatureEvidencePanel] = []
    for phase, panels in (
        (PanelFeatureEvidencePhase.SUPPORT, support_pngs),
        (PanelFeatureEvidencePhase.QUERY, query_pngs),
    ):
        for index, panel in enumerate(panels):
            context = build_panel_only_observation_context(
                panel,
                model=MODEL,
                reasoning_effort=EFFORT,
                expected_launcher_digest=LAUNCHER_DIGEST,
                **NO_TOOLS_KWARGS,
            )
            axis_artifacts = []
            for axis in axes:
                view = FeatureAxisObservationView.build(
                    context.to_observation_context(), axis
                )
                axis_artifacts.append(
                    observe_typed_panel_axis(
                        panel,
                        axis=axis,
                        panel_only_context=context,
                        model=MODEL,
                        reasoning_effort=EFFORT,
                        expected_launcher_digest=LAUNCHER_DIGEST,
                        **NO_TOOLS_KWARGS,
                        transport=_transport(
                            _axis_payload(view), expected_images=(panel,)
                        ),
                    )
                )
            artifacts = tuple(
                sorted(
                    axis_artifacts,
                    key=lambda item: item.observation.axis.axis_digest,
                )
            )
            rows.append(
                PanelFeatureEvidencePanel.derive_from_full_artifacts(
                    phase=phase,
                    phase_index=index,
                    panel_id=f"panel-{len(rows):03d}",
                    panel_png=panel,
                    owner_artifact=None,
                    axis_artifacts=artifacts,
                )
            )
    return PanelFeatureEvidenceBundle.create(
        proposer_artifact=proposer,
        proposer_result=result,
        observer_axes=axes,
        panels=rows,
    )


def _local_spec(kind: GestaltKind) -> PanelFeatureSpec:
    return PanelFeatureSpec(
        FeatureFamily.GESTALT_RESEMBLANCE,
        SubjectScope.ONE_COHERENT_FIGURE,
        ReferenceFrame.NONE,
        GestaltResemblanceParameters(kind),
    )


def _local_payload() -> dict[str, object]:
    result: dict[str, object] = {}
    kinds = (GestaltKind.BIRD_LIKE, GestaltKind.ANIMAL_LIKE)
    for block_index, block in enumerate(PANEL_FEATURE_BLOCKS):
        for slot in range(PANEL_FEATURE_SLOTS_PER_DIRECTION):
            result[f"{block}_candidate_{slot}"] = _proposer_row(
                _local_spec(kinds[block_index]),
                native_block=block,
                narration_suffix=f"{block}-{slot}",
            )
    return result


def _owner_local_bundle() -> PanelFeatureEvidenceBundle:
    panels = tuple(_png(900 + index) for index in range(12))
    proposer, result = _proposer(panels, _local_payload())
    assert result.observer_vocabulary is not None
    axes_by_digest = {
        FeatureAxis.for_spec(spec).axis_digest: FeatureAxis.for_spec(spec)
        for spec in result.observer_vocabulary.specs
    }
    assert len(axes_by_digest) == 1
    axis = axes_by_digest[sorted(axes_by_digest)[0]]
    rows: list[PanelFeatureEvidencePanel] = []
    for index, panel in enumerate(panels):
        owner = _owner_artifact(panel)
        view = FeatureAxisObservationView.build(owner.to_owner_inventory(), axis)
        artifact = observe_typed_panel_axis(
            panel,
            axis=axis,
            owner_artifact=owner,
            model=MODEL,
            reasoning_effort=EFFORT,
            expected_launcher_digest=LAUNCHER_DIGEST,
            **NO_TOOLS_KWARGS,
            transport=_transport(_axis_payload(view), expected_images=(panel,)),
        )
        rows.append(
            PanelFeatureEvidencePanel.derive_from_full_artifacts(
                phase=PanelFeatureEvidencePhase.SUPPORT,
                phase_index=index,
                panel_id=f"local-panel-{index:03d}",
                panel_png=panel,
                owner_artifact=owner,
                axis_artifacts=(artifact,),
            )
        )
    return PanelFeatureEvidenceBundle.create(
        proposer_artifact=proposer,
        proposer_result=result,
        observer_axes=(axis,),
        panels=rows,
    )


def _batched_bundle() -> tuple[PanelFeatureEvidenceBundle, list[dict[str, object]]]:
    panels = tuple(_png(1100 + index) for index in range(12))
    proposer, result = _proposer(panels, _proposer_payload())
    axes = _batch_observer.complete_whole_panel_feature_axes()
    rows: list[PanelFeatureEvidencePanel] = []
    calls: list[dict[str, object]] = []
    for index, panel in enumerate(panels):
        context = build_panel_only_observation_context(
            panel,
            model=MODEL,
            reasoning_effort=EFFORT,
            expected_launcher_digest=LAUNCHER_DIGEST,
            **NO_TOOLS_KWARGS,
        )
        request = BatchedFeatureAxisRequest.build(context, axes)
        artifact = observe_typed_panel_axes_batched(
            panel,
            axes=axes,
            panel_only_context=context,
            model=MODEL,
            reasoning_effort=EFFORT,
            expected_launcher_digest=LAUNCHER_DIGEST,
            **NO_TOOLS_KWARGS,
            transport=_batched_transport(_batched_payload(request), panel, calls),
        )
        rows.append(
            PanelFeatureEvidencePanel.derive_from_batched_artifact(
                phase=PanelFeatureEvidencePhase.SUPPORT,
                phase_index=index,
                panel_id=f"batched-panel-{index:03d}",
                panel_png=panel,
                batched_axis_artifact=artifact,
            )
        )
    return (
        PanelFeatureEvidenceBundle.create(
            proposer_artifact=proposer,
            proposer_result=result,
            observer_axes=axes,
            panels=rows,
        ),
        calls,
    )


@pytest.fixture(scope="module")
def whole_bundle() -> PanelFeatureEvidenceBundle:
    return _whole_panel_bundle()


@pytest.fixture(scope="module")
def owner_bundle() -> PanelFeatureEvidenceBundle:
    return _owner_local_bundle()


@pytest.fixture(scope="module")
def batched_bundle():
    # The batch adapter's own tests exercise the full live catalog. Keep this
    # twelve-panel custody test bounded while retaining multiple logical axes
    # behind each one-call artifact.
    test_axes = _batch_observer.complete_whole_panel_feature_axes()[:2]
    patch = pytest.MonkeyPatch()
    patch.setattr(
        _batch_observer,
        "complete_whole_panel_feature_axes",
        lambda: test_axes,
    )
    try:
        yield _batched_bundle()
    finally:
        patch.undo()


def test_full_receipts_exact_pixels_phase_tags_and_zero_call_cold_replay(
    whole_bundle: PanelFeatureEvidenceBundle,
) -> None:
    assert len(whole_bundle.panels) == 14
    assert whole_bundle.to_data()["support_panel_count"] == 12
    assert whole_bundle.to_data()["query_panel_count"] == 2
    assert whole_bundle.to_data()["query_phase_complete"] is True
    assert whole_bundle.live_model_call_count == 15
    assert whole_bundle.to_data()["owner_model_call_count"] == 0
    assert whole_bundle.to_data()["individual_axis_model_call_count"] == 14
    assert whole_bundle.to_data()["batched_axis_model_call_count"] == 0
    assert whole_bundle.to_data()["axis_model_call_count"] == 14
    assert whole_bundle.to_data()["observer_axes"] == [
        item.to_data() for item in whole_bundle.observer_axes
    ]
    assert len(whole_bundle.physical_receipt_digests) == 15
    assert len(set(whole_bundle.physical_receipt_digests)) == 15
    assert PanelFeatureEvidenceBundle.from_data(whole_bundle.to_data()) == whole_bundle
    assert (
        cold_replay_panel_feature_evidence_bundle(
            whole_bundle,
            expected_bundle_address=whole_bundle.bundle_address,
        )
        == whole_bundle
    )
    for panel in whole_bundle.panels:
        assert panel.observation_set == PanelFeatureObservationSet(
            panel.panel_png_digest,
            panel.axis_artifacts[0].observer_contract_digest,
            panel.axis_artifacts[0].measurement_protocol_digest,
            tuple(item.observation for item in panel.axis_artifacts),
        )
        observer_envelope = json.dumps(
            [item.to_data() for item in panel.axis_artifacts], sort_keys=True
        )
        assert panel.panel_id not in observer_envelope
        assert "selected_predicate" not in observer_envelope


def test_owner_calls_are_retained_once_and_mechanically_counted(
    owner_bundle: PanelFeatureEvidenceBundle,
) -> None:
    assert owner_bundle.to_data()["query_panel_count"] == 0
    assert owner_bundle.to_data()["owner_model_call_count"] == 12
    assert owner_bundle.to_data()["individual_axis_model_call_count"] == 12
    assert owner_bundle.to_data()["batched_axis_model_call_count"] == 0
    assert owner_bundle.to_data()["axis_model_call_count"] == 12
    assert owner_bundle.live_model_call_count == 25
    assert all(item.owner_artifact is not None for item in owner_bundle.panels)
    assert PanelFeatureEvidenceBundle.from_data(owner_bundle.to_data()) == owner_bundle


def test_batched_receipts_retain_all_cells_but_count_one_call_per_panel(
    batched_bundle: tuple[
        PanelFeatureEvidenceBundle, list[dict[str, object]]
    ],
) -> None:
    bundle, calls = batched_bundle
    axis_count = len(bundle.observer_axes)
    assert axis_count > 1
    assert len(calls) == 12
    assert bundle.to_data()["individual_axis_model_call_count"] == 0
    assert bundle.to_data()["batched_axis_model_call_count"] == 12
    assert bundle.to_data()["axis_model_call_count"] == 12
    assert bundle.live_model_call_count == 13
    assert len(bundle.physical_receipt_digests) == 13
    assert all(not panel.axis_artifacts for panel in bundle.panels)
    assert all(panel.batched_axis_artifact is not None for panel in bundle.panels)
    assert all(
        len(panel.observation_set.axis_observations) == axis_count
        for panel in bundle.panels
    )
    assert PanelFeatureEvidenceBundle.from_data(bundle.to_data()) == bundle
    assert (
        cold_replay_panel_feature_evidence_bundle(
            bundle,
            expected_bundle_address=bundle.bundle_address,
        )
        == bundle
    )
    assert len(calls) == 12  # Cold replay invokes no transport.
    for panel in bundle.panels:
        artifact = panel.batched_axis_artifact
        assert artifact is not None
        assert panel.observation_set == artifact.observation_set
        observer_envelope = json.dumps(artifact.to_data(), sort_keys=True)
        assert panel.panel_id not in observer_envelope
        assert "selected_predicate" not in observer_envelope


def test_batched_and_individual_paths_are_exclusive_and_batch_tampering_fails(
    whole_bundle: PanelFeatureEvidenceBundle,
    batched_bundle: tuple[
        PanelFeatureEvidenceBundle, list[dict[str, object]]
    ],
) -> None:
    batch, _calls = batched_bundle
    individual = whole_bundle.panels[0]
    batched = batch.panels[0]
    artifact = batched.batched_axis_artifact
    assert artifact is not None
    with pytest.raises(PanelFeatureEvidenceBundleError, match="exactly one"):
        PanelFeatureEvidencePanel.create(
            phase=individual.phase,
            phase_index=individual.phase_index,
            panel_id=individual.panel_id,
            panel_png=individual.panel_png,
            owner_artifact=None,
            axis_artifacts=individual.axis_artifacts,
            batched_axis_artifact=artifact,
            observation_set=individual.observation_set,
        )
    with pytest.raises(PanelFeatureEvidenceBundleError, match="exactly one"):
        PanelFeatureEvidencePanel.create(
            phase=individual.phase,
            phase_index=individual.phase_index,
            panel_id=individual.panel_id,
            panel_png=individual.panel_png,
            owner_artifact=None,
            axis_artifacts=(),
            observation_set=individual.observation_set,
        )

    tampered = deepcopy(batch.to_data())
    tampered["panels"][0]["batched_axis_artifact"]["codex_receipt"][
        "prompt_digest"
    ] = "0" * 64
    with pytest.raises(ValueError):
        PanelFeatureEvidenceBundle.from_data(tampered)


def test_missing_extra_duplicated_and_mismatched_artifacts_fail_closed(
    whole_bundle: PanelFeatureEvidenceBundle,
    owner_bundle: PanelFeatureEvidenceBundle,
) -> None:
    first = whole_bundle.panels[0]
    foreign_axis = owner_bundle.observer_axes[0]
    assert foreign_axis not in whole_bundle.observer_axes
    expanded_catalog = tuple(
        sorted(
            (*whole_bundle.observer_axes, foreign_axis),
            key=lambda item: item.axis_digest,
        )
    )
    with pytest.raises(PanelFeatureEvidenceBundleError, match="missing, extra"):
        PanelFeatureEvidenceBundle.create(
            proposer_artifact=whole_bundle.proposer_artifact,
            proposer_result=whole_bundle.proposer_result,
            observer_axes=expanded_catalog,
            panels=whole_bundle.panels,
        )

    foreign_artifact = owner_bundle.panels[0].axis_artifacts[0]
    row_with_extra_axis = PanelFeatureEvidencePanel.create(
        phase=first.phase,
        phase_index=first.phase_index,
        panel_id=first.panel_id,
        panel_png=first.panel_png,
        owner_artifact=None,
        axis_artifacts=tuple(
            sorted(
                (*first.axis_artifacts, foreign_artifact),
                key=lambda item: item.observation.axis.axis_digest,
            )
        ),
        observation_set=first.observation_set,
    )
    with pytest.raises(PanelFeatureEvidenceBundleError, match="missing, extra"):
        PanelFeatureEvidenceBundle.create(
            proposer_artifact=whole_bundle.proposer_artifact,
            proposer_result=whole_bundle.proposer_result,
            observer_axes=whole_bundle.observer_axes,
            panels=(row_with_extra_axis, *whole_bundle.panels[1:]),
        )

    with pytest.raises(PanelFeatureEvidenceBundleError, match="unique"):
        PanelFeatureEvidencePanel.create(
            phase=first.phase,
            phase_index=first.phase_index,
            panel_id=first.panel_id,
            panel_png=first.panel_png,
            owner_artifact=None,
            axis_artifacts=(first.axis_artifacts[0], first.axis_artifacts[0]),
            observation_set=first.observation_set,
        )

    extra_owner = owner_bundle.panels[0].owner_artifact
    assert extra_owner is not None
    row_with_extra_owner = PanelFeatureEvidencePanel.create(
        phase=first.phase,
        phase_index=first.phase_index,
        panel_id=first.panel_id,
        panel_png=first.panel_png,
        owner_artifact=extra_owner,
        axis_artifacts=first.axis_artifacts,
        observation_set=first.observation_set,
    )
    with pytest.raises(PanelFeatureEvidenceBundleError, match="missing when used or extra"):
        PanelFeatureEvidenceBundle.create(
            proposer_artifact=whole_bundle.proposer_artifact,
            proposer_result=whole_bundle.proposer_result,
            observer_axes=whole_bundle.observer_axes,
            panels=(row_with_extra_owner, *whole_bundle.panels[1:]),
        )

    local = owner_bundle.panels[0]
    missing_owner = PanelFeatureEvidencePanel.create(
        phase=local.phase,
        phase_index=local.phase_index,
        panel_id=local.panel_id,
        panel_png=local.panel_png,
        owner_artifact=None,
        axis_artifacts=local.axis_artifacts,
        observation_set=local.observation_set,
    )
    with pytest.raises(PanelFeatureEvidenceBundleError, match="missing when used or extra"):
        PanelFeatureEvidenceBundle.create(
            proposer_artifact=owner_bundle.proposer_artifact,
            proposer_result=owner_bundle.proposer_result,
            observer_axes=owner_bundle.observer_axes,
            panels=(missing_owner, *owner_bundle.panels[1:]),
        )

    mismatched_observation = PanelFeatureEvidencePanel.create(
        phase=first.phase,
        phase_index=first.phase_index,
        panel_id=first.panel_id,
        panel_png=first.panel_png,
        owner_artifact=None,
        axis_artifacts=first.axis_artifacts,
        observation_set=whole_bundle.panels[1].observation_set,
    )
    with pytest.raises(PanelFeatureEvidenceBundleError, match="observation set"):
        PanelFeatureEvidenceBundle.create(
            proposer_artifact=whole_bundle.proposer_artifact,
            proposer_result=whole_bundle.proposer_result,
            observer_axes=whole_bundle.observer_axes,
            panels=(mismatched_observation, *whole_bundle.panels[1:]),
        )


def test_opaque_receipt_and_content_address_tampering_are_rejected(
    whole_bundle: PanelFeatureEvidenceBundle,
) -> None:
    tampered = deepcopy(whole_bundle.to_data())
    tampered["physical_receipt_digests"][0] = "f" * 64
    with pytest.raises(PanelFeatureEvidenceBundleError):
        PanelFeatureEvidenceBundle.from_data(tampered)

    tampered = deepcopy(whole_bundle.to_data())
    tampered["proposer_result"]["receipt_digest"] = "e" * 64
    with pytest.raises(PanelFeatureEvidenceBundleError):
        PanelFeatureEvidenceBundle.from_data(tampered)

    with pytest.raises(PanelFeatureEvidenceBundleError, match="external"):
        cold_replay_panel_feature_evidence_bundle(
            whole_bundle,
            expected_bundle_address="sha256:" + "d" * 64,
        )
