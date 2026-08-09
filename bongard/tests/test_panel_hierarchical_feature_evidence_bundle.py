"""Focused exact-row, bundle-custody, and zero-call replay tests."""

from __future__ import annotations

from copy import deepcopy
import inspect
import json

import pytest

from bongard.panel_hierarchical_feature_evidence_bundle import (
    HierarchicalFeatureEvidencePhase,
    HierarchicalPanelFeatureEvidenceBundle,
    HierarchicalPanelFeatureEvidenceError,
    HierarchicalPanelFeatureEvidenceRow,
    cold_replay_hierarchical_panel_feature_evidence_bundle,
    verified_hierarchical_observation_sets,
)
from bongard.panel_hierarchical_visual_adapter import observe_hierarchical_panel
from bongard.tests.test_panel_feature_evidence_bundle import _proposer
from bongard.tests.test_panel_feature_proposer import _payload as _proposer_payload
from bongard.tests.test_panel_hierarchical_visual_adapter import (
    _payload as _hierarchical_payload,
    _request,
    _square_spans,
    _transport,
)
from bongard.tests.test_prototype_scene_observer import (
    EFFORT,
    LAUNCHER_DIGEST,
    MODEL,
    NO_TOOLS_KWARGS,
    _png,
)


@pytest.fixture(scope="module")
def hierarchical_bundle():
    support_pngs = tuple(_png(3000 + index) for index in range(12))
    query_pngs = tuple(_png(3100 + index) for index in range(2))
    proposer, result = _proposer(support_pngs, _proposer_payload())
    calls: list[dict[str, object]] = []
    rows: list[HierarchicalPanelFeatureEvidenceRow] = []
    for phase, panels in (
        (HierarchicalFeatureEvidencePhase.SUPPORT, support_pngs),
        (HierarchicalFeatureEvidencePhase.QUERY, query_pngs),
    ):
        for index, panel in enumerate(panels):
            request = _request(panel)
            payload = _hierarchical_payload(request, _square_spans())
            artifact = observe_hierarchical_panel(
                panel,
                request=request,
                model=MODEL,
                reasoning_effort=EFFORT,
                expected_launcher_digest=LAUNCHER_DIGEST,
                **NO_TOOLS_KWARGS,
                transport=_transport(payload, panel, calls),
            )
            rows.append(
                HierarchicalPanelFeatureEvidenceRow.create(
                    phase=phase,
                    phase_index=index,
                    panel_id=f"hierarchical-panel-{len(rows):03d}",
                    panel_png=panel,
                    artifact=artifact,
                )
            )
    bundle = HierarchicalPanelFeatureEvidenceBundle.create(
        proposer_artifact=proposer,
        proposer_result=result,
        panels=rows,
    )
    return bundle, calls


def test_exact_12_plus_2_rows_shared_runtime_catalog_and_no_bare_input(
    hierarchical_bundle,
) -> None:
    bundle, calls = hierarchical_bundle
    assert len(calls) == 14
    assert len(bundle.panels) == 14
    assert len(
        bundle.panels_for_phase(HierarchicalFeatureEvidencePhase.SUPPORT)
    ) == 12
    assert len(bundle.panels_for_phase(HierarchicalFeatureEvidencePhase.QUERY)) == 2
    assert bundle.live_model_call_count == 15
    assert len(bundle.physical_receipt_digests) == 15
    assert len(set(bundle.physical_receipt_digests)) == 15
    data = bundle.to_data()
    assert data["support_panel_count"] == 12
    assert data["query_panel_count"] == 2
    assert data["query_phase_complete"] is True
    assert data["bare_observation_sets_accepted"] is False
    assert len(data["observer_axes"]) == 9
    assert all(item.artifact.runtime == bundle.observer_runtime for item in bundle.panels)
    assert all(
        item.artifact.hierarchical_contract_digest
        == bundle.panels[0].artifact.hierarchical_contract_digest
        for item in bundle.panels
    )
    assert all(
        item.artifact.observer_contract_digest == data["observer_contract_digest"]
        for item in bundle.panels
    )

    row_parameters = set(
        inspect.signature(HierarchicalPanelFeatureEvidenceRow.create).parameters
    )
    bundle_parameters = set(
        inspect.signature(HierarchicalPanelFeatureEvidenceBundle.create).parameters
    )
    assert "observation_set" not in row_parameters
    assert "observation_set" not in bundle_parameters
    assert all("observation_set" not in row.to_data() for row in bundle.panels)
    for row in bundle.panels:
        observer_envelope = json.dumps(row.artifact.to_data(), sort_keys=True)
        assert row.panel_id not in observer_envelope
        assert row.phase.value not in row.artifact.request.model_data()


def test_zero_or_exactly_two_query_rows_are_the_only_bundle_shapes(
    hierarchical_bundle,
) -> None:
    bundle, _calls = hierarchical_bundle
    support = bundle.panels_for_phase(HierarchicalFeatureEvidencePhase.SUPPORT)
    support_only = HierarchicalPanelFeatureEvidenceBundle.create(
        proposer_artifact=bundle.proposer_artifact,
        proposer_result=bundle.proposer_result,
        panels=support,
    )
    assert support_only.to_data()["query_panel_count"] == 0
    assert support_only.to_data()["query_phase_complete"] is False
    first_query = bundle.panels_for_phase(HierarchicalFeatureEvidencePhase.QUERY)[0]
    with pytest.raises(
        HierarchicalPanelFeatureEvidenceError, match="zero or two"
    ):
        HierarchicalPanelFeatureEvidenceBundle.create(
            proposer_artifact=bundle.proposer_artifact,
            proposer_result=bundle.proposer_result,
            panels=(*support, first_query),
        )


def test_round_trip_cold_replay_and_verified_reconstruction_call_no_transport(
    hierarchical_bundle,
) -> None:
    bundle, calls = hierarchical_bundle
    before = len(calls)
    assert HierarchicalPanelFeatureEvidenceBundle.from_data(bundle.to_data()) == bundle
    replayed = cold_replay_hierarchical_panel_feature_evidence_bundle(
        bundle,
        expected_bundle_address=bundle.bundle_address,
    )
    assert replayed == bundle
    support = verified_hierarchical_observation_sets(
        bundle,
        phase=HierarchicalFeatureEvidencePhase.SUPPORT,
        expected_bundle_address=bundle.bundle_address,
    )
    query = verified_hierarchical_observation_sets(
        bundle,
        phase=HierarchicalFeatureEvidencePhase.QUERY,
        expected_bundle_address=bundle.bundle_address,
    )
    assert support == tuple(
        row.artifact.observation_set
        for row in bundle.panels_for_phase(HierarchicalFeatureEvidencePhase.SUPPORT)
    )
    assert query == tuple(
        row.artifact.observation_set
        for row in bundle.panels_for_phase(HierarchicalFeatureEvidencePhase.QUERY)
    )
    assert len(calls) == before


def test_wrong_pixels_metadata_artifacts_and_external_address_fail_closed(
    hierarchical_bundle,
) -> None:
    bundle, _calls = hierarchical_bundle
    first = bundle.panels[0]
    with pytest.raises(HierarchicalPanelFeatureEvidenceError, match="verification"):
        HierarchicalPanelFeatureEvidenceRow.create(
            phase=first.phase,
            phase_index=first.phase_index,
            panel_id=first.panel_id,
            panel_png=_png(3999),
            artifact=first.artifact,
        )

    tampered = deepcopy(bundle.to_data())
    tampered["panels"][0]["hierarchical_artifact"]["codex_receipt"][
        "prompt_digest"
    ] = "0" * 64
    with pytest.raises(ValueError):
        HierarchicalPanelFeatureEvidenceBundle.from_data(tampered)

    tampered = deepcopy(bundle.to_data())
    tampered["shared_model_catalog_digest"] = "f" * 64
    with pytest.raises(HierarchicalPanelFeatureEvidenceError):
        HierarchicalPanelFeatureEvidenceBundle.from_data(tampered)

    tampered = deepcopy(bundle.to_data())
    tampered["panels"][0]["phase_index"] = 1
    with pytest.raises(HierarchicalPanelFeatureEvidenceError):
        HierarchicalPanelFeatureEvidenceBundle.from_data(tampered)

    with pytest.raises(HierarchicalPanelFeatureEvidenceError, match="external"):
        cold_replay_hierarchical_panel_feature_evidence_bundle(
            bundle,
            expected_bundle_address="sha256:" + "d" * 64,
        )
