from __future__ import annotations

import copy
from pathlib import Path

import pytest

from bongard import panel_action_count_skeleton_graph_dev_command as development
from bongard.panel_action_count_skeleton_graph_passed_fit_protocol import (
    PASSED_FIT_ALGORITHM_DIGEST,
    REQUIRED_HEADS,
    SkeletonGraphPassedFitProtocol,
    SkeletonGraphPassedFitProtocolError,
    resolve_skeleton_graph_passed_fit_protocol,
    verify_skeleton_graph_passed_fit_protocol,
)
from bongard.panel_action_count_skeleton_graph_calibration_prereg import (
    PASSED_FIT_PROTOCOL_SCHEMA,
    passed_fit_slot,
    resolve_passed_fit_slot,
)


ROOT = Path(__file__).resolve().parents[2]
LIVE = (
    ROOT
    / "downloads"
    / "ShapeBongard_V2_full"
    / "panel_action_count_skeleton_graph_dev_20260810_v2"
)


def _paths() -> dict[str, Path]:
    return {
        "development_precommit_path": LIVE / "precommit.json",
        "development_result_path": LIVE / "result.json",
        "development_replay_path": LIVE / "replay.json",
        "model_path": LIVE / "model.pkl",
        "feature_artifact_path": LIVE / "features.json",
        "prediction_artifact_path": LIVE / "predictions.json",
    }


@pytest.fixture(scope="module")
def protocol() -> SkeletonGraphPassedFitProtocol:
    if not all(path.is_file() for path in _paths().values()):
        pytest.skip("authenticated local skeleton-graph v2 chain is unavailable")
    value = resolve_skeleton_graph_passed_fit_protocol(**_paths())
    assert type(value) is SkeletonGraphPassedFitProtocol
    return value


def test_real_six_file_chain_resolves_exact_pass(protocol: SkeletonGraphPassedFitProtocol) -> None:
    assert protocol.record_digest == (
        "sha256:765b77632ad35012996be71e6effb2f56a1dbfc50080acffc6239bba84ceb15a"
    )
    assert protocol.passed_fit_algorithm_digest == PASSED_FIT_ALGORITHM_DIGEST
    assert protocol.required_heads == REQUIRED_HEADS
    assert protocol.model_size_bytes == 97_911_851
    assert protocol.replay_model_refit_calls == 0
    assert protocol.replay_model_inference_panel_count == 1_392
    assert protocol.replay_pixel_reextract_group_count == 12_535
    assert protocol.model_file_sha256 == (
        "sha256:25d1c21a117fe2bb2c68f9328351ef86f8b403019afafa182ae6b7d73aed2c52"
    )
    assert protocol.development_replay_record_digest == (
        "sha256:820d65653e26e529465097c62cd0f509e699ebfbb4ba96823abcf3289ef3a92a"
    )
    data = protocol.to_data()
    assert data["calibration_pixel_authorized"] is False
    assert data["support_query_inference_authorized"] is False
    assert data["target_pixel_authorized"] is False
    assert data["benchmark_sealable"] is False
    assert data["eligible_for_calibration_execution_precommit"] is True
    resolved = resolve_passed_fit_slot(
        passed_fit_slot(),
        outcome_schema=PASSED_FIT_PROTOCOL_SCHEMA,
        addresses={
            "passed_fit_authority_source_sha256": protocol.passed_fit_authority_source_sha256,
            "passed_fit_algorithm_digest": protocol.passed_fit_algorithm_digest,
            "passed_fit_record_digest": protocol.record_digest,
        },
    )
    assert resolved["passed_fit_record_digest"] == protocol.record_digest


def test_real_chain_fresh_verification(protocol: SkeletonGraphPassedFitProtocol) -> None:
    assert (
        verify_skeleton_graph_passed_fit_protocol(
            protocol,
            **_paths(),
            expected_record_digest=protocol.record_digest,
        )
        is protocol
    )


def test_canonical_round_trip_and_policy_tamper_rejection(
    protocol: SkeletonGraphPassedFitProtocol,
) -> None:
    assert SkeletonGraphPassedFitProtocol.from_data(protocol.to_data()).to_data() == protocol.to_data()
    tampered = copy.deepcopy(protocol.to_data())
    tampered["benchmark_sealable"] = True
    with pytest.raises(SkeletonGraphPassedFitProtocolError):
        SkeletonGraphPassedFitProtocol.from_data(tampered)


def test_factory_only_and_wrong_expected_digest_fail(
    protocol: SkeletonGraphPassedFitProtocol,
) -> None:
    with pytest.raises(SkeletonGraphPassedFitProtocolError):
        SkeletonGraphPassedFitProtocol()
    with pytest.raises(SkeletonGraphPassedFitProtocolError):
        verify_skeleton_graph_passed_fit_protocol(
            protocol,
            **_paths(),
            expected_record_digest="sha256:" + "0" * 64,
        )


def test_algorithm_contract_binds_exact_domains() -> None:
    assert REQUIRED_HEADS == ("direct_pair", "catalog_three_class")
    assert len(development.OBSERVED_TRAIN_PAIR_CLASS_ORDER) == 33
    assert len(development.VALID_PAIR_CLASS_ORDER) == 54
    assert development.CATALOG_CLASS_ORDER == (-1, 0, 1)
    assert PASSED_FIT_ALGORITHM_DIGEST.startswith("sha256:")
