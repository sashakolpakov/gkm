from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image
import pytest

from bongard.artifacts import canonical_digest
from bongard.blind_soft_transport import (
    BLIND_SOFT_DECODER_ID,
    BLIND_SOFT_PROMPT_TEMPLATE_ID,
    BlindSoftVerifierContext,
    blind_soft_decoder_digest,
    blind_soft_prompt_template_digest,
    score_blind_soft_panel,
)
from bongard.corpus import SplitIndex
from bongard.semantic_calibration import (
    CalibrationLabelJoinReceipt,
    CalibrationPanelSelection,
    SemanticCalibrationArtifact,
    SemanticCalibrationError,
    SemanticCalibrationMeasurement,
    SemanticCalibrationPlan,
    fit_semantic_calibration,
    join_calibration_label,
)
from bongard.soft_predicates import (
    SoftPredicateIntegrityError,
    SoftScorerFamily,
    SoftScorerProtocol,
)
from bongard.transport import (
    CODEX_ISOLATION_POLICY,
    CODEX_RECEIPT_SCHEMA,
    NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
    CodexReceipt,
    CodexStructuredResult,
    named_image_set_digest,
)
from bongard.typed_visual_proposal import TypedSoftClaim, TypedSoftCue


MODEL = "gpt-test"
EFFORT = "medium"
SCORES = (0.0, 0.0, 0.5, 0.5, 1.0, 1.0)
LABELS = (False, False, False, True, True, True)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _address(value: str) -> str:
    return "sha256:" + _digest(value)


@pytest.fixture
def protocol() -> SoftScorerProtocol:
    return SoftScorerProtocol(
        family_id="open-semantic-positive-cues",
        version="1",
        proposer_grammar_id="positive-cue-rubric-v1",
        proposer_grammar_digest=_digest("proposer grammar"),
        proposer_model_id=MODEL,
        proposer_reasoning_effort=EFFORT,
        proposer_prompt_id="typed-visual-proposer-v1",
        proposer_prompt_digest=_digest("proposer prompt"),
        scorer_model_id=MODEL,
        scorer_reasoning_effort=EFFORT,
        scorer_prompt_template_id=BLIND_SOFT_PROMPT_TEMPLATE_ID,
        scorer_prompt_template_digest=blind_soft_prompt_template_digest(),
        scorer_decoder_id=BLIND_SOFT_DECODER_ID,
        scorer_decoder_digest=blind_soft_decoder_digest(),
        ordinal_map=(
            ("supported", 1.0),
            ("ambiguous", 0.5),
            ("unsupported", 0.0),
        ),
        aggregation="min",
        witness_extractor_id="joint-panel-witnesses-v1",
        witness_extractor_digest=_digest("witness extractor"),
        support_gate_id="exact-aligned-6-plus-6-v1",
        support_gate_digest=_digest("support gate"),
        score_bin_edges=(0.0, 0.25, 0.75, 1.0),
        affirmative_boundary=0.7,
        confidence_level=0.8,
        minimum_clusters_per_bin=2,
    )


def _split(task_ids: Sequence[str], *, test_ids: Sequence[str] = ()) -> SplitIndex:
    return SplitIndex(
        groups=(
            ("test", tuple(sorted(test_ids))),
            ("train", tuple(sorted(task_ids))),
            ("val", ()),
        ),
        source_digest=_address("official split fixture"),
    )


def _write_panel(tmp_path: Path, index: int) -> Path:
    path = tmp_path / f"opaque-development-{index:02d}.png"
    image = Image.new("L", (18, 18), color=255)
    pixels = image.load()
    assert pixels is not None
    for coordinate in range(index + 3):
        pixels[coordinate % 12, coordinate // 12] = 0
    image.save(path, format="PNG")
    return path


def _selection(path: Path, index: int) -> CalibrationPanelSelection:
    return CalibrationPanelSelection(
        observation_id=f"development-{index:02d}",
        task_id=f"train-task-{index:02d}",
        panel_id=f"dev-panel-{index:02d}",
        panel_digest=hashlib.sha256(path.read_bytes()).hexdigest(),
        split="train",
        dependence_cluster_id=f"cluster-{index:02d}",
    )


def _receipt(
    prompt: str,
    paths: Sequence[str],
    names: Sequence[str],
    schema: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    thread_index: int,
) -> CodexReceipt:
    identities = [
        {
            "name": name,
            "byte_count": len(Path(path).read_bytes()),
            "content_digest": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        }
        for path, name in zip(paths, names, strict=True)
    ]
    prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    schema_digest = canonical_digest(dict(schema))
    view_digest = canonical_digest(identities)
    set_digest = named_image_set_digest(paths, names)
    envelope = {
        "schema": NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
        "task": prompt,
        "ordered_image_identities": identities,
        "image_view_digest": view_digest,
        "image_set_digest": set_digest,
        "prompt_digest": prompt_digest,
        "output_schema_digest": schema_digest,
    }
    body: dict[str, Any] = {
        "schema": CODEX_RECEIPT_SCHEMA,
        "source": "codex-cli",
        "requested_model": MODEL,
        "reported_model": "",
        "model_identity_evidence": "explicit-cli-model-flag;jsonl-omits-model",
        "requested_reasoning_effort": EFFORT,
        "input_tokens": 10,
        "cached_input_tokens": 0,
        "output_tokens": 2,
        "reasoning_output_tokens": 1,
        "thread_id": f"00000000-0000-4000-8000-{thread_index + 1:012d}",
        "codex_cli_version": "codex-cli test",
        "codex_launcher_digest": "b" * 64,
        "cloud_config_bundle_cache_binding": "absent",
        "task_digest": prompt_digest,
        "current_source_digest": "",
        "current_log_digest": "",
        "prompt_digest": prompt_digest,
        "input_digest_schema": NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
        "input_digest": canonical_digest(envelope),
        "output_schema_digest": schema_digest,
        "panel_view_digest": view_digest,
        "panel_set_digest": set_digest,
        "structured_output_digest": canonical_digest(dict(payload)),
        "proposed_source_digest": "",
        "proposed_log_digest": "",
        "event_stream_digest": _digest(f"event stream {thread_index}"),
        "event_types": [
            "thread.started",
            "turn.started",
            "item.completed",
            "turn.completed",
        ],
        "item_types": ["agent_message"],
        "isolation_policy": CODEX_ISOLATION_POLICY,
        "outcome": "success",
    }
    body["receipt_digest"] = canonical_digest(body)
    return CodexReceipt(
        **{
            **body,
            "event_types": tuple(body["event_types"]),
            "item_types": tuple(body["item_types"]),
        }
    )


def _score_payload(score: float) -> dict[str, object]:
    judgment = {0.0: "unsupported", 0.5: "ambiguous", 1.0: "supported"}[score]
    return {
        "cue_judgments": [
            {
                "cue_id": "cue-00",
                "judgment": judgment,
                "witness_ids": [] if score == 0.0 else ["component:0"],
            }
        ]
    }


def _score_artifact(
    protocol: SoftScorerProtocol,
    path: Path,
    selection: CalibrationPanelSelection,
    score: float,
    index: int,
    *,
    proposer_call_id: str | None = None,
    scorer_call_id: str | None = None,
    fail_transport: bool = False,
):
    variant = (
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
    )[index]
    claim = TypedSoftClaim(
        atom_id="atom-00",
        positive_description=f"a bird-like organization variant {variant}",
        cues=(TypedSoftCue("cue-00", "one central body-like component"),),
        aggregation="min",
        scorer_protocol_digest=protocol.digest(),
    )
    context = BlindSoftVerifierContext(
        task_id=selection.task_id,
        panel_id=selection.panel_id,
        proposer_call_id=proposer_call_id or f"proposer-call-{index:02d}",
        proposer_receipt_digest=_digest(f"proposer receipt {index}"),
        scorer_call_id=scorer_call_id or f"scorer-call-{index:02d}",
        pre_observation_commitment_digest=_digest(
            f"frozen proposal and policy commitment {index}"
        ),
    )

    def transport(prompt, paths, names, schema, **kwargs):
        del kwargs
        if fail_transport:
            raise RuntimeError("scorer unavailable")
        payload = _score_payload(score)
        return CodexStructuredResult(
            payload=payload,
            receipt=_receipt(
                prompt,
                paths,
                names,
                schema,
                payload,
                thread_index=index,
            ),
        )

    return score_blind_soft_panel(
        path,
        claim,
        protocol=protocol,
        witness_packet_digest=_digest(f"witness packet {index}"),
        witness_summaries={
            "component:0": "largest connected component near the panel center"
        },
        context=context,
        transport=transport,
    )


def _plan_and_inputs(tmp_path: Path, protocol: SoftScorerProtocol, count: int = 6):
    paths = tuple(_write_panel(tmp_path, index) for index in range(count))
    selections = tuple(_selection(path, index) for index, path in enumerate(paths))
    split = _split(tuple(item.task_id for item in selections))
    plan = SemanticCalibrationPlan.create(
        protocol,
        split,
        selections,
        corpus_manifest_digest=_address("complete corpus manifest"),
        development_manifest_digest=_address("development corpus manifest"),
        label_reveal_protocol_digest=_digest("official side-label reveal v1"),
    )
    return plan, paths, selections


def _measurements(
    plan: SemanticCalibrationPlan,
    protocol: SoftScorerProtocol,
    paths: Sequence[Path],
    selections: Sequence[CalibrationPanelSelection],
    scores: Sequence[float],
    labels: Sequence[bool],
):
    return tuple(
        join_calibration_label(
            plan,
            protocol,
            selection.observation_id,
            _score_artifact(protocol, path, selection, score, index),
            label,
            label_reveal_receipt_digest=_digest(f"label reveal {index}"),
        )
        for index, (path, selection, score, label) in enumerate(
            zip(paths, selections, scores, labels, strict=True)
        )
    )


def test_label_free_plan_then_receipt_parented_join_and_exact_family_fit(
    tmp_path: Path, protocol: SoftScorerProtocol
) -> None:
    plan, paths, selections = _plan_and_inputs(tmp_path, protocol)
    assert SemanticCalibrationPlan.from_data(
        plan.to_data(), expected_digest=plan.digest
    ) == plan
    encoded_plan = json.dumps(plan.to_data(), sort_keys=True)
    assert "affirmative_label" not in encoded_plan
    assert "polarity" not in encoded_plan
    assert "source_path" not in encoded_plan
    assert '"label_state": "withheld"' in encoded_plan

    measurements = _measurements(
        plan, protocol, paths, selections, SCORES, LABELS
    )
    first = measurements[0]
    assert first.join_receipt.score_artifact_digest == first.score_artifact_digest
    assert (
        first.join_receipt.score_record_digest
        == first.development_unit.score_record_digest
    )
    assert (
        first.join_receipt.scorer_receipt_digest
        == _score_artifact(protocol, paths[0], selections[0], SCORES[0], 0)
        .record.scorer_receipt_digest
    )
    assert first.development_unit.annotation_receipt_digest == (
        first.join_receipt.digest
    )
    assert first.join_receipt.content_data()["causal_order"] == (
        "sealed_score_artifact_and_receipt_then_label_join/v1"
    )
    assert CalibrationLabelJoinReceipt.from_data(
        first.join_receipt.to_data(),
        expected_digest=first.join_receipt.digest,
    ) == first.join_receipt
    assert SemanticCalibrationMeasurement.from_data(
        first.to_data(), expected_digest=first.digest
    ) == first

    calibration = fit_semantic_calibration(plan, protocol, measurements)
    calibration.assert_untampered()
    assert calibration.family.development_units == tuple(
        item.development_unit for item in measurements
    )
    assert calibration.to_data()["development_manifest_digest"] == (
        calibration.family.development_manifest_digest
    )
    assert SoftScorerFamily.from_data(
        calibration.family.to_data(),
        expected_digest=calibration.family.digest(),
    ) == calibration.family
    assert SemanticCalibrationArtifact.from_data(
        calibration.to_data(), expected_digest=calibration.digest
    ) == calibration

    tampered = copy.deepcopy(calibration.to_data())
    tampered["accepted_units"][0]["affirmative_label"] = True
    with pytest.raises(SoftPredicateIntegrityError, match="accepted units differ"):
        SemanticCalibrationArtifact.from_data(tampered)


def test_official_test_task_is_rejected_without_reading_pixels(
    protocol: SoftScorerProtocol,
) -> None:
    selection = CalibrationPanelSelection(
        observation_id="development-test",
        task_id="official-test-task",
        panel_id="opaque-panel-test",
        panel_digest=_digest("unread test panel identity"),
        split="train",
        dependence_cluster_id="cluster-test",
    )
    with pytest.raises(SemanticCalibrationError, match="official test task"):
        SemanticCalibrationPlan.create(
            protocol,
            _split((), test_ids=(selection.task_id,)),
            (selection,),
            corpus_manifest_digest=_address("complete corpus manifest"),
            development_manifest_digest=_address("development corpus manifest"),
            label_reveal_protocol_digest=_digest("label reveal protocol"),
        )


@pytest.mark.parametrize("duplicate_field", ("task", "panel", "panel_digest"))
def test_plan_rejects_duplicate_task_or_panel_identity(
    tmp_path: Path,
    protocol: SoftScorerProtocol,
    duplicate_field: str,
) -> None:
    paths = (_write_panel(tmp_path, 0), _write_panel(tmp_path, 1))
    first = _selection(paths[0], 0)
    second = _selection(paths[1], 1)
    if duplicate_field == "task":
        second = replace(second, task_id=first.task_id)
    elif duplicate_field == "panel":
        second = replace(second, panel_id=first.panel_id)
    else:
        second = replace(second, panel_digest=first.panel_digest)
    with pytest.raises(SemanticCalibrationError, match="repeats"):
        SemanticCalibrationPlan.create(
            protocol,
            _split(tuple({first.task_id, second.task_id})),
            (first, second),
            corpus_manifest_digest=_address("corpus"),
            development_manifest_digest=_address("development corpus"),
            label_reveal_protocol_digest=_digest("reveal protocol"),
        )


def test_failed_score_record_cannot_receive_calibration_label(
    tmp_path: Path, protocol: SoftScorerProtocol
) -> None:
    plan, paths, selections = _plan_and_inputs(tmp_path, protocol, count=1)
    failed = _score_artifact(
        protocol,
        paths[0],
        selections[0],
        0.0,
        0,
        fail_transport=True,
    )
    assert failed.record.outcome == "transport_error"
    assert failed.record.score is None
    with pytest.raises(SemanticCalibrationError, match="failed scorer"):
        join_calibration_label(
            plan,
            protocol,
            selections[0].observation_id,
            failed,
            False,
            label_reveal_receipt_digest=_digest("label reveal"),
        )


def test_wrong_protocol_score_artifact_is_rejected(
    tmp_path: Path, protocol: SoftScorerProtocol
) -> None:
    plan, paths, selections = _plan_and_inputs(tmp_path, protocol, count=1)
    other = replace(protocol, version="2")
    artifact = _score_artifact(other, paths[0], selections[0], 0.0, 0)
    with pytest.raises(SemanticCalibrationError, match="different scorer protocol"):
        join_calibration_label(
            plan,
            protocol,
            selections[0].observation_id,
            artifact,
            False,
            label_reveal_receipt_digest=_digest("label reveal"),
        )


def test_model_call_overlap_is_rejected_before_family_admission(
    tmp_path: Path, protocol: SoftScorerProtocol
) -> None:
    plan, paths, selections = _plan_and_inputs(tmp_path, protocol)
    measurements = list(
        _measurements(plan, protocol, paths, selections, SCORES, LABELS)
    )
    repeated_call_artifact = _score_artifact(
        protocol,
        paths[1],
        selections[1],
        SCORES[1],
        1,
        scorer_call_id=measurements[0].development_unit.scorer_call_id,
    )
    measurements[1] = join_calibration_label(
        plan,
        protocol,
        selections[1].observation_id,
        repeated_call_artifact,
        LABELS[1],
        label_reveal_receipt_digest=_digest("replacement label reveal"),
    )
    with pytest.raises((SemanticCalibrationError, ValueError), match="scorer_call"):
        fit_semantic_calibration(plan, protocol, measurements)


def test_bin_underpopulation_is_rejected_before_fit(
    tmp_path: Path, protocol: SoftScorerProtocol
) -> None:
    plan, paths, selections = _plan_and_inputs(tmp_path, protocol, count=3)
    measurements = _measurements(
        plan,
        protocol,
        paths,
        selections,
        (0.0, 0.5, 1.0),
        (False, True, True),
    )
    with pytest.raises(SemanticCalibrationError, match="underpopulated"):
        fit_semantic_calibration(plan, protocol, measurements)
