from __future__ import annotations

import hashlib
import json

import pytest

from bongard.canonical import canonical_digest
from bongard.panel_feature_proposer import (
    PANEL_FEATURE_BLOCKS,
    PANEL_FEATURE_NONE,
    PANEL_FEATURE_PRESENTATION_NAMES,
    PANEL_FEATURE_SLOTS_PER_DIRECTION,
    PanelFeatureNominationGapCode,
    PanelFeatureProposerCallResult,
    PanelFeatureProposerError,
    invoke_panel_feature_proposer,
    panel_feature_proposer_output_schema,
    panel_feature_proposer_prompt,
    panel_feature_spec_from_wire,
    panel_feature_spec_to_wire,
    parse_panel_feature_proposer_payload,
)
from bongard.panel_soft_ontology import (
    ClosedCount,
    ComponentCountParameters,
    FeatureFamily,
    LanguageGapKind,
    NativeOrientation,
    PanelFeatureSpec,
    ReferenceFrame,
    SEGMENT_MEMBERSHIP_RULE_ID,
    STRAIGHT_SEGMENT_CLASSIFICATION_RULE_ID,
    StraightSegmentCountParameters,
    SubjectScope,
)
from bongard.transport import validate_codex_strict_output_schema


_D1 = "1" * 64
_D2 = "2" * 64
_D3 = "3" * 64


def _count_spec(count: ClosedCount) -> PanelFeatureSpec:
    return PanelFeatureSpec(
        FeatureFamily.COMPONENT_COUNT,
        SubjectScope.WHOLE_PANEL,
        ReferenceFrame.NONE,
        ComponentCountParameters(count),
    )


def _straight_count_spec(count: ClosedCount) -> PanelFeatureSpec:
    return PanelFeatureSpec(
        FeatureFamily.STRAIGHT_SEGMENT_COUNT,
        SubjectScope.WHOLE_PANEL,
        ReferenceFrame.NONE,
        StraightSegmentCountParameters(count),
    )


def _row(
    spec: PanelFeatureSpec,
    *,
    native_block: str,
    native_support: int = 6,
    contrast_support: int = 0,
    narration_suffix: str = "",
) -> dict[str, object]:
    wire = panel_feature_spec_to_wire(spec)
    result: dict[str, object] = {
        "candidate_kind": "registered_feature",
        **wire,
        "language_gap_kind": PANEL_FEATURE_NONE,
        "archival_summary": f"A visible registered count pattern {narration_suffix}".strip(),
        "archival_indicator_a": f"A coherent visible grouping appears {narration_suffix}".strip(),
        "archival_indicator_b": f"A complete panel region supports the grouping {narration_suffix}".strip(),
    }
    for block in PANEL_FEATURE_BLOCKS:
        count = native_support if block == native_block else contrast_support
        for index in range(6):
            result[f"{block}_panel_{index:03d}_estimate"] = (
                "supports" if index < count else "does_not_support"
            )
    return result


def _language_gap_row(kind: LanguageGapKind, *, native_block: str) -> dict[str, object]:
    result: dict[str, object] = {
        "candidate_kind": "language_gap",
        "feature_family": PANEL_FEATURE_NONE,
        "subject_scope": PANEL_FEATURE_NONE,
        "reference_frame": PANEL_FEATURE_NONE,
        "parameter_a": PANEL_FEATURE_NONE,
        "parameter_b": PANEL_FEATURE_NONE,
        "parameter_c": PANEL_FEATURE_NONE,
        "language_gap_kind": kind.value,
        "archival_summary": "A visible unfamiliar gestalt recurs",
        "archival_indicator_a": "A coherent outline supplies the first cue",
        "archival_indicator_b": "A local appendage supplies the second cue",
    }
    for block in PANEL_FEATURE_BLOCKS:
        for index in range(6):
            result[f"{block}_panel_{index:03d}_estimate"] = (
                "supports" if block == native_block else "does_not_support"
            )
    return result


def _payload() -> dict[str, object]:
    counts = tuple(ClosedCount)
    result: dict[str, object] = {}
    for block_index, block in enumerate(PANEL_FEATURE_BLOCKS):
        for slot in range(PANEL_FEATURE_SLOTS_PER_DIRECTION):
            count = counts[block_index * PANEL_FEATURE_SLOTS_PER_DIRECTION + slot]
            result[f"{block}_candidate_{slot}"] = _row(
                _count_spec(count),
                native_block=block,
                narration_suffix=f"{block} slot {slot}",
            )
    return result


def _parse(payload: dict[str, object], **kwargs):
    return parse_panel_feature_proposer_payload(
        payload,
        proposer_receipt_digest=_D1,
        support_set_digest=_D2,
        task_context_digest=_D3,
        **kwargs,
    )


def test_schema_is_strict_fixed_four_slots_per_neutral_direction() -> None:
    schema = panel_feature_proposer_output_schema()
    validate_codex_strict_output_schema(schema)
    assert set(schema["properties"]) == {
        f"{block}_candidate_{slot}"
        for block in PANEL_FEATURE_BLOCKS
        for slot in range(4)
    }
    assert schema["required"] == list(schema["properties"])
    assert schema["additionalProperties"] is False
    candidate = schema["properties"]["block_a_candidate_0"]
    assert set(candidate["properties"]) == set(_payload()["block_a_candidate_0"])
    assert sum(name.endswith("_estimate") for name in candidate["properties"]) == 12
    assert all(
        candidate["properties"][name]["enum"]
        == ["supports", "does_not_support", "unclear"]
        for name in candidate["properties"]
        if name.endswith("_estimate")
    )


def test_prompt_is_symmetric_contrastive_and_has_no_task_or_semantic_side() -> None:
    prompt = panel_feature_proposer_prompt()
    assert "block_a" in prompt and "block_b" in prompt
    assert "all twelve" in prompt
    assert "at least five" in prompt and "does_not_support" in prompt
    assert "unclear does not count" in prompt
    lowered = prompt.lower()
    assert "side0" not in lowered and "side1" not in lowered
    assert "query" not in lowered and "task_id" not in lowered


def test_closed_wire_round_trip_does_not_consult_narration() -> None:
    spec = _count_spec(ClosedCount.FOUR)
    assert panel_feature_spec_from_wire(panel_feature_spec_to_wire(spec)) == spec
    bad = panel_feature_spec_to_wire(spec)
    bad["parameter_b"] = "unexpected"
    with pytest.raises(PanelFeatureProposerError, match="unused"):
        panel_feature_spec_from_wire(bad)


def test_straight_and_generic_segment_counts_have_distinct_wire_semantics() -> None:
    straight = _straight_count_spec(ClosedCount.FOUR)
    assert panel_feature_spec_from_wire(panel_feature_spec_to_wire(straight)) == straight
    prompt = panel_feature_proposer_prompt()
    assert "exact_segment_count counts every registered segment owner" in prompt
    assert "straight_segment_count counts only visibly straight" in prompt
    assert SEGMENT_MEMBERSHIP_RULE_ID in prompt
    assert STRAIGHT_SEGMENT_CLASSIFICATION_RULE_ID in prompt


def test_exact_admission_boundary_and_global_observer_vocabulary() -> None:
    payload = _payload()
    payload["block_a_candidate_0"] = _row(
        _count_spec(ClosedCount.ONE),
        native_block="block_a",
        native_support=5,
        contrast_support=1,
        narration_suffix="exact admission boundary",
    )
    result = _parse(payload)
    assert len(result.nominations) == 8
    assert result.nomination_gaps == ()
    assert result.observer_vocabulary is not None
    assert len(result.observer_vocabulary.specs) == 8
    nomination = result.nominations[0]
    assert (
        nomination.native_support_count,
        nomination.native_unclear_count,
        nomination.contrast_support_count,
        nomination.contrast_does_not_support_count,
        nomination.contrast_unclear_count,
        nomination.support_margin,
    ) == (5, 0, 1, 5, 0, 4)


def test_all_unclear_contrast_is_missing_evidence_not_negative_evidence() -> None:
    payload = _payload()
    row = _row(
        _count_spec(ClosedCount.ONE),
        native_block="block_a",
        native_support=5,
        narration_suffix="contrast evidence is missing",
    )
    for index in range(6):
        row[f"block_a_panel_{index:03d}_estimate"] = (
            "supports" if index < 5 else "unclear"
        )
        row[f"block_b_panel_{index:03d}_estimate"] = "unclear"
    payload["block_a_candidate_0"] = row
    result = _parse(payload)
    assert not any(
        item.source_block == "block_a" and item.raw_slot == 0
        for item in result.nominations
    )
    assert any(
        gap.native_orientation is NativeOrientation.SIDE0_POSITIVE
        and gap.raw_slot == 0
        and gap.code
        is PanelFeatureNominationGapCode.CONTRASTIVE_ADMISSION_REJECTED
        for gap in result.nomination_gaps
    )


def test_shared_salience_is_rejected_instead_of_becoming_a_candidate() -> None:
    payload = _payload()
    payload["block_a_candidate_0"] = _row(
        _count_spec(ClosedCount.ONE),
        native_block="block_a",
        native_support=6,
        contrast_support=6,
        narration_suffix="shared salience",
    )
    result = _parse(payload)
    assert len(result.nominations) == 7
    assert any(
        gap.code is PanelFeatureNominationGapCode.SHARED_SALIENCE_REJECTED
        for gap in result.nomination_gaps
    )


def test_all_twelve_estimates_are_structurally_required() -> None:
    payload = _payload()
    del payload["block_a_candidate_0"]["block_b_panel_005_estimate"]
    with pytest.raises(PanelFeatureProposerError, match="candidate fields"):
        _parse(payload)


def test_same_spec_across_orientations_is_a_global_contradiction() -> None:
    payload = _payload()
    shared = _count_spec(ClosedCount.ONE)
    payload["block_a_candidate_0"] = _row(
        shared, native_block="block_a", narration_suffix="from block a"
    )
    payload["block_b_candidate_0"] = _row(
        shared, native_block="block_b", narration_suffix="from block b"
    )
    result = _parse(payload)
    shared_nominations = [item for item in result.nominations if item.spec == shared]
    assert shared_nominations == []
    conflicts = [
        gap
        for gap in result.nomination_gaps
        if gap.code is PanelFeatureNominationGapCode.GLOBAL_SPEC_CONTRADICTION
        and gap.raw_slot == 0
    ]
    assert {item.native_orientation for item in conflicts} == set(NativeOrientation)
    assert result.observer_vocabulary is not None
    assert shared not in result.observer_vocabulary.specs


def test_conflicting_estimate_vectors_reject_every_same_orientation_copy() -> None:
    payload = _payload()
    shared = _count_spec(ClosedCount.ONE)
    payload["block_a_candidate_0"] = _row(
        shared,
        native_block="block_a",
        native_support=6,
        contrast_support=0,
        narration_suffix="first estimate vector",
    )
    payload["block_a_candidate_1"] = _row(
        shared,
        native_block="block_a",
        native_support=5,
        contrast_support=1,
        narration_suffix="contradictory estimate vector",
    )
    result = _parse(payload)
    assert not any(item.spec == shared for item in result.nominations)
    conflicts = [
        gap
        for gap in result.nomination_gaps
        if gap.code is PanelFeatureNominationGapCode.GLOBAL_SPEC_CONTRADICTION
        and gap.native_orientation is NativeOrientation.SIDE0_POSITIVE
    ]
    assert {item.raw_slot for item in conflicts} >= {0, 1}


def test_duplicate_within_native_orientation_is_a_typed_nomination_gap() -> None:
    payload = _payload()
    payload["block_a_candidate_1"] = _row(
        _count_spec(ClosedCount.ONE),
        native_block="block_a",
        narration_suffix="duplicate native feature",
    )
    result = _parse(payload)
    assert any(
        gap.code is PanelFeatureNominationGapCode.DUPLICATE_NATIVE_SPEC
        for gap in result.nomination_gaps
    )


def test_explicit_unregistered_concept_becomes_typed_language_gap() -> None:
    payload = _payload()
    payload["block_a_candidate_0"] = _language_gap_row(
        LanguageGapKind.UNREGISTERED_GESTALT,
        native_block="block_a",
    )
    result = _parse(payload)
    assert len(result.language_gaps) == 1
    assert result.language_gaps[0].kind is LanguageGapKind.UNREGISTERED_GESTALT
    assert all(
        item.spec != _count_spec(ClosedCount.ONE) for item in result.nominations
    )


def test_invalid_registered_wire_becomes_language_gap_not_prose_compilation() -> None:
    payload = _payload()
    row = dict(payload["block_a_candidate_0"])
    row["feature_family"] = "invented_visual_family"
    row["archival_summary"] = "A birdlike object appears"
    payload["block_a_candidate_0"] = row
    result = _parse(payload)
    assert len(result.language_gaps) == 1
    assert result.language_gaps[0].kind is LanguageGapKind.AMBIGUOUS_FAMILY
    assert result.observer_vocabulary is not None
    assert all(item.family is not None for item in result.observer_vocabulary.specs)


def test_narration_is_archival_and_cannot_change_observer_vocabulary_identity() -> None:
    first_payload = _payload()
    second_payload = _payload()
    second_payload["block_a_candidate_0"] = {
        **second_payload["block_a_candidate_0"],
        "archival_summary": "A completely different human explanation",
        "archival_indicator_a": "A different visible cue appears",
        "archival_indicator_b": "Another different visible cue appears",
    }
    first = _parse(first_payload)
    second = _parse(second_payload)
    assert first.observer_vocabulary is not None
    assert second.observer_vocabulary is not None
    assert first.observer_vocabulary.vocabulary_digest == second.observer_vocabulary.vocabulary_digest
    assert first.result_digest != second.result_digest


def test_block_swap_preserves_global_spec_set_and_swaps_only_provenance() -> None:
    original_payload = _payload()
    swapped_payload: dict[str, object] = {}
    for target_block, source_block in (("block_a", "block_b"), ("block_b", "block_a")):
        for slot in range(4):
            source = original_payload[f"{source_block}_candidate_{slot}"]
            spec = panel_feature_spec_from_wire(
                {
                    key: source[key]
                    for key in (
                        "feature_family",
                        "subject_scope",
                        "reference_frame",
                        "parameter_a",
                        "parameter_b",
                        "parameter_c",
                    )
                }
            )
            swapped_payload[f"{target_block}_candidate_{slot}"] = _row(
                spec,
                native_block=target_block,
                narration_suffix=f"swapped {target_block} slot {slot}",
            )
    original = _parse(original_payload)
    swapped = _parse(
        swapped_payload,
        block_orientations=(
            NativeOrientation.SIDE1_POSITIVE,
            NativeOrientation.SIDE0_POSITIVE,
        ),
    )
    assert original.observer_vocabulary is not None
    assert swapped.observer_vocabulary is not None
    assert {
        item.spec_digest for item in original.observer_vocabulary.specs
    } == {item.spec_digest for item in swapped.observer_vocabulary.specs}


def test_observer_vocabulary_excludes_block_orientation_task_query_and_narration() -> None:
    result = _parse(_payload())
    assert result.observer_vocabulary is not None
    rendered = json.dumps(result.observer_vocabulary.to_data(), sort_keys=True)
    for forbidden in (
        "block_a",
        "block_b",
        "side0_positive",
        "side1_positive",
        "query",
        _D3,
        "archival_summary",
    ):
        assert forbidden not in rendered


def test_canonical_vnext_result_has_no_theorem_prover_policy_fields() -> None:
    rendered = json.dumps(_parse(_payload()).to_data(), sort_keys=True).lower()
    assert "lean" not in rendered


def test_injected_receipted_boundary_binds_prompt_schema_and_exact_presentations() -> None:
    payload = _payload()
    panels = tuple(f"panel-{index}".encode() for index in range(12))
    calls = 0

    def call(named_images, prompt, schema):
        nonlocal calls
        calls += 1
        assert tuple(name for name, _ in named_images) == PANEL_FEATURE_PRESENTATION_NAMES
        presentation_digest = canonical_digest(
            {
                "schema": "gkm.bongard-panel-feature-proposer-presentation.v1",
                "images": [
                    {"name": name, "sha256": hashlib.sha256(raw).hexdigest()}
                    for name, raw in named_images
                ],
            }
        )
        return PanelFeatureProposerCallResult.seal(
            payload,
            prompt=prompt,
            output_schema=schema,
            presentation_digest=presentation_digest,
            external_receipt_digest=_D1,
        )

    result = invoke_panel_feature_proposer(
        panels,
        task_context_digest=_D3,
        call=call,
    )
    assert calls == 1
    assert len(result.nominations) == 8


def test_injected_boundary_rejects_receipt_for_another_request() -> None:
    panels = tuple(f"panel-{index}".encode() for index in range(12))

    def call(named_images, prompt, schema):
        result = PanelFeatureProposerCallResult.seal(
            _payload(),
            prompt=prompt,
            output_schema=schema,
            presentation_digest=_D2,
            external_receipt_digest=_D1,
        )
        assert result.presentation_digest == _D2
        return result

    with pytest.raises(PanelFeatureProposerError, match="does not bind"):
        invoke_panel_feature_proposer(
            panels,
            task_context_digest=_D3,
            call=call,
        )
