from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import replace

import pytest

from bongard.evidence import Disposition
from bongard.soft_predicates import (
    BlindSoftScoreRecord,
    SoftCueJudgment,
    SoftFamilyDevelopmentUnit,
    SoftPredicateIntegrityError,
    SoftScorerFamily,
    SoftScorerProtocol,
    blind_soft_score_output_schema,
    compare_blind_soft_score,
    measure_blind_soft_score,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def protocol() -> SoftScorerProtocol:
    return SoftScorerProtocol(
        family_id="open-semantic-positive-cues",
        version="1",
        proposer_grammar_id="positive-cue-rubric-v1",
        proposer_grammar_digest=_digest("proposer-grammar"),
        proposer_model_id="gpt-5.6-sol",
        proposer_reasoning_effort="medium",
        proposer_prompt_id="soft-claim-proposer-v1",
        proposer_prompt_digest=_digest("proposer-prompt"),
        scorer_model_id="gpt-5.6-sol",
        scorer_reasoning_effort="medium",
        scorer_prompt_template_id="blind-one-panel-cue-scorer-template-v1",
        scorer_prompt_template_digest=_digest("scorer-prompt-template"),
        scorer_decoder_id="closed-cue-ordinal-decoder-v1",
        scorer_decoder_digest=_digest("scorer-decoder"),
        ordinal_map=(
            ("supported", 1.0),
            ("ambiguous", 0.5),
            ("unsupported", 0.0),
        ),
        aggregation="min",
        witness_extractor_id="joint-panel-witnesses-v1",
        witness_extractor_digest=_digest("witness-extractor"),
        support_gate_id="exact-aligned-6-plus-6-v1",
        support_gate_digest=_digest("support-gate"),
        score_bin_edges=(0.0, 0.25, 0.75, 1.0),
        affirmative_boundary=0.7,
        confidence_level=0.8,
        minimum_clusters_per_bin=40,
    )


def _model_output(first: str, second: str) -> dict[str, object]:
    return {
        "cue_judgments": [
            {
                "cue_id": "bird.body",
                "judgment": first,
                "witness_ids": (
                    [] if first == "unsupported" else ["component:0"]
                ),
            },
            {
                "cue_id": "bird.beak",
                "judgment": second,
                "witness_ids": (
                    [] if second == "unsupported" else ["contour:3"]
                ),
            },
        ]
    }


def _context(
    protocol: SoftScorerProtocol,
    *,
    prefix: str = "fresh",
    claim: str = "fresh-claim",
) -> dict[str, object]:
    return {
        "scorer_protocol_digest": protocol.digest(),
        "task_id": f"{prefix}-task",
        "panel_id": f"{prefix}-opaque-panel",
        "panel_digest": _digest(f"{prefix}-panel"),
        "claim_digest": _digest(claim),
        "proposer_call_id": f"{prefix}-proposer-call",
        "proposer_receipt_digest": _digest(f"{prefix}-proposer-receipt"),
        "scorer_call_id": f"{prefix}-scorer-call",
        "scorer_receipt_digest": _digest(f"{prefix}-scorer-receipt"),
        "witness_packet_digest": _digest(f"{prefix}-witness-packet"),
        "pre_observation_commitment_digest": _digest(
            f"{prefix}-proposal-policy-commitment"
        ),
        "declared_cue_ids": ("bird.body", "bird.beak"),
        "verifier_witness_ids": ("component:0", "contour:3"),
    }


def _record(
    protocol: SoftScorerProtocol,
    first: str = "supported",
    second: str = "supported",
    *,
    prefix: str = "fresh",
    claim: str = "fresh-claim",
) -> BlindSoftScoreRecord:
    return BlindSoftScoreRecord.from_model_output(
        _model_output(first, second),
        **_context(protocol, prefix=prefix, claim=claim),  # type: ignore[arg-type]
    )


def _development_material(
    protocol: SoftScorerProtocol,
) -> tuple[
    tuple[BlindSoftScoreRecord, ...],
    tuple[SoftFamilyDevelopmentUnit, ...],
]:
    records: list[BlindSoftScoreRecord] = []
    units: list[SoftFamilyDevelopmentUnit] = []
    levels = (
        (0.0, "unsupported", "supported"),
        (0.5, "ambiguous", "supported"),
        (1.0, "supported", "supported"),
    )
    for bin_index, (score, first, second) in enumerate(levels):
        for local_index in range(40):
            index = bin_index * 40 + local_index
            prefix = f"development-{index:03d}"
            record = _record(
                protocol,
                first,
                second,
                prefix=prefix,
                claim=f"development-claim-{index}",
            )
            records.append(record)
            label = (
                False
                if bin_index == 0
                else True
                if bin_index == 2
                else local_index >= 20
            )
            units.append(
                SoftFamilyDevelopmentUnit(
                    observation_id=prefix,
                    task_id=record.task_id,
                    panel_digest=record.panel_digest,
                    claim_digest=record.claim_digest,
                    scorer_protocol_digest=record.scorer_protocol_digest,
                    proposer_call_id=record.proposer_call_id,
                    scorer_call_id=record.scorer_call_id,
                    dependence_cluster_id=f"development-cluster-{index:03d}",
                    score_record_digest=record.digest(),
                    annotation_receipt_digest=_digest(
                        f"development-annotation-{index}"
                    ),
                    score=record.score,  # type: ignore[arg-type]
                    affirmative_label=label,
                    score_bin_index=bin_index,
                )
            )
    return tuple(records), tuple(units)


@pytest.fixture(scope="module")
def development_material(
    protocol: SoftScorerProtocol,
) -> tuple[
    tuple[BlindSoftScoreRecord, ...],
    tuple[SoftFamilyDevelopmentUnit, ...],
]:
    return _development_material(protocol)


@pytest.fixture(scope="module")
def family(
    protocol: SoftScorerProtocol,
    development_material: tuple[
        tuple[BlindSoftScoreRecord, ...],
        tuple[SoftFamilyDevelopmentUnit, ...],
    ],
) -> SoftScorerFamily:
    _, units = development_material
    return SoftScorerFamily.fit(
        protocol,
        units,
        expected_protocol_digest=protocol.digest(),
    )


def test_protocol_is_strict_prospective_identity(
    protocol: SoftScorerProtocol,
) -> None:
    restored = SoftScorerProtocol.from_data(
        protocol.to_data(), expected_digest=protocol.digest()
    )
    assert restored == protocol
    data = protocol.to_data()
    assert data["proposer"]["prompt_digest"] == _digest("proposer-prompt")
    assert data["scorer"]["prompt_template_digest"] == _digest(
        "scorer-prompt-template"
    )
    assert data["calibration_protocol"]["algorithm_id"] == (
        "fixed_bin_cluster_family_raw_simultaneous_hoeffding_v2"
    )
    assert "development_manifest" not in data
    assert "calibrated_support_intervals" not in str(data)
    unknown = dict(data)
    unknown["claim_digest"] = _digest("forbidden-exact-claim")
    with pytest.raises(ValueError, match="unknown fields"):
        SoftScorerProtocol.from_data(unknown)


def test_prospective_protocol_then_records_then_family_has_no_cycle(
    protocol: SoftScorerProtocol,
    development_material: tuple[
        tuple[BlindSoftScoreRecord, ...],
        tuple[SoftFamilyDevelopmentUnit, ...],
    ],
) -> None:
    protocol_digest_before_development = protocol.digest()
    records, units = development_material
    assert records
    assert all(
        record.scorer_protocol_digest == protocol_digest_before_development
        for record in records
    )
    assert all(
        unit.scorer_protocol_digest == protocol_digest_before_development
        for unit in units
    )
    assert all("family_digest" not in record.to_data() for record in records)

    fitted = SoftScorerFamily.fit(
        protocol,
        units,
        expected_protocol_digest=protocol_digest_before_development,
    )
    assert fitted.protocol_digest == protocol_digest_before_development
    assert fitted.digest() != protocol_digest_before_development
    assert all(
        record.scorer_protocol_digest == fitted.protocol_digest
        for record in records
    )
    fitted.verify_calibration()


def test_family_roundtrip_binds_protocol_manifest_and_fit(
    family: SoftScorerFamily,
    protocol: SoftScorerProtocol,
) -> None:
    family.verify_calibration()
    data = family.to_data()
    restored = SoftScorerFamily.from_data(data, expected_digest=family.digest())
    assert restored == family
    assert data["protocol"] == protocol.to_data()
    assert data["protocol_digest"] == protocol.digest()
    assert data["development_manifest"]["protocol_digest"] == protocol.digest()
    assert len(data["dependence_clusters"]) == 120
    assert len(
        data["calibration_artifact"]["calibrated_support_intervals"]
    ) == 3
    assert data["calibration_artifact"]["algorithm_id"] == (
        "fixed_bin_cluster_family_raw_simultaneous_hoeffding_v2"
    )

    first = _record(protocol, claim="fresh-claim-a")
    second = _record(protocol, claim="fresh-claim-b")
    assert first.claim_digest != second.claim_digest
    assert (
        first.scorer_protocol_digest
        == second.scorer_protocol_digest
        == protocol.digest()
    )


def test_nonmonotone_bin_population_keeps_raw_simultaneous_intervals(
    protocol: SoftScorerProtocol,
    development_material: tuple[
        tuple[BlindSoftScoreRecord, ...],
        tuple[SoftFamilyDevelopmentUnit, ...],
    ],
) -> None:
    _, units = development_material
    nonmonotone_units = tuple(
        replace(unit, affirmative_label=unit.score_bin_index == 0)
        for unit in units
    )

    fitted = SoftScorerFamily.fit(
        protocol,
        nonmonotone_units,
        expected_protocol_digest=protocol.digest(),
    )

    bin_count = len(protocol.score_bin_edges) - 1
    radius = math.sqrt(
        math.log((2.0 * bin_count) / (1.0 - protocol.confidence_level))
        / (2.0 * protocol.minimum_clusters_per_bin)
    )
    intervals = fitted.calibrated_support_intervals
    assert intervals[0][0] == pytest.approx(1.0 - radius)
    assert intervals[0][1] == 1.0
    for lower, upper in intervals[1:]:
        assert lower == 0.0
        assert upper == pytest.approx(radius)

    # The population really is nonmonotone.  No cross-bin cumulative max/min
    # may silently tighten it, and fitting it must not reject it.
    assert intervals[0][0] > intervals[1][1]
    fitted.verify_calibration()


def test_strict_cue_and_blind_record_roundtrip(
    family: SoftScorerFamily,
    protocol: SoftScorerProtocol,
) -> None:
    cue = SoftCueJudgment("bird.body", "supported", ("component:0",))
    assert SoftCueJudgment.from_data(
        cue.to_data(), expected_digest=cue.digest()
    ) == cue
    record = _record(protocol)
    assert BlindSoftScoreRecord.from_data(
        record.to_data(), expected_digest=record.digest()
    ) == record

    unknown = dict(cue.to_data())
    unknown["score"] = 1.0
    with pytest.raises(ValueError, match="unknown fields"):
        SoftCueJudgment.from_data(unknown)

    changed_score = copy.deepcopy(record.to_data())
    changed_score["derived_score"] = 0.5
    with pytest.raises(SoftPredicateIntegrityError, match="differs from Python"):
        BlindSoftScoreRecord.from_data(changed_score)

    assert measure_blind_soft_score(
        family, record, expected_family_digest=family.digest()
    ).disposition is Disposition.PRESENT


def test_missing_or_repeated_declared_cues_are_rejected(
    protocol: SoftScorerProtocol,
) -> None:
    with pytest.raises(ValueError, match="missing cues"):
        BlindSoftScoreRecord.from_model_output(
            {
                "cue_judgments": [
                    {
                        "cue_id": "bird.body",
                        "judgment": "supported",
                        "witness_ids": ["component:0"],
                    }
                ]
            },
            **_context(protocol),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="repeated"):
        BlindSoftScoreRecord.from_model_output(
            {
                "cue_judgments": [
                    {
                        "cue_id": "bird.body",
                        "judgment": "supported",
                        "witness_ids": ["component:0"],
                    },
                    {
                        "cue_id": "bird.body",
                        "judgment": "ambiguous",
                        "witness_ids": ["contour:3"],
                    },
                ]
            },
            **_context(protocol),  # type: ignore[arg-type]
        )


def test_forged_model_witness_id_is_rejected(
    protocol: SoftScorerProtocol,
) -> None:
    with pytest.raises(SoftPredicateIntegrityError, match="non-verifier witness"):
        BlindSoftScoreRecord.from_model_output(
            {
                "cue_judgments": [
                    {
                        "cue_id": "bird.body",
                        "judgment": "supported",
                        "witness_ids": ["component:0"],
                    },
                    {
                        "cue_id": "bird.beak",
                        "judgment": "supported",
                        "witness_ids": ["invented:99"],
                    },
                ]
            },
            **_context(protocol),  # type: ignore[arg-type]
        )


def test_zero_is_present_before_python_comparison(
    family: SoftScorerFamily,
    protocol: SoftScorerProtocol,
) -> None:
    record = _record(protocol, first="unsupported")
    measurement = measure_blind_soft_score(
        family, record, expected_family_digest=family.digest()
    )
    assert measurement.disposition is Disposition.PRESENT
    assert measurement.unwrap() == 0.0
    assert measurement.uncertainty is not None
    assert measurement.uncertainty.lower == measurement.uncertainty.upper == 0.0

    compared = compare_blind_soft_score(
        family, record, expected_family_digest=family.digest()
    )
    assert compared.disposition is Disposition.CERTIFIED_ABSENT
    assert "operational family-calibrated nonmatch" in (compared.certificate or "")
    assert "not emitted by the model" in (compared.certificate or "")


def test_ambiguity_and_supported_scores_use_calibrated_intervals(
    family: SoftScorerFamily,
    protocol: SoftScorerProtocol,
) -> None:
    ambiguous = _record(protocol, first="ambiguous")
    measurement = measure_blind_soft_score(
        family, ambiguous, expected_family_digest=family.digest()
    )
    assert measurement.disposition is Disposition.PRESENT
    assert measurement.unwrap() == 0.5
    compared = compare_blind_soft_score(
        family, ambiguous, expected_family_digest=family.digest()
    )
    assert compared.disposition is Disposition.INDETERMINATE
    assert compared.uncertainty is not None
    assert compared.uncertainty.lower < 0.7 < compared.uncertainty.upper

    supported = compare_blind_soft_score(
        family,
        _record(protocol),
        expected_family_digest=family.digest(),
    )
    assert supported.disposition is Disposition.PRESENT
    assert supported.unwrap() is True


@pytest.mark.parametrize(
    ("outcome", "error_type"),
    [
        ("transport_error", "SoftScorerTransportError"),
        ("parser_error", "SoftScorerParserError"),
    ],
)
def test_transport_and_parser_failures_are_errors_not_zeroes(
    family: SoftScorerFamily,
    protocol: SoftScorerProtocol,
    outcome: str,
    error_type: str,
) -> None:
    failed = replace(
        _record(protocol),
        outcome=outcome,
        cue_judgments=(),
        failure_reason="scorer did not produce an admitted judgment packet",
    )
    measurement = measure_blind_soft_score(
        family, failed, expected_family_digest=family.digest()
    )
    assert measurement.disposition is Disposition.ERROR
    assert measurement.error_type == error_type
    assert compare_blind_soft_score(
        family, failed, expected_family_digest=family.digest()
    ).disposition is Disposition.ERROR


def test_model_has_no_boolean_or_certified_absence_channel(
    family: SoftScorerFamily,
    protocol: SoftScorerProtocol,
) -> None:
    schema = blind_soft_score_output_schema(
        ("bird.body", "bird.beak"), ("component:0", "contour:3")
    )
    assert set(schema["properties"]) == {"cue_judgments"}
    judgment_schema = schema["properties"]["cue_judgments"]["items"]
    assert set(judgment_schema["properties"]) == {
        "cue_id",
        "judgment",
        "witness_ids",
    }
    assert judgment_schema["properties"]["judgment"]["enum"] == [
        "supported",
        "ambiguous",
        "unsupported",
    ]
    low = _record(protocol, first="unsupported")
    assert measure_blind_soft_score(
        family, low, expected_family_digest=family.digest()
    ).disposition is Disposition.PRESENT

    with pytest.raises(ValueError, match="unknown fields"):
        BlindSoftScoreRecord.from_model_output(
            {
                "cue_judgments": [],
                "disposition": "certified_absent",
            },
            **_context(protocol),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="supported, ambiguous, or unsupported"):
        BlindSoftScoreRecord.from_model_output(
            {
                "cue_judgments": [
                    {
                        "cue_id": "bird.body",
                        "judgment": "certified_absent",
                        "witness_ids": [],
                    },
                    {
                        "cue_id": "bird.beak",
                        "judgment": "supported",
                        "witness_ids": ["contour:3"],
                    },
                ]
            },
            **_context(protocol),  # type: ignore[arg-type]
        )


def test_blind_score_schema_stays_in_responses_api_strict_subset() -> None:
    schema = blind_soft_score_output_schema(
        ("cue-00", "cue-01"),
        ("component:0", "contour:3"),
    )
    forbidden = {
        "oneOf",
        "uniqueItems",
        "minItems",
        "maxItems",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "const",
        "not",
    }
    stack: list[object] = [schema]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            assert forbidden.isdisjoint(node)
            if node.get("type") == "object":
                properties = node.get("properties", {})
                assert node.get("additionalProperties") is False
                assert set(node.get("required", [])) == set(properties)
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
def test_repeated_rubric_allowed_only_on_fresh_observations(
    family: SoftScorerFamily,
    protocol: SoftScorerProtocol,
) -> None:
    repeated_rubric = replace(
        _record(protocol),
        claim_digest=family.development_units[0].claim_digest,
    )
    result = measure_blind_soft_score(
        family, repeated_rubric, expected_family_digest=family.digest()
    )
    assert result.disposition is Disposition.PRESENT

    exact_observation_leak = replace(
        repeated_rubric, task_id=family.development_units[0].task_id
    )
    rejected = measure_blind_soft_score(
        family, exact_observation_leak, expected_family_digest=family.digest()
    )
    assert rejected.disposition is Disposition.ERROR
    assert "overlaps" in (rejected.reason or "")


def test_protocol_final_family_and_interval_tampering_are_detected(
    family: SoftScorerFamily,
    protocol: SoftScorerProtocol,
) -> None:
    protocol_tamper = copy.deepcopy(family.to_data())
    protocol_tamper["protocol"]["support_gate"]["digest"] = _digest(
        "substituted-gate"
    )
    with pytest.raises(SoftPredicateIntegrityError, match="protocol digest mismatch"):
        SoftScorerFamily.from_data(protocol_tamper)

    interval_tamper = copy.deepcopy(family.to_data())
    interval_tamper["calibration_artifact"][
        "calibrated_support_intervals"
    ][0][1] += 0.01
    with pytest.raises(SoftPredicateIntegrityError, match="do not reproduce"):
        SoftScorerFamily.from_data(interval_tamper)

    wrong_final = measure_blind_soft_score(
        family,
        _record(protocol),
        expected_family_digest=_digest("different-final-family"),
    )
    assert wrong_final.disposition is Disposition.ERROR
    assert "frozen policy digest" in (wrong_final.reason or "")

    other_protocol = replace(
        protocol, support_gate_digest=_digest("other-protocol-gate")
    )
    wrong_protocol_record = _record(other_protocol)
    mismatch = measure_blind_soft_score(
        family,
        wrong_protocol_record,
        expected_family_digest=family.digest(),
    )
    assert mismatch.disposition is Disposition.ERROR
    assert "different scorer protocol" in (mismatch.reason or "")

    unsafe_family = SoftScorerFamily.from_data(family.to_data())
    frozen_digest = unsafe_family.digest()
    record = _record(protocol)
    object.__setattr__(
        unsafe_family.protocol,
        "support_gate_digest",
        _digest("unsafe-gate"),
    )
    result = measure_blind_soft_score(
        unsafe_family,
        record,
        expected_family_digest=frozen_digest,
    )
    assert result.disposition is Disposition.ERROR
    assert "changed after sealing" in (result.reason or "")
