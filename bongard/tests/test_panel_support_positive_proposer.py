"""Focused support-only positive proposer custody and admission tests."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from bongard.panel_support_positive_proposer import (
    SUPPORT_POSITIVE_PRESENTATION_NAMES,
    PositiveConjunctionRubric,
    SupportPositiveProposalGap,
    SupportPositiveProposerArtifact,
    SupportPositiveProposerError,
    SupportPositiveProposerRequest,
    SupportPositiveTransportProvenance,
    propose_support_positive_rubric,
    support_positive_proposer_output_schema,
    support_positive_proposer_prompt,
    verify_support_positive_proposer_artifact,
)
from bongard.panel_typed_codex_observer import build_panel_only_observation_context
from bongard.tests.test_prototype_scene_observer import (
    EFFORT,
    LAUNCHER_DIGEST,
    MODEL,
    NO_TOOLS_KWARGS,
    _png,
    _receipt,
)
from bongard.transport import CodexStructuredResult


def _groups() -> tuple[tuple[bytes, ...], tuple[bytes, ...]]:
    return (
        tuple(_png(170 + index) for index in range(6)),
        tuple(_png(180 + index) for index in range(6)),
    )


def _request(
    first: tuple[bytes, ...], second: tuple[bytes, ...]
) -> SupportPositiveProposerRequest:
    runtime = build_panel_only_observation_context(
        first[0],
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        **NO_TOOLS_KWARGS,
    ).runtime
    return SupportPositiveProposerRequest.build(first, second, runtime=runtime)


def _payload(*, admitted: bool = True) -> dict[str, object]:
    result: dict[str, object] = {
        "cue_text": "convex carrier and four straight structural runs",
        "component_1": "convex carrier",
        "component_2": "four straight structural runs",
    }
    for index, name in enumerate(SUPPORT_POSITIVE_PRESENTATION_NAMES):
        result[name.removesuffix(".png") + "_estimate"] = (
            "supports" if index < 6 else "does_not_support"
        )
    if not admitted:
        result["group_a_05_estimate"] = "unclear"
        result["group_b_04_estimate"] = "supports"
        result["group_b_05_estimate"] = "unclear"
    return result


def _transport(payload, expected, calls):
    def call(prompt, paths, names, schema, **kwargs):
        assert tuple(names) == SUPPORT_POSITIVE_PRESENTATION_NAMES
        assert tuple(Path(path).read_bytes() for path in paths) == expected
        calls.append((prompt, tuple(names), deepcopy(schema)))
        return CodexStructuredResult(
            deepcopy(payload), _receipt(prompt, paths, names, schema, payload)
        )

    return call


def _observe(first, second, request, payload, calls):
    return propose_support_positive_rubric(
        first,
        second,
        request=request,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        **NO_TOOLS_KWARGS,
        transport=_transport(payload, (*first, *second), calls),
    )


def test_request_prompt_and_schema_are_exact_support_only_and_contrastive() -> None:
    first, second = _groups()
    request = _request(first, second)
    assert SupportPositiveProposerRequest.from_data(request.to_data()) == request
    assert len(request.presentation) == 12
    assert tuple(item.name for item in request.presentation) == (
        SUPPORT_POSITIVE_PRESENTATION_NAMES
    )

    prompt = support_positive_proposer_prompt(request)
    schema = support_positive_proposer_output_schema(request)
    assert "Group B may be heterogeneous" in prompt
    assert "one short affirmative visual conjunction" in prompt
    assert "Do not state a foil, complement, negative predicate" in prompt
    assert "all six Group A drawings support it" in prompt
    assert "at least five Group B drawings do_not_support it" in prompt
    assert len(schema["properties"]) == 15
    assert schema["additionalProperties"] is False
    assert schema["properties"]["group_a_00_estimate"]["enum"] == [
        "supports",
        "does_not_support",
        "unclear",
    ]


def test_admitted_one_call_artifact_roundtrips_and_cold_replays_without_call() -> None:
    first, second = _groups()
    request = _request(first, second)
    calls: list[object] = []
    artifact = _observe(first, second, request, _payload(), calls)

    assert len(calls) == 1
    assert type(artifact.rubric) is PositiveConjunctionRubric
    assert artifact.proposal_gap is None
    assert artifact.rubric.component_2 == "four straight structural runs"
    assert artifact.transport_provenance.kind == "injected_unverified"
    assert artifact.benchmark_sealable is False
    assert SupportPositiveProposerArtifact.from_data(artifact.to_data()) == artifact

    restored = verify_support_positive_proposer_artifact(
        artifact,
        first,
        second,
        expected_artifact_digest=artifact.artifact_digest,
    )
    assert restored == artifact
    assert len(calls) == 1


def test_failed_estimates_produce_typed_gap_not_a_rubric() -> None:
    first, second = _groups()
    request = _request(first, second)
    artifact = _observe(first, second, request, _payload(admitted=False), [])
    assert artifact.rubric is None
    assert type(artifact.proposal_gap) is SupportPositiveProposalGap
    assert artifact.proposal_gap.group_a_supports == 5
    assert artifact.proposal_gap.group_b_does_not_support == 4
    assert SupportPositiveProposerArtifact.from_data(artifact.to_data()) == artifact


@pytest.mark.parametrize(
    "field,value",
    [
        ("cue_text", "convex carrier without dents and four straight runs"),
        ("component_1", "confidence above 80 percent"),
        ("component_2", "python function returning four"),
    ],
)
def test_prose_rejects_negative_threshold_or_code(
    field: str, value: str
) -> None:
    payload = _payload()
    payload[field] = value
    first, second = _groups()
    request = _request(first, second)
    with pytest.raises(SupportPositiveProposerError):
        _observe(first, second, request, payload, [])


def test_replay_binds_exact_pixels_and_only_durable_journal_is_sealable() -> None:
    first, second = _groups()
    request = _request(first, second)
    artifact = _observe(first, second, request, _payload(), [])
    changed = list(second)
    changed[5] = _png(250)
    with pytest.raises(SupportPositiveProposerError):
        verify_support_positive_proposer_artifact(
            artifact,
            first,
            tuple(changed),
            expected_artifact_digest=artifact.artifact_digest,
        )

    address = "sha256:" + "a" * 64
    journal = SupportPositiveTransportProvenance.create(
        "production_exactly_once_journal",
        journal_summary=SimpleNamespace(
            terminal_status="success",
            manifest_digest=address,
            turn_key=address,
            claim_digest=address,
            result_digest=address,
            outcome_digest=address,
            record_digest=address,
        ),
    )
    assert journal.benchmark_sealable is True
    assert SupportPositiveTransportProvenance.create(
        "production_direct"
    ).benchmark_sealable is False
    assert SupportPositiveTransportProvenance.create(
        "injected_unverified"
    ).benchmark_sealable is False
