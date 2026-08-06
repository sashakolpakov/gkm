from __future__ import annotations

import pytest

from bongard.evidence import (
    Disposition,
    Evidence,
    Provenance,
    SoftSemanticObservation,
    Uncertainty,
)


def provenance(name: str = "vision") -> Provenance:
    return Provenance(
        producer=name,
        version="1",
        method="empirical",
        input_digests=("panel-digest",),
        details=(("model", "headless-codex"),),
    )


def test_four_dispositions_are_distinct_and_not_boolean() -> None:
    origin = provenance()
    values = (
        Evidence.present(3.0, origin, Uncertainty(2.8, 3.2)),
        Evidence.certified_absent(origin, "exhaustive contour certificate"),
        Evidence.indeterminate(origin, "occluded junction"),
        Evidence.error(origin, "RuntimeError", "extractor crashed"),
    )
    assert tuple(item.disposition for item in values) == tuple(Disposition)
    for item in values:
        with pytest.raises(TypeError, match="four dispositions"):
            bool(item)


def test_failed_computation_becomes_error_not_negative_evidence() -> None:
    result = Evidence.present(1, provenance()).map(
        lambda _: (_ for _ in ()).throw(RuntimeError("bad fit"))
    )
    assert result.disposition is Disposition.ERROR
    assert result.error_type == "RuntimeError"
    assert result.disposition is not Disposition.CERTIFIED_ABSENT


def test_certified_absence_requires_a_real_certificate() -> None:
    with pytest.raises(ValueError, match="certificate"):
        Evidence(disposition=Disposition.CERTIFIED_ABSENT, provenance=provenance())
    with pytest.raises(ValueError, match="reason"):
        Evidence(disposition=Disposition.INDETERMINATE, provenance=provenance())


def test_soft_semantics_is_provenance_bearing_observation_not_truth() -> None:
    origin = provenance()
    observation = SoftSemanticObservation(
        phrase="bird-like object",
        description="two lobes and an oblique beak-like protrusion",
        support=Uncertainty(0.71, 0.84, confidence_level=0.95),
        provenance=origin,
        witness_ids=("vlm-call-17",),
    )
    evidence = Evidence.present(observation, origin)
    assert evidence.uncertainty == observation.support
    with pytest.raises(TypeError, match="empirical evidence, not truth"):
        bool(observation)

    with pytest.raises(ValueError, match="provenance differ"):
        Evidence.present(observation, provenance("different-model"))


def test_provenance_composition_hashes_exact_parents() -> None:
    left = provenance("left")
    right = provenance("right")
    combined = Provenance.composed(
        "closed-ir", "1", "and", (left, right), details=(("arity", "2"),)
    )
    assert combined.input_digests == (left.digest(), right.digest())
    assert combined.digest() == combined.digest()
