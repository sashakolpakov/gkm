"""Offline tests for the closed ordinal-rubric Python version space."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
from io import BytesIO
import re
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw
import pytest

from bongard.evidence import Disposition
from bongard.object_bongard_rubric_observer import (
    ObjectBongardRubricObserverArtifact,
    ObjectBongardRubricSpec,
    OrdinalLevelInterval,
    RubricObservationState,
    RubricScope,
    RubricScopeObservation,
    object_bongard_catalog_contrast_rubric,
    object_bongard_rubric_ordinal_scale_digest,
    observe_object_bongard_rubric,
    verify_object_bongard_rubric_observer_artifact,
)
from bongard.object_bongard_rubric_version_space import (
    ObjectBongardRubricCandidate,
    ObjectBongardRubricSupportVersionSpace,
    ObjectBongardRubricVersionSpaceError,
    RUBRIC_THRESHOLDS,
    RubricPredicateOperator,
    RubricSupportGapKind,
    build_object_bongard_rubric_support_version_space,
    cold_verify_object_bongard_rubric_support_version_space,
    enumerate_object_bongard_rubric_candidates,
    evaluate_object_bongard_rubric_candidate,
)
import bongard.object_bongard_rubric_version_space as version_module
from bongard.prototype_object_hypotheses import extract_object_hypothesis_packet
from bongard.prototype_object_lineages import extract_object_lineage_packet
from bongard.tests.no_tools_fixture import (
    canonical_codex_receipt,
    canonical_no_tools_runtime,
)
from bongard.transport import (
    DEFAULT_CODEX_MODEL,
    CloudPolicyCacheSnapshot,
    CodexStructuredResult,
)


MODEL = DEFAULT_CODEX_MODEL
EFFORT = "medium"
LAUNCHER_DIGEST = "b" * 64


def _raw(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _spec() -> ObjectBongardRubricSpec:
    return ObjectBongardRubricSpec.create(
        _raw("semantic-artifact"),
        object_bongard_catalog_contrast_rubric(
            "oblique_span_support_ppm", "bird_like_support_ppm"
        ),
        ("oblique_span_support_ppm", "bird_like_support_ppm"),
    )


@lru_cache(maxsize=1)
def _runtime():
    return canonical_no_tools_runtime(LAUNCHER_DIGEST)


def _png(image_index: int) -> bytes:
    """Two stable objects with byte-distinct, geometry-preserving variants."""

    image = Image.new("RGB", (96, 64), "white")
    draw = ImageDraw.Draw(image)
    offset = image_index % 12
    draw.polygon(
        ((8 + offset, 34), (18 + offset, 12), (28 + offset, 34)),
        fill="black",
    )
    right = 72 - offset
    draw.rectangle((right, 16, right + 18, 34), fill="black")
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


class _RubricTransport:
    def __init__(
        self,
        object_interval: tuple[int, int],
        scene_interval: tuple[int, int],
        *,
        fail: bool = False,
    ) -> None:
        self.object_interval = object_interval
        self.scene_interval = scene_interval
        self.fail = fail
        self.calls = 0

    def __call__(
        self,
        prompt: str,
        paths: Sequence[str],
        names: Sequence[str],
        schema: Mapping[str, Any],
        **kwargs: object,
    ) -> CodexStructuredResult:
        self.calls += 1
        if self.fail:
            raise OSError("synthetic observer transport failure")
        slot_ids = tuple(re.findall(r"^- (slot-[0-9]{8}):", prompt, re.MULTILINE))
        lower, upper = self.object_interval
        scene_lower, scene_upper = self.scene_interval
        payload = {
            "scene": {"lower": scene_lower, "upper": scene_upper},
            "slots": [
                {"slot_id": slot_id, "lower": lower, "upper": upper}
                for slot_id in slot_ids
            ],
        }
        receipt = canonical_codex_receipt(
            prompt,
            paths,
            schema,
            payload,
            launcher_digest=LAUNCHER_DIGEST,
            reasoning_effort=EFFORT,
            model=MODEL,
            names=names,
        )
        return CodexStructuredResult(payload, receipt)


@lru_cache(maxsize=None)
def _observed_artifact(
    panel_id: str,
    *,
    image_index: int,
    object_interval: tuple[int, int],
    scene_interval: tuple[int, int],
    rubric_spec: ObjectBongardRubricSpec | None = None,
    transport_error: bool = False,
) -> ObjectBongardRubricObserverArtifact:
    """Build and exact-PNG-verify one real public observer artifact."""

    spec = _spec() if rubric_spec is None else rubric_spec
    png_bytes = _png(image_index)
    panel_digest = hashlib.sha256(png_bytes).hexdigest()
    hypothesis_packet = extract_object_hypothesis_packet(png_bytes)
    lineage_packet = extract_object_lineage_packet(png_bytes, hypothesis_packet)
    model_catalog, no_tools = _runtime()
    transport = _RubricTransport(
        object_interval, scene_interval, fail=transport_error
    )
    artifact = observe_object_bongard_rubric(
        png_bytes,
        panel_id=panel_id,
        rubric_spec=spec,
        hypothesis_packet=hypothesis_packet,
        lineage_packet=lineage_packet,
        expected_scene_sha256=panel_digest,
        expected_rubric_spec_digest=spec.spec_digest,
        expected_hypothesis_packet_digest=hypothesis_packet.digest(),
        expected_lineage_packet_digest=lineage_packet.digest(),
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        cloud_policy_cache_snapshot=CloudPolicyCacheSnapshot(None),
        model_catalog_snapshot=model_catalog,
        no_tools_attestation=no_tools,
        transport=transport,
    )
    assert transport.calls == len(hypothesis_packet.atlas_sheets)
    return verify_object_bongard_rubric_observer_artifact(
        artifact,
        png_bytes,
        panel_id=panel_id,
        rubric_spec=spec,
        hypothesis_packet=hypothesis_packet,
        lineage_packet=lineage_packet,
        expected_artifact_digest=artifact.artifact_digest,
        expected_runtime_identity_digest=artifact.runtime_identity_digest,
    )


@lru_cache(maxsize=1)
def _base_support() -> tuple[
    ObjectBongardRubricSpec,
    tuple[ObjectBongardRubricObserverArtifact, ...],
    tuple[ObjectBongardRubricObserverArtifact, ...],
]:
    """Support with exactly OBJECT >= 3 as the sole survivor."""

    spec = _spec()
    positives = tuple(
        _observed_artifact(
            f"bd/rubric_fixture/1/{index}.png",
            image_index=index,
            object_interval=(3, 3),
            scene_interval=(0, 0),
            rubric_spec=spec,
        )
        for index in range(6)
    )
    negatives = tuple(
        _observed_artifact(
            f"bd/rubric_fixture/0/{index}.png",
            image_index=index + 6,
            object_interval=(1, 1),
            scene_interval=(4, 4),
            rubric_spec=spec,
        )
        for index in range(6)
    )
    return spec, positives, negatives


def _observation(
    spec: ObjectBongardRubricSpec,
    observation_id: str,
    scope: RubricScope,
    *,
    lower: int | None = None,
    upper: int | None = None,
    state: RubricObservationState = RubricObservationState.SCORED,
) -> RubricScopeObservation:
    interval = (
        None
        if lower is None
        else OrdinalLevelInterval(lower, lower if upper is None else upper)
    )
    return RubricScopeObservation.create(
        rubric_spec_digest=spec.spec_digest,
        scope=scope,
        observation_id=observation_id,
        member_hypothesis_ids=(f"member-{observation_id}",),
        geometry_digest=_raw(f"geometry:{observation_id}"),
        state=state,
        interval=interval,
        reason=(
            None
            if state is RubricObservationState.SCORED
            else "observer_abstained"
        ),
        error_type=(
            "FixtureTransportError"
            if state is RubricObservationState.ERROR
            else None
        ),
    )


def _candidate(
    spec: ObjectBongardRubricSpec, scope: RubricScope, threshold: int
) -> ObjectBongardRubricCandidate:
    return next(
        item
        for item in enumerate_object_bongard_rubric_candidates(spec)
        if item.scope is scope and item.threshold == threshold
    )


def test_complete_two_candidate_inventory_is_positive_closed_and_scale_bound() -> None:
    spec = _spec()
    candidates = enumerate_object_bongard_rubric_candidates(spec)

    assert len(candidates) == 2
    assert [(item.scope, item.threshold) for item in candidates] == [
        *((RubricScope.OBJECT, value) for value in RUBRIC_THRESHOLDS),
        *((RubricScope.SCENE, value) for value in RUBRIC_THRESHOLDS),
    ]
    assert all(
        item.operator is RubricPredicateOperator.AT_LEAST for item in candidates
    )
    assert [item.formula for item in candidates] == [
        *(
            f"EXISTS OBJECT witness with rubric_level >= {value}"
            for value in RUBRIC_THRESHOLDS
        ),
        *(f"SCENE rubric_level >= {value}" for value in RUBRIC_THRESHOLDS),
    ]
    assert all(
        ObjectBongardRubricCandidate.from_data(item.to_data()) == item
        for item in candidates
    )
    assert all(
        not ({"negation", "not", "polarity", "disjunction", "code"} & set(item.to_data()))
        for item in candidates
    )
    assert version_module._language_data()["ordinal_scale_digest"] == (
        object_bongard_rubric_ordinal_scale_digest()
    )
    assert version_module._language_data()["deadband_levels"] == [2]
    assert version_module._language_data()["tie_can_certify_absence"] is False


def test_real_observation_interval_is_closed_and_canonical() -> None:
    spec = _spec()
    observation = _observation(
        spec, "object-0000", RubricScope.OBJECT, lower=2, upper=3
    )
    assert RubricScopeObservation.from_data(observation.to_data()) == observation
    assert observation.interval == OrdinalLevelInterval(2, 3)
    with pytest.raises(ValueError, match="0..4"):
        OrdinalLevelInterval(-1, 2)


def test_public_observer_artifact_separates_object_and_scene_interval_logic() -> None:
    spec = _spec()
    artifact = _observed_artifact(
        "bd/rubric_fixture/1/0.png",
        image_index=0,
        object_interval=(2, 3),
        scene_interval=(0, 0),
        rubric_spec=spec,
    )

    assert ObjectBongardRubricObserverArtifact.from_data(artifact.to_data()) == artifact
    assert artifact.object_observations
    assert artifact.canonical_scene_observation is not None
    assert evaluate_object_bongard_rubric_candidate(
        _candidate(spec, RubricScope.OBJECT, 3), artifact
    ).disposition is Disposition.INDETERMINATE
    assert evaluate_object_bongard_rubric_candidate(
        _candidate(spec, RubricScope.SCENE, 3), artifact
    ).disposition is Disposition.CERTIFIED_ABSENT


def test_exact_six_plus_six_support_retains_only_the_verified_threshold() -> None:
    spec, positives, negatives = _base_support()
    version = build_object_bongard_rubric_support_version_space(
        spec, positives, negatives
    )

    assert len(version.candidates) == 2
    assert len(version.support_panel_ids) == 12
    assert len(set(version.support_panel_ids)) == 12
    assert len(version.survivor_candidate_digests) == 1
    survivor = version.survivor(version.survivor_candidate_digests[0])
    assert (survivor.scope, survivor.threshold) == (RubricScope.OBJECT, 3)
    assert version.gap is None
    assert ObjectBongardRubricSupportVersionSpace.from_data(version.to_data()) == (
        version
    )
    assert cold_verify_object_bongard_rubric_support_version_space(
        version, spec, positives, negatives
    ) == version


def test_empty_support_space_distinguishes_language_and_witness_gaps() -> None:
    spec, positives, negatives = _base_support()

    definite_positive = _observed_artifact(
        positives[0].panel_id,
        image_index=0,
        object_interval=(0, 0),
        scene_interval=(0, 0),
        rubric_spec=spec,
    )
    language = build_object_bongard_rubric_support_version_space(
        spec, (definite_positive, *positives[1:]), negatives
    )
    assert language.survivor_candidate_digests == ()
    assert language.gap is not None
    assert language.gap.kind is RubricSupportGapKind.LANGUAGE_GAP
    assert all(
        item.definite_counterexample_panel_ids
        for item in language.gap.diagnostics
    )

    uncertain_negative = _observed_artifact(
        negatives[0].panel_id,
        image_index=6,
        object_interval=(2, 3),
        scene_interval=(4, 4),
        rubric_spec=spec,
    )
    witness = build_object_bongard_rubric_support_version_space(
        spec, positives, (uncertain_negative, *negatives[1:])
    )
    assert witness.survivor_candidate_digests == ()
    assert witness.gap is not None
    assert witness.gap.kind is RubricSupportGapKind.WITNESS_GAP
    threshold_three = _candidate(spec, RubricScope.OBJECT, 3)
    diagnostic = next(
        item
        for item in witness.gap.diagnostics
        if item.candidate_digest == threshold_three.candidate_digest
    )
    assert diagnostic.definite_counterexample_panel_ids == ()
    assert diagnostic.indeterminate_panel_ids == (uncertain_negative.panel_id,)


def test_missing_errors_and_unresolved_possible_objects_never_become_absence() -> None:
    spec = _spec()
    stable_low = _observation(
        spec, "stable-low", RubricScope.OBJECT, lower=0, upper=0
    )
    stable_high = _observation(
        spec, "stable-high", RubricScope.OBJECT, lower=1, upper=1
    )
    unresolved_low = _observation(
        spec, "unresolved-low", RubricScope.OBJECT, lower=0, upper=0
    )
    unresolved_crossing = _observation(
        spec, "unresolved-crossing", RubricScope.OBJECT, lower=0, upper=2
    )
    unresolved_error = _observation(
        spec,
        "unresolved-error",
        RubricScope.OBJECT,
        state=RubricObservationState.ERROR,
    )

    assert version_module._evaluate_object_scope(
        (stable_low,), (unresolved_low,), 3
    ) is Disposition.CERTIFIED_ABSENT
    assert version_module._evaluate_object_scope(
        (stable_low,), (unresolved_crossing,), 3
    ) is Disposition.INDETERMINATE
    assert version_module._evaluate_object_scope(
        (stable_low,), (unresolved_error,), 3
    ) is Disposition.ERROR
    assert version_module._evaluate_object_scope(
        (stable_high,), (unresolved_error,), 3
    ) is Disposition.ERROR
    stable_target = _observation(
        spec, "stable-target", RubricScope.OBJECT, lower=3, upper=3
    )
    assert version_module._evaluate_object_scope(
        (stable_target,), (unresolved_error,), 3
    ) is Disposition.PRESENT
    stable_tie = _observation(
        spec, "stable-tie", RubricScope.OBJECT, lower=2, upper=2
    )
    assert version_module._evaluate_object_scope(
        (stable_tie,), (), 3
    ) is Disposition.INDETERMINATE
    assert version_module._evaluate_object_scope((), (), 3) is (
        Disposition.INDETERMINATE
    )
    assert version_module._evaluate_scene_scope(None, 3) is (
        Disposition.INDETERMINATE
    )
    scene_tie = _observation(
        spec, "scene-tie", RubricScope.SCENE, lower=2, upper=2
    )
    assert version_module._evaluate_scene_scope(scene_tie, 3) is (
        Disposition.INDETERMINATE
    )

    failed = _observed_artifact(
        "bd/rubric_error/0/0.png",
        image_index=0,
        object_interval=(0, 0),
        scene_interval=(0, 0),
        rubric_spec=spec,
        transport_error=True,
    )
    assert evaluate_object_bongard_rubric_candidate(
        _candidate(spec, RubricScope.OBJECT, 3), failed
    ).disposition is Disposition.ERROR
    assert evaluate_object_bongard_rubric_candidate(
        _candidate(spec, RubricScope.SCENE, 3), failed
    ).disposition is Disposition.ERROR


def test_serialization_tamper_and_wrong_support_fail_cold_replay() -> None:
    spec, positives, negatives = _base_support()
    version = build_object_bongard_rubric_support_version_space(
        spec, positives, negatives
    )

    polluted_candidate = deepcopy(version.candidates[0].to_data())
    polluted_candidate["negation"] = True
    with pytest.raises(ObjectBongardRubricVersionSpaceError, match="fields differ"):
        ObjectBongardRubricCandidate.from_data(polluted_candidate)

    polluted_version = deepcopy(version.to_data())
    polluted_version["rows"][0][0] = Disposition.ERROR.value
    with pytest.raises(ObjectBongardRubricVersionSpaceError):
        ObjectBongardRubricSupportVersionSpace.from_data(polluted_version)

    replacement = _observed_artifact(
        positives[0].panel_id,
        image_index=0,
        object_interval=(0, 0),
        scene_interval=(0, 0),
        rubric_spec=spec,
    )
    with pytest.raises(ObjectBongardRubricVersionSpaceError, match="cold.*differs"):
        cold_verify_object_bongard_rubric_support_version_space(
            version, spec, (replacement, *positives[1:]), negatives
        )
