from __future__ import annotations

from copy import deepcopy

import pytest

from bongard.evidence import Disposition
from bongard.prototype_object_profiles import (
    OBJECT_FEATURE_IDS,
    IntegerInterval,
    ObjectProfile,
)
from bongard.prototype_object_version_space import (
    COUNT_THRESHOLDS,
    PPM_THRESHOLDS,
    ObjectPredicateGrid,
    ObjectSceneEvidence,
    ObjectSceneFeatureValue,
    ObjectStableLineageEvidence,
    ObjectSupportGapKind,
    ObjectSupportVersionSpace,
    ObjectVersionSpaceError,
    build_object_support_version_space,
    cold_verify_object_support_version_space,
    enumerate_object_profile_candidates,
    evaluate_object_profile_candidate,
    object_version_space_algorithm_digest,
)


_LINEAGE_CATALOG_DIGEST = "a" * 64


def _scored(feature_id: str, lower: int, upper: int | None = None) -> ObjectSceneFeatureValue:
    return ObjectSceneFeatureValue(
        feature_id,
        Disposition.PRESENT,
        IntegerInterval(lower, lower if upper is None else upper),
    )


def _absent(feature_id: str) -> ObjectSceneFeatureValue:
    return ObjectSceneFeatureValue(
        feature_id,
        Disposition.CERTIFIED_ABSENT,
        None,
        certificate="stable-lineage feature exclusion",
    )


def _indeterminate(feature_id: str) -> ObjectSceneFeatureValue:
    return ObjectSceneFeatureValue(
        feature_id,
        Disposition.INDETERMINATE,
        None,
        reason="calibrated observer abstained",
    )


def _error(feature_id: str) -> ObjectSceneFeatureValue:
    return ObjectSceneFeatureValue(
        feature_id,
        Disposition.ERROR,
        None,
        reason="observer transport failed",
        error_type="TransportError",
    )


def _lineage(
    lineage_id: str,
    overrides: dict[str, ObjectSceneFeatureValue] | None = None,
) -> ObjectStableLineageEvidence:
    supplied = overrides or {}
    values = tuple(supplied.get(feature_id, _scored(feature_id, 0)) for feature_id in OBJECT_FEATURE_IDS)
    return ObjectStableLineageEvidence.create(lineage_id, values)


def _scene(
    scene_id: str,
    *lineages: ObjectStableLineageEvidence,
    unresolved: bool = False,
) -> ObjectSceneEvidence:
    return ObjectSceneEvidence.create(
        scene_id,
        _LINEAGE_CATALOG_DIGEST,
        lineages,
        unresolved_lineage_possible=unresolved,
    )


def _candidate(
    grid: ObjectPredicateGrid, targets: tuple[tuple[str, int], ...]
) -> ObjectProfile:
    return next(
        profile
        for profile in enumerate_object_profile_candidates(grid)
        if tuple((atom.feature_id, atom.target) for atom in profile.atoms) == targets
    )


def test_grid_is_explicit_finite_source_bound_and_python_canonical() -> None:
    allowed = ("straight_span_count", "open_outline_support_ppm")
    grid = ObjectPredicateGrid.create(allowed)
    assert ObjectPredicateGrid.from_data(grid.to_data()) == grid
    assert grid.allowed_feature_ids == allowed
    assert grid.algorithm_digest == object_version_space_algorithm_digest()
    assert grid.to_data()["count_thresholds"] == list(COUNT_THRESHOLDS)
    assert grid.to_data()["ppm_thresholds"] == list(PPM_THRESHOLDS)
    assert grid.to_data()["python_is_canonical_authority"] is True
    assert grid.to_data()["lean_present"] is False
    assert grid.to_data()["lean_required"] is False
    assert grid.to_data()["lean_removal_changes_decision"] is False
    assert grid.to_data()["no_nomination_means_language_gap"] is True

    # 8 atoms plus 4x4 two-feature conjunctions.  Same-feature redundant
    # conjunctions are excluded by the existing one-feature-per-profile grammar.
    candidates = enumerate_object_profile_candidates(grid)
    assert len(candidates) == 24
    assert [atom.target for atom in candidates[0:4] for atom in atom.atoms] == list(
        COUNT_THRESHOLDS
    )
    assert [atom.target for atom in candidates[4:8] for atom in atom.atoms] == list(
        PPM_THRESHOLDS
    )
    assert all(len({atom.feature_id for atom in item.atoms}) == len(item.atoms) for item in candidates)
    assert all(atom.operator.value == "at_least" for item in candidates for atom in item.atoms)

    with pytest.raises(ObjectVersionSpaceError, match="catalog order"):
        ObjectPredicateGrid.create(tuple(reversed(allowed)))

    polluted = deepcopy(grid.to_data())
    polluted["negation"] = True
    with pytest.raises(ObjectVersionSpaceError, match="fields differ"):
        ObjectPredicateGrid.from_data(polluted)


def test_empty_nomination_is_empty_language_not_full_catalog_fallback() -> None:
    grid = ObjectPredicateGrid.create(())
    assert enumerate_object_profile_candidates(grid) == ()
    version = build_object_support_version_space(
        grid,
        (_scene("positive", _lineage("lp")),),
        (_scene("negative", _lineage("ln")),),
    )
    assert version.candidates == ()
    assert version.survivor_profile_digests == ()
    assert version.gap is not None
    assert version.gap.kind is ObjectSupportGapKind.LANGUAGE_GAP
    assert version.gap.diagnostics == ()


def test_scene_evidence_is_full_ordered_side_free_and_canonical() -> None:
    lineage = _lineage("lineage-0")
    scene = _scene("scene-0", lineage)
    assert ObjectSceneEvidence.from_data(scene.to_data()) == scene
    assert ObjectStableLineageEvidence.from_data(lineage.to_data()) == lineage
    assert "expected_side" not in scene.to_data()
    assert "support_side" not in scene.to_data()
    assert scene.to_data()["support_side_is_visual_evidence"] is False
    assert tuple(item.feature_id for item in lineage.feature_values) == OBJECT_FEATURE_IDS

    official = _scene(
        "ff/ff_nact2_5_0042/1/0.png", _lineage("official-lineage")
    )
    assert ObjectSceneEvidence.from_data(official.to_data()) == official

    with pytest.raises(ObjectVersionSpaceError, match="exhaust"):
        ObjectStableLineageEvidence.create("broken", lineage.feature_values[:-1])
    with pytest.raises(ObjectVersionSpaceError, match="empty lineage inventory"):
        _scene("missing-is-not-negative")

    unresolved_empty = _scene("known-missing", unresolved=True)
    grid = ObjectPredicateGrid.create(("straight_span_count",))
    result = evaluate_object_profile_candidate(
        grid,
        _candidate(grid, (("straight_span_count", 1),)),
        unresolved_empty,
    )
    assert result.disposition is Disposition.INDETERMINATE


def test_feature_four_states_are_closed_and_interval_safe() -> None:
    assert ObjectSceneFeatureValue.from_data(
        _scored("straight_span_count", 1, 2).to_data()
    ) == _scored("straight_span_count", 1, 2)
    assert ObjectSceneFeatureValue.from_data(
        _absent("straight_span_count").to_data()
    ).disposition is Disposition.CERTIFIED_ABSENT
    assert ObjectSceneFeatureValue.from_data(
        _indeterminate("straight_span_count").to_data()
    ).disposition is Disposition.INDETERMINATE
    assert ObjectSceneFeatureValue.from_data(
        _error("straight_span_count").to_data()
    ).disposition is Disposition.ERROR

    with pytest.raises(ObjectVersionSpaceError, match="requires only an integer interval"):
        ObjectSceneFeatureValue(
            "straight_span_count", Disposition.PRESENT, None
        )
    with pytest.raises(ObjectVersionSpaceError, match="requires only a certificate"):
        ObjectSceneFeatureValue(
            "straight_span_count", Disposition.CERTIFIED_ABSENT, None
        )


def test_conjunction_never_maxes_atoms_across_different_lineages() -> None:
    grid = ObjectPredicateGrid.create(
        ("straight_span_count", "endpoint_count")
    )
    candidate = _candidate(
        grid,
        (("straight_span_count", 1), ("endpoint_count", 1)),
    )
    split = _scene(
        "split",
        _lineage(
            "a",
            {
                "straight_span_count": _scored("straight_span_count", 2),
                "endpoint_count": _scored("endpoint_count", 0),
            },
        ),
        _lineage(
            "b",
            {
                "straight_span_count": _scored("straight_span_count", 0),
                "endpoint_count": _scored("endpoint_count", 2),
            },
        ),
    )
    result = evaluate_object_profile_candidate(grid, candidate, split)
    assert result.disposition is Disposition.CERTIFIED_ABSENT
    assert tuple(item.disposition for item in result.lineages) == (
        Disposition.CERTIFIED_ABSENT,
        Disposition.CERTIFIED_ABSENT,
    )
    assert result.to_data()["same_lineage_conjunction"] is True
    assert type(result).from_data(result.to_data()) == result

    unresolved = ObjectSceneEvidence.create(
        "split-unresolved",
        _LINEAGE_CATALOG_DIGEST,
        split.lineages,
        unresolved_lineage_possible=True,
    )
    assert evaluate_object_profile_candidate(
        grid, candidate, unresolved
    ).disposition is Disposition.INDETERMINATE

    together = _scene(
        "together",
        _lineage(
            "one-object",
            {
                "straight_span_count": _scored("straight_span_count", 2),
                "endpoint_count": _scored("endpoint_count", 2),
            },
        ),
    )
    assert evaluate_object_profile_candidate(
        grid, candidate, together
    ).disposition is Disposition.PRESENT


def test_error_dominates_indeterminate_but_absence_and_presence_are_proofs() -> None:
    grid = ObjectPredicateGrid.create(("straight_span_count",))
    candidate = _candidate(grid, (("straight_span_count", 1),))
    scene = _scene(
        "mixed-failure",
        _lineage("a", {"straight_span_count": _error("straight_span_count")}),
        _lineage(
            "b",
            {"straight_span_count": _indeterminate("straight_span_count")},
        ),
    )
    assert evaluate_object_profile_candidate(
        grid, candidate, scene
    ).disposition is Disposition.ERROR

    proved = _scene(
        "proved",
        *scene.lineages,
        _lineage(
            "c", {"straight_span_count": _scored("straight_span_count", 1)}
        ),
    )
    assert evaluate_object_profile_candidate(
        grid, candidate, proved
    ).disposition is Disposition.PRESENT


def test_build_retains_exact_support_consistent_candidates_and_cold_replays() -> None:
    grid = ObjectPredicateGrid.create(("straight_span_count",))
    positive = _scene(
        "z-positive",
        _lineage(
            "positive-object",
            {"straight_span_count": _scored("straight_span_count", 3)},
        ),
    )
    negative = _scene(
        "a-negative",
        _lineage(
            "negative-object",
            {"straight_span_count": _scored("straight_span_count", 0)},
        ),
    )
    version = build_object_support_version_space(grid, (positive,), (negative,))
    assert ObjectSupportVersionSpace.from_data(version.to_data()) == version
    assert version.support_scene_ids == ("z-positive", "a-negative")
    assert [item.value for item in version.support_sides] == ["positive", "negative"]
    survivors = tuple(version.survivor(item) for item in version.survivor_profile_digests)
    assert tuple(item.atoms[0].target for item in survivors) == (1, 2, 3)
    assert version.gap is None
    assert cold_verify_object_support_version_space(
        version, grid, (positive,), (negative,)
    ) == version

    changed = _scene(
        positive.scene_id,
        _lineage(
            "positive-object",
            {"straight_span_count": _scored("straight_span_count", 2)},
        ),
    )
    with pytest.raises(ObjectVersionSpaceError, match="cold version-space replay"):
        cold_verify_object_support_version_space(
            version, grid, (changed,), (negative,)
        )


def test_gap_is_witness_only_when_uncertainty_is_needed_for_a_candidate() -> None:
    grid = ObjectPredicateGrid.create(("straight_span_count",))
    negative_absent = _scene("negative", _lineage("n"))
    uncertain_positive = _scene(
        "positive",
        _lineage(
            "p",
            {"straight_span_count": _scored("straight_span_count", 0, 4)},
        ),
    )
    witness = build_object_support_version_space(
        grid, (uncertain_positive,), (negative_absent,)
    )
    assert witness.gap is not None
    assert witness.gap.kind is ObjectSupportGapKind.WITNESS_GAP
    assert all(
        item.indeterminate_scene_ids == ("positive",)
        and not item.definite_counterexample_scene_ids
        for item in witness.gap.diagnostics
    )

    # Every candidate is already disproved on the positive.  Failures on the
    # negative are therefore not needed to decide that this finite language is
    # empty, so this is a language gap rather than a witness gap.
    definite_positive = _scene("positive", _lineage("p"))
    error_negative = _scene(
        "negative",
        _lineage(
            "n", {"straight_span_count": _error("straight_span_count")}
        ),
    )
    language = build_object_support_version_space(
        grid, (definite_positive,), (error_negative,)
    )
    assert language.gap is not None
    assert language.gap.kind is ObjectSupportGapKind.LANGUAGE_GAP
    assert all(item.error_scene_ids == ("negative",) for item in language.gap.diagnostics)
    assert all(item.definite_counterexample_scene_ids == ("positive",) for item in language.gap.diagnostics)


def test_serialized_tampering_cannot_change_a_decision() -> None:
    grid = ObjectPredicateGrid.create(("straight_span_count",))
    version = build_object_support_version_space(
        grid,
        (
            _scene(
                "positive",
                _lineage(
                    "p", {"straight_span_count": _scored("straight_span_count", 2)}
                ),
            ),
        ),
        (_scene("negative", _lineage("n")),),
    )
    polluted = deepcopy(version.to_data())
    polluted["rows"][0][0] = Disposition.CERTIFIED_ABSENT.value
    with pytest.raises(ObjectVersionSpaceError):
        ObjectSupportVersionSpace.from_data(polluted)
