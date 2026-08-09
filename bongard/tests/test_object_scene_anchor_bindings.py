from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from bongard.canonical import canonical_digest
from bongard.evidence import Disposition
from bongard.object_scene_anchor_catalog import ObjectSceneAnchorDecisionManifest
from bongard.object_scene_anchor_bindings import (
    ObjectSceneAnchorBindingCatalog,
    ObjectSceneAnchorBindingError,
    ObjectSceneAnchorBindingEvaluation,
    ObjectSceneAnchorBindingSpec,
    ObjectSceneAnchorCatalogEvaluation,
    ObjectSceneAnchorWitnessCell,
    ObjectSceneAnchorWitnessSpec,
    ObjectSceneResolvedAnchorBinding,
    build_object_scene_anchor_binding_catalog,
    compile_object_scene_anchor_binding,
    compile_object_scene_anchor_catalog,
    merge_repeated_object_scene_anchor_catalog_evaluations,
)
from bongard.object_scene_anchor_salience import (
    AnchorSalienceLimits,
    extract_object_scene_anchor_salience,
)


def _line() -> np.ndarray:
    mask = np.zeros((61, 61), dtype=bool)
    mask[30, 10:51] = True
    return mask


def _plus() -> np.ndarray:
    mask = _line()
    mask[10:51, 30] = True
    return mask


def _manifest(
    mask: np.ndarray,
    object_id: str,
    limits: AnchorSalienceLimits | None = None,
) -> tuple[object, ObjectSceneAnchorDecisionManifest]:
    salience = extract_object_scene_anchor_salience(mask, object_id, limits)
    return salience, ObjectSceneAnchorDecisionManifest.from_salience(salience)


def _witnesses(count: int = 1) -> tuple[ObjectSceneAnchorWitnessSpec, ...]:
    return tuple(
        ObjectSceneAnchorWitnessSpec(
            f"witness_{index:02d}", canonical_digest({"witness": index})
        )
        for index in range(count)
    )


def _binding_evaluation(
    binding: object,
    witnesses: tuple[ObjectSceneAnchorWitnessSpec, ...],
    states: tuple[Disposition, ...],
) -> object:
    cells = tuple(
        ObjectSceneAnchorWitnessCell.create(binding, witness, state)
        for witness, state in zip(witnesses, states, strict=True)
    )
    return compile_object_scene_anchor_binding(binding, witnesses, cells)


def _catalog_evaluation_with_one_winner(
    catalog: object,
    witnesses: tuple[ObjectSceneAnchorWitnessSpec, ...],
    winner_index: int,
) -> object:
    rows = tuple(
        _binding_evaluation(
            binding,
            witnesses,
            (
                Disposition.PRESENT
                if index == winner_index
                else Disposition.CERTIFIED_ABSENT,
            ),
        )
        for index, binding in enumerate(catalog.bindings)
    )
    return compile_object_scene_anchor_catalog(catalog, rows)


def _all_keys(value: object) -> tuple[str, ...]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            keys.append(str(key))
            keys.extend(_all_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.extend(_all_keys(item))
    return tuple(keys)


def test_binding_spec_is_closed_and_frame_interval_is_digest_bound() -> None:
    entity = ObjectSceneAnchorBindingSpec.entity()
    part = ObjectSceneAnchorBindingSpec.part()
    frame = ObjectSceneAnchorBindingSpec.frame(3, 4)

    assert entity.incident_part_count is None
    assert part.incident_part_count is None
    assert frame.incident_part_count == (3, 4)
    assert ObjectSceneAnchorBindingSpec.from_data(frame.to_data()) == frame

    with pytest.raises(ObjectSceneAnchorBindingError, match="only frame"):
        ObjectSceneAnchorBindingSpec.create("entity", (3, 4))
    with pytest.raises(ObjectSceneAnchorBindingError, match="interval"):
        ObjectSceneAnchorBindingSpec.frame(2, 4)
    with pytest.raises(ObjectSceneAnchorBindingError, match="interval"):
        ObjectSceneAnchorBindingSpec.frame(4, 3)
    with pytest.raises(ObjectSceneAnchorBindingError, match="kind"):
        ObjectSceneAnchorBindingSpec.create("join")


def test_binding_accepts_one_to_four_witnesses_only() -> None:
    _, manifest = _manifest(_plus(), "object_0000")
    catalog = build_object_scene_anchor_binding_catalog(
        manifest,
        ObjectSceneAnchorBindingSpec.entity(),
        expected_object_id="object_0000",
    )
    binding = catalog.bindings[0]

    four = _witnesses(4)
    cells = tuple(
        ObjectSceneAnchorWitnessCell.create(
            binding, witness, Disposition.PRESENT
        )
        for witness in four
    )
    assert compile_object_scene_anchor_binding(
        binding, four, cells
    ).disposition is Disposition.PRESENT

    with pytest.raises(ObjectSceneAnchorBindingError, match="catalog"):
        compile_object_scene_anchor_binding(binding, (), ())
    with pytest.raises(ObjectSceneAnchorBindingError, match="catalog"):
        compile_object_scene_anchor_binding(binding, _witnesses(5), ())


def test_v8_style_no_frame_target_is_certified_absent_before_vision() -> None:
    false_target, false_manifest = _manifest(_line(), "object_0000")
    true_target, true_manifest = _manifest(_plus(), "object_0001")
    spec = ObjectSceneAnchorBindingSpec.frame(3, 8)

    false_catalog = build_object_scene_anchor_binding_catalog(
        false_manifest,
        spec,
        expected_object_id="object_0000",
    )
    true_catalog = build_object_scene_anchor_binding_catalog(
        true_manifest,
        spec,
        expected_object_id="object_0001",
    )

    assert false_target.selected_graph is not None
    assert len(false_target.selected_graph.cyclic_frames) == 0
    assert false_catalog.hard_disposition is Disposition.CERTIFIED_ABSENT
    assert false_catalog.reason == "complete_empty"
    assert false_catalog.bindings == ()
    assert compile_object_scene_anchor_catalog(false_catalog).disposition is (
        Disposition.CERTIFIED_ABSENT
    )

    assert true_target.selected_graph is not None
    assert len(true_target.selected_graph.cyclic_frames) == 1
    assert true_catalog.hard_disposition is Disposition.PRESENT
    assert len(true_catalog.bindings) == 1
    binding = true_catalog.bindings[0]
    assert binding.binding_alias == "binding_000"
    assert binding.anchor_kind == "frame"
    assert binding.anchor_id == "frame-00000000"
    assert binding.spec_digest == spec.spec_digest
    assert binding.selected_graph_digest == true_target.selected_graph.artifact_digest


def test_part_catalog_includes_compact_selected_anchors() -> None:
    mask = np.zeros((9, 9), dtype=bool)
    mask[4, 4] = True
    salience, manifest = _manifest(mask, "object_0000")
    catalog = build_object_scene_anchor_binding_catalog(
        manifest,
        ObjectSceneAnchorBindingSpec.part(),
        expected_object_id="object_0000",
    )

    assert salience.selected_graph is not None
    assert salience.selected_graph.parts == ()
    assert len(salience.selected_graph.compact_components) == 1
    assert catalog.hard_disposition is Disposition.PRESENT
    assert [item.anchor_id for item in catalog.bindings] == ["compact-00000000"]
    assert catalog.bindings[0].anchor_digest == (
        salience.selected_graph.compact_components[0].digest()
    )


def test_binding_identity_uses_decision_projection_not_raw_or_audit_identity() -> None:
    salience, manifest = _manifest(_plus(), "object_0000")
    catalog = build_object_scene_anchor_binding_catalog(
        manifest,
        ObjectSceneAnchorBindingSpec.part(),
        expected_object_id="object_0000",
    )
    replayed = build_object_scene_anchor_binding_catalog(
        ObjectSceneAnchorDecisionManifest.from_data(manifest.to_data()),
        ObjectSceneAnchorBindingSpec.part(),
        expected_object_id="object_0000",
    )
    data = catalog.to_data()
    keys = _all_keys(data)

    assert replayed == catalog
    assert catalog.selected_graph_digest == salience.selected_graph.artifact_digest
    assert all("entry" not in key for key in keys)
    assert all("salience_artifact" not in key for key in keys)
    assert all("raw_graph" not in key for key in keys)
    assert all("audit" not in key for key in keys)
    assert all("lean" not in key.casefold() for key in keys)
    assert data["python_is_canonical_authority"] is True
    assert [item.binding_id for item in catalog.bindings] == [
        f"binding_{index:03d}" for index in range(4)
    ]


def test_binding_artifacts_round_trip_and_reject_resealed_or_extra_fields() -> None:
    _, manifest = _manifest(_plus(), "object_0000")
    spec = ObjectSceneAnchorBindingSpec.part()
    catalog = build_object_scene_anchor_binding_catalog(
        manifest, spec, expected_object_id="object_0000"
    )
    witness_specs = _witnesses(2)
    binding = catalog.bindings[0]
    witness_cell = ObjectSceneAnchorWitnessCell.create(
        binding, witness_specs[0], Disposition.PRESENT
    )
    binding_evaluation = _binding_evaluation(
        binding,
        witness_specs,
        (Disposition.PRESENT, Disposition.PRESENT),
    )
    catalog_evaluation = compile_object_scene_anchor_catalog(
        catalog,
        tuple(
            _binding_evaluation(
                item,
                witness_specs,
                (Disposition.PRESENT, Disposition.PRESENT),
            )
            for item in catalog.bindings
        ),
    )

    rows = (
        (ObjectSceneAnchorBindingSpec, spec, "spec_digest"),
        (ObjectSceneResolvedAnchorBinding, binding, "binding_digest"),
        (ObjectSceneAnchorBindingCatalog, catalog, "catalog_digest"),
        (ObjectSceneAnchorWitnessCell, witness_cell, "cell_digest"),
        (
            ObjectSceneAnchorBindingEvaluation,
            binding_evaluation,
            "evaluation_digest",
        ),
        (
            ObjectSceneAnchorCatalogEvaluation,
            catalog_evaluation,
            "evaluation_digest",
        ),
    )
    for record_type, record, digest_field in rows:
        assert record_type.from_data(record.to_data()) == record
        tampered = deepcopy(record.to_data())
        tampered[digest_field] = "0" * 64
        with pytest.raises(ObjectSceneAnchorBindingError):
            record_type.from_data(tampered)

    assert (
        ObjectSceneAnchorWitnessSpec.from_data(witness_specs[0].to_data())
        == witness_specs[0]
    )
    extra = {**witness_specs[0].to_data(), "unexpected": True}
    with pytest.raises(ObjectSceneAnchorBindingError, match="fields"):
        ObjectSceneAnchorWitnessSpec.from_data(extra)


def test_cap_is_indeterminate_and_extractor_error_is_error() -> None:
    capped, capped_manifest = _manifest(
        _plus(), "object_0000", AnchorSalienceLimits(max_frames=0)
    )
    failed, failed_manifest = _manifest(
        np.zeros((8, 8), dtype=bool), "object_0001"
    )
    spec = ObjectSceneAnchorBindingSpec.frame()

    capped_catalog = build_object_scene_anchor_binding_catalog(
        capped_manifest,
        spec,
        expected_object_id="object_0000",
    )
    failed_catalog = build_object_scene_anchor_binding_catalog(
        failed_manifest,
        spec,
        expected_object_id="object_0001",
    )

    assert capped.status.state == "indeterminate"
    assert capped_catalog.hard_disposition is Disposition.INDETERMINATE
    assert capped_catalog.bindings == ()
    assert compile_object_scene_anchor_catalog(capped_catalog).disposition is (
        Disposition.INDETERMINATE
    )

    assert failed.status.state == "error"
    assert failed_catalog.hard_disposition is Disposition.ERROR
    assert failed_catalog.bindings == ()
    assert compile_object_scene_anchor_catalog(failed_catalog).disposition is (
        Disposition.ERROR
    )


def test_tampered_or_foreign_salience_is_error_never_absence() -> None:
    _, valid = _manifest(_plus(), "object_0000")
    forged = deepcopy(valid)
    object.__setattr__(forged, "manifest_digest", "0" * 64)
    spec = ObjectSceneAnchorBindingSpec.frame()

    tampered = build_object_scene_anchor_binding_catalog(
        forged,
        spec,
        expected_object_id="object_0000",
    )
    foreign = build_object_scene_anchor_binding_catalog(
        valid,
        spec,
        expected_object_id="object_0001",
    )

    assert tampered.hard_disposition is Disposition.ERROR
    assert tampered.reason == "salience_verification_error"
    assert tampered.bindings == ()
    assert foreign.hard_disposition is Disposition.ERROR
    assert foreign.reason == "foreign_object"
    assert foreign.bindings == ()


def test_catalog_requires_every_binding_and_every_witness_on_same_binding() -> None:
    _, manifest = _manifest(_plus(), "object_0000")
    catalog = build_object_scene_anchor_binding_catalog(
        manifest,
        ObjectSceneAnchorBindingSpec.part(),
        expected_object_id="object_0000",
    )
    witnesses = _witnesses(2)
    first, second = catalog.bindings[:2]

    wrong_binding_cell = ObjectSceneAnchorWitnessCell.create(
        second, witnesses[0], Disposition.PRESENT
    )
    right_binding_cell = ObjectSceneAnchorWitnessCell.create(
        first, witnesses[1], Disposition.PRESENT
    )
    pooled = compile_object_scene_anchor_binding(
        first, witnesses, (wrong_binding_cell, right_binding_cell)
    )
    assert pooled.structurally_valid is False
    assert pooled.disposition is Disposition.ERROR

    valid_rows = tuple(
        _binding_evaluation(
            binding,
            witnesses,
            (Disposition.CERTIFIED_ABSENT, Disposition.PRESENT),
        )
        for binding in catalog.bindings
    )
    missing = compile_object_scene_anchor_catalog(catalog, valid_rows[:-1])
    exhaustive = compile_object_scene_anchor_catalog(catalog, valid_rows)
    assert missing.structurally_valid is False
    assert missing.disposition is Disposition.ERROR
    assert exhaustive.structurally_valid is True
    assert exhaustive.disposition is Disposition.CERTIFIED_ABSENT


def test_error_dominates_existential_binding_aggregation() -> None:
    _, manifest = _manifest(_plus(), "object_0000")
    catalog = build_object_scene_anchor_binding_catalog(
        manifest,
        ObjectSceneAnchorBindingSpec.part(),
        expected_object_id="object_0000",
    )
    witnesses = _witnesses()
    rows = []
    for index, binding in enumerate(catalog.bindings):
        state = (
            Disposition.PRESENT
            if index == 0
            else Disposition.ERROR
            if index == 1
            else Disposition.CERTIFIED_ABSENT
        )
        rows.append(_binding_evaluation(binding, witnesses, (state,)))
    result = compile_object_scene_anchor_catalog(catalog, rows)
    assert result.disposition is Disposition.ERROR


def test_repeated_passes_cannot_pool_different_present_bindings() -> None:
    _, manifest = _manifest(_plus(), "object_0000")
    catalog = build_object_scene_anchor_binding_catalog(
        manifest,
        ObjectSceneAnchorBindingSpec.part(),
        expected_object_id="object_0000",
    )
    witnesses = _witnesses()

    pass_a = _catalog_evaluation_with_one_winner(catalog, witnesses, 0)
    pass_b = _catalog_evaluation_with_one_winner(catalog, witnesses, 1)
    assert pass_a.disposition is Disposition.PRESENT
    assert pass_b.disposition is Disposition.PRESENT

    merged = merge_repeated_object_scene_anchor_catalog_evaluations(
        catalog, witnesses, pass_a, pass_b
    )
    assert merged.disposition is Disposition.INDETERMINATE
    assert merged.binding_evaluations[0].disposition is Disposition.INDETERMINATE
    assert merged.binding_evaluations[1].disposition is Disposition.INDETERMINATE

    same_binding = merge_repeated_object_scene_anchor_catalog_evaluations(
        catalog, witnesses, pass_a, pass_a
    )
    assert same_binding.disposition is Disposition.PRESENT


def test_foreign_catalog_evaluation_merges_to_error() -> None:
    spec = ObjectSceneAnchorBindingSpec.part()
    witnesses = _witnesses()
    _, first_manifest = _manifest(_plus(), "object_0000")
    _, second_manifest = _manifest(_plus(), "object_0001")
    first_catalog = build_object_scene_anchor_binding_catalog(
        first_manifest,
        spec,
        expected_object_id="object_0000",
    )
    second_catalog = build_object_scene_anchor_binding_catalog(
        second_manifest,
        spec,
        expected_object_id="object_0001",
    )
    first_eval = _catalog_evaluation_with_one_winner(
        first_catalog, witnesses, 0
    )
    foreign_eval = _catalog_evaluation_with_one_winner(
        second_catalog, witnesses, 0
    )

    merged = merge_repeated_object_scene_anchor_catalog_evaluations(
        first_catalog, witnesses, first_eval, foreign_eval
    )
    assert merged.structurally_valid is False
    assert merged.disposition is Disposition.ERROR


def test_resealed_catalog_evaluation_cannot_bypass_outer_catalog_binding() -> None:
    _, manifest = _manifest(_plus(), "object_0000")
    catalog = build_object_scene_anchor_binding_catalog(
        manifest,
        ObjectSceneAnchorBindingSpec.part(),
        expected_object_id="object_0000",
    )
    witnesses = _witnesses()
    valid = _catalog_evaluation_with_one_winner(catalog, witnesses, 0)

    forged_data = valid.to_data()
    forged_data["hard_disposition"] = Disposition.CERTIFIED_ABSENT.value
    forged_data["expected_binding_digests"] = []
    forged_data["binding_evaluations"] = []
    forged_data["structurally_valid"] = True
    forged_data["disposition"] = Disposition.CERTIFIED_ABSENT.value
    forged_data["evaluation_digest"] = canonical_digest(
        {
            key: value
            for key, value in forged_data.items()
            if key != "evaluation_digest"
        }
    )
    forged = ObjectSceneAnchorCatalogEvaluation.from_data(forged_data)

    merged = merge_repeated_object_scene_anchor_catalog_evaluations(
        catalog, witnesses, valid, forged
    )
    assert merged.structurally_valid is False
    assert merged.disposition is Disposition.ERROR
