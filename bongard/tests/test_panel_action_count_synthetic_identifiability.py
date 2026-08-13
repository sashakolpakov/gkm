from __future__ import annotations

from dataclasses import replace
from io import BytesIO

import numpy as np
from PIL import Image
import pytest

from bongard import panel_action_count_synthetic_identifiability as subject


EXPECTED_SOURCE_SHA256 = "7e52729cfda9effd831352802c47f3cf7f8d29f00da3a90dc7250bf1cdc722bf"


def _mask(raw: bytes) -> np.ndarray:
    with Image.open(BytesIO(raw)) as image:
        return np.asarray(image.convert("L")) < 128


def test_complete_pair_grid_and_exact_typed_records() -> None:
    assert subject.source_sha256() == EXPECTED_SOURCE_SHA256
    pairs = subject.valid_count_pairs()
    assert pairs == subject.valid_count_pairs()
    assert all(
        left is not right
        for left, right in zip(pairs, subject.valid_count_pairs(), strict=True)
    )
    assert len(pairs) == len(set(pairs)) == 54
    assert {(pair.straight, pair.arc) for pair in pairs} == {
        (straight, arc)
        for straight in range(10)
        for arc in range(10)
        if 1 <= straight + arc <= 9
    }
    with pytest.raises(TypeError, match="exact integer"):
        subject.CountPair(True, 0)
    with pytest.raises(ValueError, match="total"):
        subject.CountPair(0, 0)
    with pytest.raises(ValueError, match="total"):
        subject.CountPair(9, 1)


def test_renderer_is_deterministic_and_retains_action_pixel_provenance() -> None:
    program = subject.Program(
        "unit-carrier",
        (
            subject.LineAction(
                "line", subject.Point(160, 180), subject.Point(440, 380)
            ),
            subject.ArcAction(
                "arc",
                subject.Point(560, 600),
                subject.Point(720, 440),
                subject.Point(880, 600),
            ),
        ),
    )
    nuisance = subject.Nuisance("mirror_x_r90", 3, 1100)
    first = subject.render_program(program, nuisance)
    second = subject.render_program(program, nuisance)
    assert first == second
    assert first.declared_pair == subject.CountPair(1, 1)
    assert first.canonical_visible_pair == subject.visible_raster_component_normal_form(
        first.png_bytes
    )
    assert first.carrier_id == first.carrier_family == "unit-carrier"
    assert first.nuisance == nuisance
    assert tuple(row.action_id for row in first.provenance) == ("line", "arc")
    combined = set().union(*(set(row.ink_pixels) for row in first.provenance))
    assert combined == set(np.flatnonzero(_mask(first.png_bytes)))
    assert all(row.ink_pixels == tuple(sorted(set(row.ink_pixels))) for row in first.provenance)


def test_exact_history_ambiguities_have_one_renderer_visible_support() -> None:
    exact_cases = [case for case in subject.ambiguity_cases() if case.expected_relation == "exact"]
    assert {case.case_id for case in exact_cases} == {
        "one-line-vs-raster-aliased-split-collinear",
        "one-arc-vs-split-cocircular",
        "full-arc-plus-contained-left-arc",
        "one-line-plus-raster-subsumed-near-parallel-line",
        "endpoint-branch-vs-one-raster-equivalent-line",
        "endpoint-branch-alias-with-disconnected-context",
        "endpoint-branch-alias-with-touching-context",
        "three-line-chain-vs-one-raster-equivalent-arc",
        "line-chain-arc-alias-with-disconnected-context",
        "line-chain-arc-alias-with-touching-context",
    }
    for case in exact_cases:
        left = subject.render_program(case.left)
        right = subject.render_program(case.right)
        assert left.declared_pair != right.declared_pair
        assert left.png_bytes == right.png_bytes
        assert left.canonical_visible_pair == right.canonical_visible_pair

    # Explicit red-team geometry: the contained left arc must not inflate the
    # renderer-visible target relative to the full upper semicircle.
    containment = next(
        case for case in exact_cases
        if case.case_id == "full-arc-plus-contained-left-arc"
    )
    assert subject.render_program(containment.left).canonical_visible_pair == subject.CountPair(0, 1)
    assert subject.render_program(containment.right).canonical_visible_pair == subject.CountPair(0, 1)

    endpoint = next(
        case for case in exact_cases
        if case.case_id == "endpoint-branch-vs-one-raster-equivalent-line"
    )
    assert subject.render_program(endpoint.left).canonical_visible_pair == subject.CountPair(1, 0)
    assert subject.render_program(endpoint.right).canonical_visible_pair == subject.CountPair(1, 0)

    line_chain_arc = next(
        case for case in exact_cases
        if case.case_id == "three-line-chain-vs-one-raster-equivalent-arc"
    )
    line_chain = subject.render_program(line_chain_arc.left)
    one_arc = subject.render_program(line_chain_arc.right)
    assert line_chain.png_sha256 == (
        "sha256:b5bd0067160223fe3aca48471e63df7d5ded95066ab2ee446470c68e99be1e63"
    )
    assert line_chain.canonical_visible_pair == one_arc.canonical_visible_pair == (
        subject.CountPair(0, 1)
    )

    contextual = {
        case.case_id: (
            subject.render_program(case.left),
            subject.render_program(case.right),
        )
        for case in exact_cases
        if case.case_id.endswith("with-disconnected-context")
    }
    assert contextual[
        "endpoint-branch-alias-with-disconnected-context"
    ][0].canonical_visible_pair == subject.CountPair(2, 0)
    assert contextual[
        "line-chain-arc-alias-with-disconnected-context"
    ][0].canonical_visible_pair == subject.CountPair(1, 1)

    touching = {
        case.case_id: (
            subject.render_program(case.left),
            subject.render_program(case.right),
        )
        for case in exact_cases
        if case.case_id.endswith("with-touching-context")
    }
    assert set(touching) == {
        "endpoint-branch-alias-with-touching-context",
        "line-chain-arc-alias-with-touching-context",
    }
    assert all(
        left.canonical_visible_pair is None
        and right.canonical_visible_pair is None
        and left.png_bytes == right.png_bytes
        for left, right in touching.values()
    )
    for case in exact_cases:
        for panel in (
            subject.render_program(case.left),
            subject.render_program(case.right),
        ):
            assert panel.canonical_visible_pair == (
                subject.visible_raster_component_normal_form(panel.png_bytes)
            )


def test_arc_support_quotient_handles_reversal_and_wrap() -> None:
    forward = subject.ArcAction(
        "forward",
        subject.Point(192, 512), subject.Point(512, 192), subject.Point(832, 512),
    )
    reverse = subject.ArcAction(
        "reverse",
        subject.Point(832, 512), subject.Point(512, 192), subject.Point(192, 512),
    )
    left = subject.render_program(subject.Program("arc-reversal", (forward,)))
    right = subject.render_program(subject.Program("arc-reversal", (reverse,)))
    overdraw = subject.Program("arc-reversal", (forward, reverse))
    assert left.png_bytes == right.png_bytes
    assert subject.canonical_visible_pair(overdraw) == subject.CountPair(0, 1)


def test_near_collision_is_nonidentical_and_audit_is_explicitly_bounded() -> None:
    case = next(case for case in subject.ambiguity_cases() if case.expected_relation == "near")
    left = subject.render_program(case.left)
    right = subject.render_program(case.right)
    xor = int(np.logical_xor(_mask(left.png_bytes), _mask(right.png_bytes)).sum())
    assert left.png_bytes != right.png_bytes
    assert 1 <= xor <= 8
    assert left.declared_pair != right.declared_pair

    audit = subject.ambiguity_audit()
    assert audit.scope == "bounded_synthetic_only_not_exhaustive"
    assert audit.denominator_pixels == 4096
    assert audit.compared_different_target_pairs <= audit.max_near_comparisons
    assert audit.qualifying_near_collision_count >= len(audit.near_collisions)
    assert len(audit.near_collisions) <= audit.max_retained_near_collisions
    assert audit.exact_canonical_conflict_count == 0
    assert any(row.xor_pixels == xor for row in audit.near_collisions)
    assert any(row.declared_target_conflict for row in audit.exact_collisions)
    assert not any(row.canonical_target_conflict for row in audit.exact_collisions)


def test_raster_indistinguishable_shallow_arc_is_outside_bounded_grammar() -> None:
    with pytest.raises(ValueError, match="curvature"):
        subject.ArcAction(
            "too-shallow",
            subject.Point(183, 512),
            subject.Point(512, 504),
            subject.Point(841, 512),
        )


def test_every_family_nuisance_cell_has_each_visible_target_exactly_once() -> None:
    nuisances = (
        subject.Nuisance("identity", 2, 1000),
        subject.Nuisance("mirror_x_r270", 2, 1000),
    )
    corpus = subject.build_balanced_corpus(
        carrier_families=subject.CARRIER_FAMILIES,
        nuisances=nuisances,
    )
    assert len(corpus) == len(subject.CARRIER_FAMILIES) * len(nuisances) * 54
    for family in subject.CARRIER_FAMILIES:
        for nuisance in nuisances:
            cell = [
                row for row in corpus
                if row.carrier_family == family and row.nuisance == nuisance
            ]
            assert len(cell) == 54
            assert [row.declared_pair for row in cell] == list(subject.valid_count_pairs())
            assert [row.canonical_visible_pair for row in cell] == list(subject.valid_count_pairs())
            assert all(row.declared_pair == row.canonical_visible_pair for row in cell)
            assert all(row.carrier_id == row.panel.carrier_family == family for row in cell)

    with pytest.raises(ValueError, match="exactly one sample"):
        subject.build_balanced_corpus(samples_per_pair_per_carrier=2)


def test_carrier_split_fails_closed_against_identity_relabelling_and_leakage() -> None:
    corpus = subject.build_balanced_corpus(
        carrier_families=("lattice", "radial"),
        nuisances=(subject.Nuisance(),),
    )
    split = subject.carrier_disjoint_split(corpus, held_out_families=("radial",))
    assert {row.carrier_id for row in split.train} == {"lattice"}
    assert {row.carrier_id for row in split.held_out} == {"radial"}
    assert not (
        {subject.d4_raster_orbit_digest(row.png_bytes) for row in split.train}
        & {subject.d4_raster_orbit_digest(row.png_bytes) for row in split.held_out}
    )

    panel = corpus[0].panel
    with pytest.raises(ValueError, match="binds carrier_id"):
        replace(panel, carrier_id="radial")

    # A nuisance change cannot turn a train carrier into a held-out carrier:
    # split identity is the immutable carrier ID, not PNG or nuisance identity.
    changed_panel = subject.render_program(
        subject.Program("lattice", (
            subject.LineAction("x", subject.Point(200, 200), subject.Point(400, 200)),
        )),
        subject.Nuisance("r90", 4, 800),
    )
    adversarial = subject.SyntheticSample("adversarial-style", changed_panel)
    with pytest.raises(ValueError, match="leakage"):
        subject.CorpusSplit(
            split.train,
            (*split.held_out, adversarial),
            ("lattice", "radial"),
        )


def test_collision_records_reject_semantically_inconsistent_metrics() -> None:
    audit = subject.ambiguity_audit()
    exact = audit.exact_collisions[0]
    near = audit.near_collisions[0]
    with pytest.raises(ValueError, match="collision fields|identical rasters"):
        replace(exact, raster_digests=(exact.raster_digests[0], "sha256:" + "0" * 64))
    with pytest.raises(ValueError, match="IoU"):
        replace(near, iou_millionths=near.iou_millionths - 1)
    with pytest.raises(ValueError, match="fixed denominator"):
        replace(near, denominator_pixels=4095)
    with pytest.raises(ValueError, match="conflict count"):
        replace(audit, exact_canonical_conflict_count=audit.exact_canonical_conflict_count + 1)
    with pytest.raises(ValueError, match="accounting"):
        replace(audit, compared_different_target_pairs=audit.possible_different_target_pairs + 1)
    with pytest.raises(ValueError, match="retention accounting"):
        replace(audit, qualifying_near_collision_count=len(audit.near_collisions) - 1)


def test_collision_audit_rejects_resealed_or_forged_panel_targets() -> None:
    case = next(
        case for case in subject.ambiguity_cases()
        if case.case_id == "three-line-chain-vs-one-raster-equivalent-arc"
    )
    issued = subject.render_program(case.left)
    forged = replace(issued, canonical_visible_pair=subject.CountPair(3, 0))
    with pytest.raises(subject.SyntheticIdentifiabilityError, match="not issued"):
        subject.AuditCandidate("forged", forged)
    with pytest.raises(subject.SyntheticIdentifiabilityError, match="not issued"):
        subject.SyntheticSample("forged", forged)

    nested = subject.render_program(case.left)
    object.__setattr__(nested.canonical_visible_pair, "straight", 3)
    object.__setattr__(nested.canonical_visible_pair, "arc", 0)
    with pytest.raises(subject.SyntheticIdentifiabilityError, match="not issued"):
        subject.require_issued_rendered_panel(nested)
    with pytest.raises(subject.SyntheticIdentifiabilityError, match="not issued"):
        subject.AuditCandidate("nested-forged", nested)

    unresolved_case = next(
        case for case in subject.ambiguity_cases()
        if case.case_id == "line-chain-arc-alias-with-touching-context"
    )
    unresolved = subject.render_program(unresolved_case.left)
    assert unresolved.canonical_visible_pair is None
    forged_unresolved = replace(
        unresolved, canonical_visible_pair=subject.CountPair(4, 0)
    )
    with pytest.raises(subject.SyntheticIdentifiabilityError, match="not issued"):
        subject.AuditCandidate("forged-unresolved", forged_unresolved)


@pytest.mark.parametrize(
    "field,value",
    (
        ("stroke_width", 2.0),
        ("scale_milli", 1000.0),
    ),
)
def test_issued_panel_seal_rejects_numeric_type_aliases_in_nuisance(
    field: str, value: object
) -> None:
    caller_nuisance = subject.Nuisance()
    panel = subject.render_program(
        subject._carrier_program(subject.CountPair(1, 0), "lattice"),
        caller_nuisance,
    )
    assert panel.nuisance is not caller_nuisance
    object.__setattr__(panel.nuisance, field, value)
    with pytest.raises(subject.SyntheticIdentifiabilityError, match="not issued"):
        subject.require_issued_rendered_panel(panel)
    with pytest.raises(subject.SyntheticIdentifiabilityError, match="not issued"):
        subject.AuditCandidate(f"forged-{field}", panel)
    clean = subject.render_program(
        subject._carrier_program(subject.CountPair(1, 0), "lattice"),
        caller_nuisance,
    )
    subject.require_issued_rendered_panel(clean)


def test_issued_panel_seal_rejects_boolean_count_alias() -> None:
    panel = subject.render_program(
        subject._carrier_program(subject.CountPair(1, 0), "lattice")
    )
    object.__setattr__(panel.declared_pair, "straight", True)
    with pytest.raises(subject.SyntheticIdentifiabilityError, match="not issued"):
        subject.require_issued_rendered_panel(panel)
    with pytest.raises(subject.SyntheticIdentifiabilityError, match="not issued"):
        subject.SyntheticSample("forged-count", panel)


@pytest.mark.parametrize("value", (192.0, True, -1))
def test_renderer_rejects_mutated_point_before_issuance(value: object) -> None:
    program = subject._carrier_program(subject.CountPair(1, 0), "lattice")
    object.__setattr__(program.actions[0].start, "x", value)
    with pytest.raises((TypeError, ValueError), match="exact integer|bounded"):
        subject.render_program(program)
    with pytest.raises((TypeError, ValueError), match="exact integer|bounded"):
        subject.canonical_visible_pair(program)


@pytest.mark.parametrize("value", (2.0, True, 0))
def test_public_target_and_renderer_reject_mutated_nuisance(
    value: object,
) -> None:
    program = subject._carrier_program(subject.CountPair(1, 0), "lattice")
    nuisance = subject.Nuisance()
    object.__setattr__(nuisance, "stroke_width", value)
    with pytest.raises((TypeError, ValueError), match="exact integer|stroke_width"):
        subject.render_program(program, nuisance)
    with pytest.raises((TypeError, ValueError), match="exact integer|stroke_width"):
        subject.canonical_visible_pair(program, nuisance)


def test_renderer_revalidates_action_geometry_and_has_no_mutable_default() -> None:
    program = subject._carrier_program(subject.CountPair(1, 0), "lattice")
    action = program.actions[0]
    object.__setattr__(action, "end", subject.Point(action.start.x + 1, action.start.y))
    with pytest.raises(ValueError, match="at least four output pixels"):
        subject.render_program(program)
    with pytest.raises(ValueError, match="at least four output pixels"):
        subject.canonical_visible_pair(program)
    assert subject.render_program.__defaults__ == (None,)
    assert subject.canonical_visible_pair.__defaults__ == (None,)
    assert subject.ambiguity_audit.__defaults__ == (None,)

    container_mutation = subject._carrier_program(
        subject.CountPair(1, 0), "lattice"
    )
    object.__setattr__(container_mutation, "actions", list(container_mutation.actions))
    with pytest.raises(ValueError, match="between one and nine actions"):
        subject.render_program(container_mutation)

    kind_mutation = subject._carrier_program(
        subject.CountPair(1, 0), "lattice"
    )
    class Text(str):
        pass
    object.__setattr__(kind_mutation.actions[0], "kind", Text("line"))
    with pytest.raises(ValueError, match="kind"):
        subject.render_program(kind_mutation)

    point_subclass_mutation = subject._carrier_program(
        subject.CountPair(1, 0), "lattice"
    )
    class PointSubclass(subject.Point):
        pass
    start = point_subclass_mutation.actions[0].start
    object.__setattr__(
        point_subclass_mutation.actions[0],
        "start",
        PointSubclass(start.x, start.y),
    )
    with pytest.raises(TypeError, match="exact Point"):
        subject.render_program(point_subclass_mutation)


def test_default_nuisance_inventory_is_fresh_and_exact() -> None:
    first = subject.default_nuisances()
    second = subject.default_nuisances()
    assert first == second
    assert all(left is not right for left, right in zip(first, second, strict=True))
    object.__setattr__(first[0], "d4", "r180")
    assert subject.default_nuisances()[0] == subject.Nuisance(
        "identity", 2, 1000
    )
    assert subject.build_balanced_corpus.__kwdefaults__["nuisances"] is None

    poisoned = subject.valid_count_pairs()
    object.__setattr__(poisoned[0], "arc", True)
    fresh_pairs = subject.valid_count_pairs()
    assert all(type(value) is int for pair in fresh_pairs for value in pair)
    assert fresh_pairs[0] == subject.CountPair(0, 1)


def test_split_and_collision_audit_revalidate_nested_public_records() -> None:
    corpus = subject.build_balanced_corpus(
        carrier_families=("lattice", "radial"),
        nuisances=(subject.Nuisance(),),
    )
    sample = corpus[0]
    object.__setattr__(sample, "sample_id", " forged ")
    with pytest.raises(ValueError, match="canonical nonempty"):
        subject.carrier_disjoint_split(corpus, held_out_families=("radial",))

    clean = subject.render_program(
        subject._carrier_program(subject.CountPair(1, 0), "lattice")
    )
    first = subject.AuditCandidate("first", clean)
    second = subject.AuditCandidate("second", clean)
    object.__setattr__(first, "candidate_id", " bad ")
    with pytest.raises(ValueError, match="canonical and nonempty"):
        subject.audit_collisions((first, second))


def test_singleton_normal_form_uses_line_as_exact_equal_complexity_tiebreak() -> None:
    panel = subject.render_program(
        subject._carrier_program(subject.CountPair(1, 0), "lattice"),
        subject.Nuisance("r90", 1, 900),
    )
    assert subject.has_bounded_exact_single_line_explanation(panel.png_bytes)
    assert subject.has_bounded_exact_single_arc_explanation(panel.png_bytes)
    assert panel.canonical_visible_pair == subject.CountPair(1, 0)
