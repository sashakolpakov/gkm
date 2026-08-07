from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from io import BytesIO

from PIL import Image, ImageDraw
import pytest

from bongard.artifacts import canonical_digest
from bongard.evidence import Disposition, Provenance
from bongard.legs.contracts import Unit
from bongard.relational_scene import (
    SCENE_FRAME_Q16,
    ScalarInterval,
    SceneEntity,
    SceneFact,
    SceneFragment,
    SceneGlueError,
    SceneGlueFailureCode,
    SceneSnapshot,
    glue_scene_fragment,
    start_scene_snapshot,
    verify_scene_snapshot,
)
from bongard.visual_witness_bundle import extract_visual_witness_bundle


GRAPH_SCHEMA = canonical_digest(
    {
        "fixture": "typed relational scene",
        "entity_types": ["component", "hole", "loop"],
        "predicates": ["area_ratio", "touches"],
    }
)
LEG_DIGEST = canonical_digest({"fixture": "test relational leg", "version": 1})


def _panel(*, second_component: bool = True) -> bytes:
    image = Image.new("RGB", (64, 64), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((6, 8, 30, 44), fill="black")
    draw.rectangle((12, 14, 24, 38), fill="white")
    if second_component:
        draw.rectangle((44, 20, 54, 34), fill="black")
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def _provenance(snapshot: SceneSnapshot, suffix: str = "a") -> Provenance:
    return Provenance(
        producer="tests.relational_scene",
        version="1",
        method=f"fixture-{suffix}",
        input_digests=(snapshot.digest(),),
        artifact_digest=LEG_DIGEST,
        details=(("purpose", "transaction-test"),),
    )


def _component_entities(snapshot: SceneSnapshot) -> tuple[SceneEntity, SceneEntity]:
    first_scenario = snapshot.scenario_ids[0]
    components = tuple(
        entity
        for entity in snapshot.entities
        if entity.scenario_id == first_scenario and entity.entity_type == "component"
    )
    assert len(components) == 2
    return components


def _fact(
    snapshot: SceneSnapshot,
    provenance: Provenance,
    *,
    fact_id: str = "fact-area-ratio-a",
    predicate: str = "area_ratio",
    arguments: tuple[str, str] | None = None,
    argument_types: tuple[str, str] = ("component", "component"),
    disposition: Disposition = Disposition.PRESENT,
    interval: ScalarInterval | None = None,
    certificate: str | None = None,
    reason: str | None = None,
    error_type: str | None = None,
) -> SceneFact:
    components = _component_entities(snapshot)
    if arguments is None:
        arguments = (components[0].entity_id, components[1].entity_id)
    regions = tuple(
        dict.fromkeys(
            entity.source_region_digest
            for entity in components
            if entity.entity_id in arguments
        )
    )
    return SceneFact(
        fact_id=fact_id,
        predicate=predicate,
        arguments=arguments,
        argument_types=argument_types,
        scenario_id=snapshot.scenario_ids[0],
        frame_id=snapshot.frame_id,
        disposition=disposition,
        provenance_digest=provenance.digest(),
        source_region_digests=regions,
        unit=(interval.unit if interval is not None else Unit.DIMENSIONLESS),
        interval=interval or ScalarInterval(1.25, 1.5, Unit.DIMENSIONLESS),
        certificate=certificate,
        reason=reason,
        error_type=error_type,
    )


def _fragment(
    snapshot: SceneSnapshot,
    *,
    provenances: tuple[Provenance, ...] = (),
    entities: tuple[SceneEntity, ...] = (),
    facts: tuple[SceneFact, ...] = (),
    **changes: object,
) -> SceneFragment:
    values: dict[str, object] = {
        "panel_digest": snapshot.panel_digest,
        "parent_bundle_digest": snapshot.parent_bundle_digest,
        "parent_snapshot_digest": snapshot.digest(),
        "graph_schema_digest": snapshot.graph_schema_digest,
        "frame_id": snapshot.frame_id,
        "scenario_ids": snapshot.scenario_ids,
        "producer_leg": "typed_relation_leg",
        "producer_leg_digest": LEG_DIGEST,
        "provenances": tuple(sorted(provenances, key=lambda item: item.digest())),
        "entities": tuple(sorted(entities, key=lambda item: item.entity_id)),
        "facts": tuple(sorted(facts, key=lambda item: item.fact_id)),
    }
    values.update(changes)
    return SceneFragment(**values)  # type: ignore[arg-type]


def _assert_glue_code(
    snapshot: SceneSnapshot,
    fragment: SceneFragment,
    code: SceneGlueFailureCode,
) -> None:
    before = snapshot.to_data()
    before_digest = snapshot.digest()
    with pytest.raises(SceneGlueError) as raised:
        glue_scene_fragment(snapshot, fragment)
    assert raised.value.code is code
    assert snapshot.to_data() == before
    assert snapshot.digest() == before_digest


def test_start_snapshot_exactly_binds_bundle_and_qualifies_foundation_entities() -> None:
    bundle = extract_visual_witness_bundle(_panel())
    snapshot = start_scene_snapshot(bundle, GRAPH_SCHEMA)

    assert snapshot.panel_digest == bundle.panel_digest
    assert snapshot.parent_bundle_digest == bundle.digest()
    assert snapshot.frame_id == SCENE_FRAME_Q16
    assert snapshot.scenario_ids == tuple(
        scenario.scenario_id for scenario in bundle.base_packet.scenarios
    )
    assert snapshot.generation == 0
    assert snapshot.previous_snapshot_digest is None
    assert snapshot.applied_fragment_digest is None
    assert SceneSnapshot.from_data(snapshot.to_data()) == snapshot
    assert len(snapshot.digest()) == 64
    assert verify_scene_snapshot(snapshot, bundle) is snapshot

    expected_entity_count = sum(
        len(scenario.components) + len(scenario.holes)
        for scenario in bundle.base_packet.scenarios
    )
    assert len(snapshot.entities) == expected_entity_count
    for entity in snapshot.entities:
        assert entity.entity_id.startswith(entity.scenario_id + "/")
        assert entity.frame_id == SCENE_FRAME_Q16
        assert len(entity.source_witness_digest) == 64
        assert len(entity.source_region_digest) == 64
        assert SceneEntity.from_data(entity.to_data()) == entity
        if entity.entity_type == "hole":
            assert entity.owner_entity_id is not None
            owner = next(
                item for item in snapshot.entities if item.entity_id == entity.owner_entity_id
            )
            assert owner.entity_type == "component"
            assert owner.scenario_id == entity.scenario_id

    other_bundle = extract_visual_witness_bundle(_panel(second_component=False))
    with pytest.raises(ValueError, match="panel differs|exact parent bundle"):
        verify_scene_snapshot(snapshot, other_bundle)


def test_fact_keeps_argument_order_four_dispositions_units_and_canonical_bytes() -> None:
    snapshot = start_scene_snapshot(extract_visual_witness_bundle(_panel()), GRAPH_SCHEMA)
    provenance = _provenance(snapshot)
    components = _component_entities(snapshot)
    forward = _fact(snapshot, provenance)
    reverse = _fact(
        snapshot,
        provenance,
        fact_id="fact-area-ratio-b",
        arguments=(components[1].entity_id, components[0].entity_id),
    )

    assert forward.arguments == tuple(reversed(reverse.arguments))
    assert forward.logical_key != reverse.logical_key
    assert forward.digest() != reverse.digest()
    assert forward.interval == ScalarInterval(1.25, 1.5, Unit.DIMENSIONLESS)
    assert forward.unit is Unit.DIMENSIONLESS
    assert ScalarInterval.from_data(forward.interval.to_data()) == forward.interval
    assert SceneFact.from_data(forward.to_data()) == forward

    absent = replace(
        forward,
        fact_id="fact-absent",
        predicate="touches",
        disposition=Disposition.CERTIFIED_ABSENT,
        interval=None,
        certificate="exhaustive contact-region scan",
    )
    indeterminate = replace(
        absent,
        fact_id="fact-indeterminate",
        disposition=Disposition.INDETERMINATE,
        certificate=None,
        reason="scenario disagreement",
        unit=Unit.PROBABILITY,
        interval=ScalarInterval(0.4, 0.6, Unit.PROBABILITY),
    )
    error = replace(
        absent,
        fact_id="fact-error",
        disposition=Disposition.ERROR,
        certificate=None,
        reason="leg timed out",
        error_type="TimeoutError",
    )
    assert [item.disposition for item in (forward, absent, indeterminate, error)] == [
        Disposition.PRESENT,
        Disposition.CERTIFIED_ABSENT,
        Disposition.INDETERMINATE,
        Disposition.ERROR,
    ]
    for fact in (absent, indeterminate, error):
        assert SceneFact.from_data(fact.to_data()) == fact
    assert absent.unit is Unit.DIMENSIONLESS and absent.interval is None

    with pytest.raises(TypeError, match="literal canonical floats"):
        ScalarInterval(1, 2.0, Unit.COUNT)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cannot have.*none"):
        ScalarInterval(0.0, 1.0, Unit.NONE)


def test_successful_glue_is_additive_transactional_and_replayable() -> None:
    bundle = extract_visual_witness_bundle(_panel())
    parent = start_scene_snapshot(bundle, GRAPH_SCHEMA)
    provenance = _provenance(parent)
    components = _component_entities(parent)
    loop = SceneEntity(
        entity_id=f"{parent.scenario_ids[0]}/loop/loop-00000000",
        entity_type="loop",
        scenario_id=parent.scenario_ids[0],
        frame_id=parent.frame_id,
        source_witness_digest=canonical_digest({"derived": "loop-0"}),
        source_region_digest=components[0].source_region_digest,
        provenance_digest=provenance.digest(),
        owner_entity_id=components[0].entity_id,
    )
    fact = _fact(parent, provenance)
    fragment = _fragment(
        parent, provenances=(provenance,), entities=(loop,), facts=(fact,)
    )
    parent_data = parent.to_data()

    assert SceneFragment.from_data(fragment.to_data()) == fragment
    child = glue_scene_fragment(parent, fragment)

    assert parent.to_data() == parent_data
    assert parent.generation == 0
    assert child.generation == 1
    assert child.previous_snapshot_digest == parent.digest()
    assert child.applied_fragment_digest == fragment.digest()
    assert len(child.entities) == len(parent.entities) + 1
    assert len(child.facts) == 1
    assert set(parent.entities).issubset(child.entities)
    assert set(parent.provenances).issubset(child.provenances)
    assert (
        verify_scene_snapshot(
            child,
            bundle,
            previous_snapshot=parent,
            applied_fragment=fragment,
        )
        is child
    )
    with pytest.raises(FrozenInstanceError):
        child.generation = 7  # type: ignore[misc]


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"panel_digest": "0" * 64}, SceneGlueFailureCode.PANEL_MISMATCH),
        (
            {"parent_snapshot_digest": "0" * 64},
            SceneGlueFailureCode.PARENT_MISMATCH,
        ),
        (
            {"parent_bundle_digest": "0" * 64},
            SceneGlueFailureCode.PARENT_MISMATCH,
        ),
        ({"graph_schema_digest": "0" * 64}, SceneGlueFailureCode.SCHEMA_MISMATCH),
        ({"frame_id": "another.frame.v1"}, SceneGlueFailureCode.FRAME_MISMATCH),
    ],
)
def test_boundary_mismatch_codes_leave_parent_unchanged(
    change: dict[str, object], code: SceneGlueFailureCode
) -> None:
    snapshot = start_scene_snapshot(extract_visual_witness_bundle(_panel()), GRAPH_SCHEMA)
    _assert_glue_code(snapshot, _fragment(snapshot, **change), code)


def test_entity_owner_missing_provenance_and_leg_error_codes_are_typed() -> None:
    snapshot = start_scene_snapshot(extract_visual_witness_bundle(_panel()), GRAPH_SCHEMA)
    component, other = _component_entities(snapshot)

    _assert_glue_code(
        snapshot,
        _fragment(snapshot, entities=(component,)),
        SceneGlueFailureCode.ENTITY_CONFLICT,
    )
    changed_owner = replace(component, owner_entity_id=other.entity_id)
    _assert_glue_code(
        snapshot,
        _fragment(snapshot, entities=(changed_owner,)),
        SceneGlueFailureCode.OWNER_CONFLICT,
    )

    provenance = _provenance(snapshot)
    missing_owner = SceneEntity(
        entity_id=f"{snapshot.scenario_ids[0]}/loop/missing-owner",
        entity_type="loop",
        scenario_id=snapshot.scenario_ids[0],
        frame_id=snapshot.frame_id,
        source_witness_digest=canonical_digest({"fixture": "missing owner"}),
        source_region_digest=component.source_region_digest,
        provenance_digest=provenance.digest(),
        owner_entity_id=f"{snapshot.scenario_ids[0]}/component/not-present",
    )
    _assert_glue_code(
        snapshot,
        _fragment(snapshot, provenances=(provenance,), entities=(missing_owner,)),
        SceneGlueFailureCode.MISSING_ENTITY,
    )
    no_provenance = replace(missing_owner, owner_entity_id=component.entity_id)
    _assert_glue_code(
        snapshot,
        _fragment(snapshot, entities=(no_provenance,)),
        SceneGlueFailureCode.MISSING_PROVENANCE,
    )
    _assert_glue_code(
        snapshot,
        _fragment(
            snapshot,
            leg_error_type="RasterLegError",
            leg_error_reason="source mask unavailable",
        ),
        SceneGlueFailureCode.LEG_ERROR,
    )


def test_fact_duplicate_conflict_unit_missing_entity_and_schema_codes_are_typed() -> None:
    snapshot = start_scene_snapshot(extract_visual_witness_bundle(_panel()), GRAPH_SCHEMA)
    provenance = _provenance(snapshot)
    existing = _fact(snapshot, provenance)
    child = glue_scene_fragment(
        snapshot, _fragment(snapshot, provenances=(provenance,), facts=(existing,))
    )

    _assert_glue_code(
        child,
        _fragment(child, facts=(existing,)),
        SceneGlueFailureCode.DUPLICATE_FACT,
    )
    conflict = replace(
        existing,
        fact_id="fact-area-ratio-conflict",
        interval=ScalarInterval(4.0, 5.0, Unit.DIMENSIONLESS),
    )
    _assert_glue_code(
        child,
        _fragment(child, facts=(conflict,)),
        SceneGlueFailureCode.CONFLICTING_FACT,
    )

    components = _component_entities(child)
    other_scenario_components = tuple(
        entity
        for entity in child.entities
        if entity.scenario_id == child.scenario_ids[1]
        and entity.entity_type == "component"
    )
    assert len(other_scenario_components) == 2
    unit_conflict = SceneFact(
        fact_id="fact-area-ratio-other-scenario",
        predicate="area_ratio",
        arguments=tuple(item.entity_id for item in other_scenario_components),
        argument_types=("component", "component"),
        scenario_id=child.scenario_ids[1],
        frame_id=child.frame_id,
        disposition=Disposition.PRESENT,
        provenance_digest=provenance.digest(),
        source_region_digests=tuple(
            item.source_region_digest for item in other_scenario_components
        ),
        unit=Unit.FRACTION,
        interval=ScalarInterval(1.0, 2.0, Unit.FRACTION),
    )
    _assert_glue_code(
        child,
        _fragment(child, facts=(unit_conflict,)),
        SceneGlueFailureCode.UNIT_MISMATCH,
    )

    missing = replace(
        existing,
        fact_id="fact-missing-entity",
        predicate="touches",
        arguments=(components[0].entity_id, "missing/entity"),
        unit=Unit.NONE,
        interval=None,
    )
    _assert_glue_code(
        child,
        _fragment(child, facts=(missing,)),
        SceneGlueFailureCode.MISSING_ENTITY,
    )
    wrong_type = replace(
        existing,
        fact_id="fact-wrong-type",
        predicate="touches",
        argument_types=("hole", "component"),
        unit=Unit.NONE,
        interval=None,
    )
    _assert_glue_code(
        child,
        _fragment(child, facts=(wrong_type,)),
        SceneGlueFailureCode.SCHEMA_MISMATCH,
    )
    missing_provenance = replace(
        existing,
        fact_id="fact-missing-provenance",
        predicate="touches",
        provenance_digest="f" * 64,
        unit=Unit.NONE,
        interval=None,
    )
    _assert_glue_code(
        child,
        _fragment(child, facts=(missing_provenance,)),
        SceneGlueFailureCode.MISSING_PROVENANCE,
    )


def test_cold_decoders_reject_tampering_noncanonical_order_and_broken_references() -> None:
    bundle = extract_visual_witness_bundle(_panel())
    snapshot = start_scene_snapshot(bundle, GRAPH_SCHEMA)

    extra = snapshot.to_data()
    extra["unexpected"] = True
    with pytest.raises(ValueError, match="fields differ"):
        SceneSnapshot.from_data(extra)

    reordered = snapshot.to_data()
    reordered["entities"] = list(reversed(reordered["entities"]))
    with pytest.raises(ValueError, match="entity-ID sorted"):
        SceneSnapshot.from_data(reordered)

    missing_provenance = snapshot.to_data()
    missing_provenance["entities"][0]["provenance_digest"] = "0" * 64
    with pytest.raises(ValueError, match="missing provenance"):
        SceneSnapshot.from_data(missing_provenance)

    changed_foundation = replace(
        snapshot.entities[0], source_region_digest="0" * 64
    )
    forged = replace(snapshot, entities=(changed_foundation,) + snapshot.entities[1:])
    with pytest.raises(ValueError, match="changed a foundation"):
        verify_scene_snapshot(forged, bundle)
