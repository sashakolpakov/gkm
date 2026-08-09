from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from bongard.evidence import Disposition
import bongard.panel_soft_ontology as o


def _d(char: str) -> str:
    return char * 64


def _parameters() -> dict[o.FeatureFamily, o.FeatureParameters]:
    return {
        o.FeatureFamily.COMPONENT_COUNT: o.ComponentCountParameters(o.ClosedCount.TWO),
        o.FeatureFamily.EXACT_SEGMENT_COUNT: o.ExactSegmentCountParameters(o.ClosedCount.TWO),
        o.FeatureFamily.STRAIGHT_SEGMENT_COUNT: o.StraightSegmentCountParameters(
            o.ClosedCount.TWO
        ),
        o.FeatureFamily.MARKER_PATTERN: o.MarkerPatternParameters(
            o.MarkerPrimitive.DOT, o.ClosedCount.TWO, o.MarkerArrangement.LINEAR
        ),
        o.FeatureFamily.GESTALT_RESEMBLANCE: o.GestaltResemblanceParameters(o.GestaltKind.BIRD_LIKE),
        o.FeatureFamily.SEGMENT_ORIENTATION: o.SegmentOrientationParameters(
            o.OrientationClass.OBLIQUE_ASCENDING, o.ClosedAggregation.ONE_WITNESSED
        ),
        o.FeatureFamily.CORNER_ANGLE: o.CornerAngleParameters(
            o.CornerAngleClass.OBTUSE, o.ClosedAggregation.ONE_WITNESSED
        ),
        o.FeatureFamily.TURN_PROFILE: o.TurnProfileParameters(o.TurnProfileClass.ALTERNATING),
        o.FeatureFamily.OPEN_TRACE: o.OpenTraceParameters(o.OpenTraceKind.SIMPLE_UNBRANCHED),
        o.FeatureFamily.CLOSED_LOOP: o.ClosedLoopParameters(o.ClosedLoopKind.SIMPLE),
        o.FeatureFamily.POINT_CONTACT: o.PointContactParameters(o.PointContactKind.TANGENTIAL),
        o.FeatureFamily.VISIBLE_GAP: o.VisibleGapParameters(o.VisibleGapKind.BETWEEN_CONTOURS),
        o.FeatureFamily.ENCLOSURE: o.EnclosureParameters(o.EnclosureKind.FULLY_INSIDE),
        o.FeatureFamily.SYMMETRY: o.SymmetryParameters(o.SymmetryKind.REFLECTIONAL),
        o.FeatureFamily.SHARED_BOUNDARY_ADJACENCY: o.SharedBoundaryAdjacencyParameters(
            o.SharedBoundaryKind.STRAIGHT_SEGMENT
        ),
        o.FeatureFamily.ASPECT_RATIO: o.AspectRatioParameters(o.AspectRatioClass.WIDE),
        o.FeatureFamily.TEXTURE_COMPOSITION: o.TextureCompositionParameters(
            o.TextureCompositionClass.MIXED_REGIONS
        ),
    }


def _first_spec(family: o.FeatureFamily) -> o.PanelFeatureSpec:
    scope, frame = sorted(
        o.FAMILY_CONTRACTS[family].allowed_scope_frames,
        key=lambda item: (item[0].value, item[1].value),
    )[0]
    return o.PanelFeatureSpec(family, scope, frame, _parameters()[family])


def _region(x: int = 1, y: int = 1) -> o.QuantizedRegion:
    return o.QuantizedRegion(o.QuantizedPoint(x, y), o.QuantizedPoint(x + 2, y + 2))


def _inventory() -> o.OwnerInventory:
    owners = (
        o.PanelLocalOwner(
            o.OwnerId("owner_0001"),
            o.OwnerKind.FIGURE,
            o.QuantizedRegion(o.QuantizedPoint(0, 0), o.QuantizedPoint(7, 7)),
        ),
        o.PanelLocalOwner(
            o.OwnerId("owner_0002"),
            o.OwnerKind.FIGURE,
            o.QuantizedRegion(o.QuantizedPoint(7, 0), o.QuantizedPoint(14, 7)),
        ),
        o.PanelLocalOwner(
            o.OwnerId("owner_0003"),
            o.OwnerKind.TRACE,
            _region(1, 1),
            (o.OwnerId("owner_0001"),),
        ),
        o.PanelLocalOwner(
            o.OwnerId("owner_0004"),
            o.OwnerKind.LOOP,
            _region(9, 1),
            (o.OwnerId("owner_0002"),),
        ),
        o.PanelLocalOwner(
            o.OwnerId("owner_0005"),
            o.OwnerKind.SEGMENT,
            _region(2, 2),
            (o.OwnerId("owner_0001"),),
        ),
        o.PanelLocalOwner(
            o.OwnerId("owner_0006"),
            o.OwnerKind.SEGMENT,
            _region(3, 3),
            (o.OwnerId("owner_0001"),),
        ),
        o.PanelLocalOwner(
            o.OwnerId("owner_0007"),
            o.OwnerKind.MARKER,
            _region(4, 4),
            (o.OwnerId("owner_0001"),),
        ),
        o.PanelLocalOwner(
            o.OwnerId("owner_0008"),
            o.OwnerKind.MARKER,
            _region(5, 5),
            (o.OwnerId("owner_0001"),),
        ),
    )
    return o.OwnerInventory(
        _d("a"),
        _d("b"),
        o.EnumerationResolution.GRID16_FULL_PANEL,
        _d("c"),
        True,
        owners,
    )


def _point_spec() -> o.PanelFeatureSpec:
    return o.PanelFeatureSpec(
        o.FeatureFamily.POINT_CONTACT,
        o.SubjectScope.FIGURE_PAIR,
        o.ReferenceFrame.NONE,
        o.PointContactParameters(o.PointContactKind.TANGENTIAL),
    )


def _relation_inventory() -> o.OwnerInventory:
    base = _inventory()
    third = o.PanelLocalOwner(
        o.OwnerId("owner_0009"), o.OwnerKind.FIGURE, _region(12, 12)
    )
    return o.OwnerInventory(
        base.panel_digest,
        base.enumeration_protocol_digest,
        base.enumeration_resolution,
        base.enumeration_receipt_digest,
        base.enumeration_complete,
        base.owners + (third,),
    )


def _point_witness(inventory: o.OwnerInventory) -> o.PanelFeatureWitness:
    spec = _point_spec()
    subject = o.SubjectBinding(
        o.SubjectBindingKind.UNORDERED_PAIR,
        (o.OwnerId("owner_0001"), o.OwnerId("owner_0002")),
    )
    rays = (
        o.OwnerRay(o.OwnerId("owner_0001"), o.RayDirection.E),
        o.OwnerRay(o.OwnerId("owner_0001"), o.RayDirection.W),
        o.OwnerRay(o.OwnerId("owner_0002"), o.RayDirection.E),
        o.OwnerRay(o.OwnerId("owner_0002"), o.RayDirection.W),
    )
    payload = o.PointContactWitnessPayload(
        o.PointContactKind.TANGENTIAL,
        o.QuantizedPoint(7, 1),
        rays,
        (_region(0, 4), _region(10, 4)),
    )
    return o.PanelFeatureWitness(spec, inventory, _d("d"), subject, payload, _d("e"))


def _pair_absence(inventory: o.OwnerInventory) -> o.AbsenceCertificate:
    spec = _point_spec()
    domain = o.SearchResolutionDomain.for_spec(spec)
    subjects = o.eligible_subject_bindings(spec, inventory, domain)
    rejections = tuple(
        o.OwnerRejection(
            spec,
            inventory,
            _d("d"),
            subject,
            o.ExhaustiveSearchNonmatchEvidence(
                (o.subject_search_region(subject, inventory),), _d("f")
            ),
        )
        for subject in subjects
    )
    return o.AbsenceCertificate(
        spec,
        inventory,
        _d("d"),
        domain,
        _d("1"),
        subjects,
        rejections,
        True,
        _d("2"),
    )


def _assessment(risk: o.CalibrationRisk, char: str) -> o.CalibrationAssessment:
    return o.CalibrationAssessment(
        risk,
        _d(char),
        _d("3"),
        200,
        50_000,
        20_000,
        950_000,
        100,
        200,
        _d("4"),
    )


def test_total_family_scope_frame_and_parameter_matrix() -> None:
    params = _parameters()
    assert set(o.FAMILY_CONTRACTS) == set(o.FeatureFamily)
    assert len(o.FeatureFamily) == 17
    for family in o.FeatureFamily:
        contract = o.FAMILY_CONTRACTS[family]
        for scope in o.SubjectScope:
            for frame in o.ReferenceFrame:
                if (scope, frame) in contract.allowed_scope_frames:
                    o.PanelFeatureSpec(family, scope, frame, params[family])
                else:
                    with pytest.raises(o.PanelSoftOntologyError):
                        o.PanelFeatureSpec(family, scope, frame, params[family])
        scope, frame = next(iter(contract.allowed_scope_frames))
        for wrong_family, wrong_parameters in params.items():
            if wrong_family is family:
                continue
            with pytest.raises(o.PanelSoftOntologyError):
                o.PanelFeatureSpec(family, scope, frame, wrong_parameters)
    assert all(
        (family, scope, o.ReferenceFrame.FIGURE_PAIR_AXIS)
        not in {
            (family, item[0], item[1])
            for item in o.FAMILY_CONTRACTS[family].allowed_scope_frames
        }
        for family in o.FeatureFamily
        for scope in o.SubjectScope
    )


def test_spec_identity_is_context_and_implementation_independent() -> None:
    spec = _first_spec(o.FeatureFamily.GESTALT_RESEMBLANCE)
    narration_a = o.PanelFeatureNarration(
        spec.spec_digest, "A bird-like object", ("A beak-like tip is visible",)
    )
    narration_b = o.PanelFeatureNarration(
        spec.spec_digest, "An avian silhouette", ("Two wing-like strokes are visible",)
    )
    proposal_a = o.NativeFeatureProposal(
        spec,
        narration_a,
        o.NativeProposalProvenance(
            o.NativeOrientation.SIDE0_POSITIVE, _d("a"), _d("b"), _d("c"), _d("d")
        ),
    )
    proposal_b = o.NativeFeatureProposal(
        spec,
        narration_b,
        o.NativeProposalProvenance(
            o.NativeOrientation.SIDE1_POSITIVE, _d("e"), _d("f"), _d("1"), _d("2")
        ),
    )
    assert proposal_a.spec.spec_digest == proposal_b.spec.spec_digest
    assert proposal_a.proposal_digest != proposal_b.proposal_digest
    data = spec.to_data()
    rendered = repr(data).lower()
    for forbidden in ("lean", "prose", "narration", "task", "side", "orientation", "owner", "source", "receipt", "model"):
        assert forbidden not in rendered
    assert "parameter_type" not in repr(o.feature_catalog_data())
    tampered = dict(data)
    tampered["native_orientation"] = "side0_positive"
    with pytest.raises(o.PanelSoftOntologyError):
        o.PanelFeatureSpec.from_data(tampered)


def test_grid16_exact_types_and_nested_roundtrip() -> None:
    with pytest.raises(o.PanelSoftOntologyError):
        o.QuantizedPoint(True, 1)
    with pytest.raises(o.PanelSoftOntologyError):
        o.QuantizedPoint(16, 1)
    inventory = _inventory()
    assert o.OwnerInventory.from_data(inventory.to_data()) == inventory
    injected = deepcopy(inventory.to_data())
    injected["spec_digest"] = _d("0")
    with pytest.raises(o.PanelSoftOntologyError):
        o.OwnerInventory.from_data(injected)


def test_inventory_is_candidate_independent_and_bindings_are_typed() -> None:
    inventory = _inventory()
    before = inventory.inventory_digest
    open_spec = _first_spec(o.FeatureFamily.OPEN_TRACE)
    closed_spec = _first_spec(o.FeatureFamily.CLOSED_LOOP)
    assert [item.owner_ids for item in o.eligible_subject_bindings(open_spec, inventory)] == [
        (o.OwnerId("owner_0003"),)
    ]
    assert [item.owner_ids for item in o.eligible_subject_bindings(closed_spec, inventory)] == [
        (o.OwnerId("owner_0004"),)
    ]
    assert o.registered_sibling_relation(open_spec, closed_spec) is None
    sibling_rows = {
        item["relation_id"]: item for item in o.feature_catalog_data()["sibling_registry"]
    }
    assert sibling_rows["open-trace-vs-closed-loop-v1"]["direct_conflict_enabled"] is False
    assert sibling_rows["point-contact-and-visible-gap-v1"]["mutually_exclusive"] is False
    assert inventory.inventory_digest == before
    pair_bindings = o.eligible_subject_bindings(_point_spec(), inventory)
    assert len(pair_bindings) == 1
    assert all(item.kind is o.SubjectBindingKind.UNORDERED_PAIR for item in pair_bindings)
    enclosure = _first_spec(o.FeatureFamily.ENCLOSURE)
    assert len(o.eligible_subject_bindings(enclosure, inventory)) == 2
    with pytest.raises(o.PanelSoftOntologyError):
        o.SubjectBinding(
            o.SubjectBindingKind.UNORDERED_PAIR,
            (o.OwnerId("owner_0002"), o.OwnerId("owner_0001")),
        )


def test_closed_counts_require_exact_registered_membership() -> None:
    inventory = _inventory()
    component_spec = _first_spec(o.FeatureFamily.COMPONENT_COUNT)
    panel = o.SubjectBinding(o.SubjectBindingKind.PANEL, ())
    payload = o.CountWitnessPayload(
        (o.OwnerId("owner_0001"), o.OwnerId("owner_0002")), True, _d("5")
    )
    component_witness = o.PanelFeatureWitness(
        component_spec, inventory, _d("d"), panel, payload, _d("e")
    )
    with pytest.raises(o.PanelSoftOntologyError):
        o.PanelFeatureWitness(
            component_spec,
            inventory,
            _d("d"),
            panel,
            o.CountWitnessPayload((o.OwnerId("owner_0001"),), True, _d("5")),
            _d("e"),
        )
    segment_spec = o.PanelFeatureSpec(
        o.FeatureFamily.EXACT_SEGMENT_COUNT,
        o.SubjectScope.ONE_COHERENT_FIGURE,
        o.ReferenceFrame.NONE,
        o.ExactSegmentCountParameters(o.ClosedCount.TWO),
    )
    subject = o.SubjectBinding(o.SubjectBindingKind.UNARY, (o.OwnerId("owner_0001"),))
    segment_payload = o.CountWitnessPayload(
        (o.OwnerId("owner_0005"), o.OwnerId("owner_0006")), True, _d("5")
    )
    o.PanelFeatureWitness(segment_spec, inventory, _d("d"), subject, segment_payload, _d("e"))
    domain = o.FeatureDomain(
        component_spec.family,
        component_spec.subject_scope,
        component_spec.reference_frame,
        (component_spec,),
    )
    presence = o.PresenceCalibrationGrant(
        domain,
        _d("d"),
        _d("7"),
        _assessment(o.CalibrationRisk.FALSE_POSITIVE_CLAIM, "8"),
        _d("9"),
    )
    authority = o.FeatureCalibrationAuthority(
        "calibration.count.v1", domain, _d("d"), _d("c"), _d("e"), presence
    )
    token = o.verify_feature_calibration_authority(
        authority,
        capability=o.CalibrationCapability.PRESENCE,
        expected_authority_digest=authority.authority_digest,
        expected_grant_digest=presence.grant_digest,
        trusted_root_digest=_d("c"),
        verifier_receipt_digest=_d("f"),
        campaign_time_unix=150,
    )
    raw = o.RawFeatureMeasurement(
        component_spec,
        inventory,
        _d("d"),
        _d("7"),
        o.RawMeasurementState.WITNESS_ASSERTED,
        witness=component_witness,
    )
    custody = o.verify_raw_measurement_custody(
        raw,
        expected_measurement_digest=raw.measurement_digest,
        expected_inventory_digest=inventory.inventory_digest,
        expected_enumeration_receipt_digest=inventory.enumeration_receipt_digest,
        expected_evidence_receipt_digest=component_witness.witness_receipt_digest,
        verifier_receipt_digest=_d("6"),
    )
    assert o.project_raw_measurement(raw, token, custody) is Disposition.INDETERMINATE


def test_straight_segment_count_requires_explicit_exhaustive_classification() -> None:
    inventory = _inventory()
    spec = o.PanelFeatureSpec(
        o.FeatureFamily.STRAIGHT_SEGMENT_COUNT,
        o.SubjectScope.ONE_COHERENT_FIGURE,
        o.ReferenceFrame.NONE,
        o.StraightSegmentCountParameters(o.ClosedCount.TWO),
    )
    subject = o.SubjectBinding(
        o.SubjectBindingKind.UNARY, (o.OwnerId("owner_0001"),)
    )
    eligible = (o.OwnerId("owner_0005"), o.OwnerId("owner_0006"))
    lines = (
        o.QuantizedSegment(o.QuantizedPoint(2, 2), o.QuantizedPoint(4, 4)),
        o.QuantizedSegment(o.QuantizedPoint(3, 3), o.QuantizedPoint(5, 4)),
    )
    payload = o.StraightSegmentCountWitnessPayload(
        eligible,
        eligible,
        lines,
        True,
        _d("5"),
    )
    witness = o.PanelFeatureWitness(
        spec, inventory, _d("d"), subject, payload, _d("e")
    )
    assert o.PanelFeatureWitness.from_data(witness.to_data()) == witness
    assert o.segment_owner_ids_for_subject(subject, inventory) == eligible

    # Generic segment ownership is a different fact and cannot serve as the
    # straightness classification payload.
    with pytest.raises(o.PanelSoftOntologyError, match="wrong witness payload"):
        o.PanelFeatureWitness(
            spec,
            inventory,
            _d("d"),
            subject,
            o.CountWitnessPayload(eligible, True, _d("5")),
            _d("e"),
        )
    with pytest.raises(o.PanelSoftOntologyError, match="exact membership"):
        o.PanelFeatureWitness(
            spec,
            inventory,
            _d("d"),
            subject,
            o.StraightSegmentCountWitnessPayload(
                eligible + (o.OwnerId("owner_0007"),),
                eligible,
                lines,
                True,
                _d("5"),
            ),
            _d("e"),
        )
    with pytest.raises(o.PanelSoftOntologyError, match="complete classification"):
        o.StraightSegmentCountWitnessPayload(
            eligible,
            eligible,
            lines,
            False,
            _d("5"),
        )

    rules = o.feature_catalog_data()["count_membership_rules"]
    assert rules[o.FeatureFamily.EXACT_SEGMENT_COUNT.value] == o.SEGMENT_MEMBERSHIP_RULE_ID
    assert (
        rules[o.FeatureFamily.STRAIGHT_SEGMENT_COUNT.value]
        == o.STRAIGHT_SEGMENT_CLASSIFICATION_RULE_ID
    )


def test_count_membership_uses_coherent_roots_and_transitive_descendants() -> None:
    root_trace = o.PanelLocalOwner(
        o.OwnerId("owner_0001"), o.OwnerKind.TRACE, _region(1, 1)
    )
    root_loop = o.PanelLocalOwner(
        o.OwnerId("owner_0002"), o.OwnerKind.LOOP, _region(5, 1)
    )
    component_inventory = o.OwnerInventory(
        _d("a"),
        _d("b"),
        o.EnumerationResolution.GRID16_FULL_PANEL,
        _d("c"),
        True,
        (root_trace, root_loop),
    )
    assert o.coherent_top_level_component_owner_ids(component_inventory) == (
        root_trace.owner_id,
        root_loop.owner_id,
    )
    component_spec = o.PanelFeatureSpec(
        o.FeatureFamily.COMPONENT_COUNT,
        o.SubjectScope.WHOLE_PANEL,
        o.ReferenceFrame.NONE,
        o.ComponentCountParameters(o.ClosedCount.TWO),
    )
    o.PanelFeatureWitness(
        component_spec,
        component_inventory,
        _d("d"),
        o.SubjectBinding(o.SubjectBindingKind.PANEL, ()),
        o.CountWitnessPayload(
            (root_trace.owner_id, root_loop.owner_id), True, _d("e")
        ),
        _d("f"),
    )

    figure = o.PanelLocalOwner(
        o.OwnerId("owner_0001"),
        o.OwnerKind.FIGURE,
        o.QuantizedRegion(o.QuantizedPoint(0, 0), o.QuantizedPoint(8, 8)),
    )
    trace = o.PanelLocalOwner(
        o.OwnerId("owner_0002"),
        o.OwnerKind.TRACE,
        _region(1, 1),
        (figure.owner_id,),
    )
    segment = o.PanelLocalOwner(
        o.OwnerId("owner_0003"),
        o.OwnerKind.SEGMENT,
        _region(2, 2),
        (trace.owner_id,),
    )
    nested_inventory = o.OwnerInventory(
        _d("a"),
        _d("b"),
        o.EnumerationResolution.GRID16_FULL_PANEL,
        _d("c"),
        True,
        (figure, trace, segment),
    )
    assert o.descendant_segment_owner_ids(
        figure.owner_id, nested_inventory
    ) == (segment.owner_id,)
    segment_spec = o.PanelFeatureSpec(
        o.FeatureFamily.EXACT_SEGMENT_COUNT,
        o.SubjectScope.ONE_COHERENT_FIGURE,
        o.ReferenceFrame.NONE,
        o.ExactSegmentCountParameters(o.ClosedCount.ONE),
    )
    o.PanelFeatureWitness(
        segment_spec,
        nested_inventory,
        _d("d"),
        o.SubjectBinding(o.SubjectBindingKind.UNARY, (figure.owner_id,)),
        o.CountWitnessPayload((segment.owner_id,), True, _d("e")),
        _d("f"),
    )


def test_point_contact_signature_is_owner_labelled_and_canonical() -> None:
    inventory = _inventory()
    witness = _point_witness(inventory)
    assert o.PanelFeatureWitness.from_data(witness.to_data()) == witness
    rays = witness.payload.owner_rays
    with pytest.raises(o.PanelSoftOntologyError):
        o.PointContactWitnessPayload(
            o.PointContactKind.TANGENTIAL,
            o.QuantizedPoint(7, 1),
            tuple(reversed(rays)),
            witness.payload.exterior_gap_regions,
        )
    assert o.registered_sibling_relation(
        _point_spec(), _first_spec(o.FeatureFamily.VISIBLE_GAP)
    ) is None
    outside_payload = o.PointContactWitnessPayload(
        o.PointContactKind.TANGENTIAL,
        o.QuantizedPoint(6, 1),
        rays,
        witness.payload.exterior_gap_regions,
    )
    with pytest.raises(o.PanelSoftOntologyError):
        o.PanelFeatureWitness(
            witness.spec,
            inventory,
            witness.observer_contract_digest,
            witness.subject,
            outside_payload,
            witness.witness_receipt_digest,
        )


def test_absence_requires_exact_relation_binding_coverage() -> None:
    inventory = _relation_inventory()
    certificate = _pair_absence(inventory)
    assert len(certificate.eligible_subjects) == 3
    assert o.AbsenceCertificate.from_data(certificate.to_data()) == certificate
    with pytest.raises(o.PanelSoftOntologyError):
        o.AbsenceCertificate(
            certificate.target_spec,
            inventory,
            certificate.observer_contract_digest,
            certificate.search_domain,
            certificate.search_protocol_digest,
            certificate.eligible_subjects,
            certificate.rejections[:-1],
            True,
            certificate.search_receipt_digest,
        )
    wrong_subject = certificate.eligible_subjects[1]
    with pytest.raises(o.PanelSoftOntologyError):
        o.OwnerRejection(
            certificate.target_spec,
            inventory,
            certificate.observer_contract_digest,
            wrong_subject,
            o.ExhaustiveSearchNonmatchEvidence(
                (o.subject_search_region(certificate.eligible_subjects[0], inventory),),
                _d("f"),
            ),
        )
    with pytest.raises(o.PanelSoftOntologyError):
        o.AbsenceCertificate(
            certificate.target_spec,
            inventory,
            certificate.observer_contract_digest,
            certificate.search_domain,
            certificate.search_protocol_digest,
            certificate.eligible_subjects,
            certificate.rejections,
            False,
            certificate.search_receipt_digest,
        )


def test_empty_eligible_set_needs_independent_enumeration_evidence() -> None:
    figure = o.PanelLocalOwner(o.OwnerId("owner_0001"), o.OwnerKind.FIGURE, _region())
    inventory = o.OwnerInventory(
        _d("a"), _d("b"), o.EnumerationResolution.GRID16_FULL_PANEL, _d("c"), True, (figure,)
    )
    spec = _first_spec(o.FeatureFamily.OPEN_TRACE)
    domain = o.SearchResolutionDomain.for_spec(spec)
    assert o.eligible_subject_bindings(spec, inventory) == ()
    with pytest.raises(o.PanelSoftOntologyError):
        o.AbsenceCertificate(spec, inventory, _d("d"), domain, _d("1"), (), (), True, _d("2"))
    empty = o.EmptyEligibleDomainCertificate(
        inventory.inventory_digest, domain.domain_digest, inventory.enumeration_receipt_digest, _d("6")
    )
    certificate = o.AbsenceCertificate(
        spec, inventory, _d("d"), domain, _d("1"), (), (), True, _d("2"), empty
    )
    assert o.AbsenceCertificate.from_data(certificate.to_data()) == certificate


def test_registered_conflict_is_local_not_automatic_complement() -> None:
    inventory = _inventory()
    target = o.PanelFeatureSpec(
        o.FeatureFamily.COMPONENT_COUNT,
        o.SubjectScope.WHOLE_PANEL,
        o.ReferenceFrame.NONE,
        o.ComponentCountParameters(o.ClosedCount.ONE),
    )
    sibling = _first_spec(o.FeatureFamily.COMPONENT_COUNT)
    subject = o.SubjectBinding(o.SubjectBindingKind.PANEL, ())
    sibling_witness = o.PanelFeatureWitness(
        sibling,
        inventory,
        _d("d"),
        subject,
        o.CountWitnessPayload(
            (o.OwnerId("owner_0001"), o.OwnerId("owner_0002")), True, _d("5")
        ),
        _d("e"),
    )
    relation = o.registered_sibling_relation(target, sibling)
    assert relation is not None and relation.exhaustive is False
    rejection = o.OwnerRejection(
        target,
        inventory,
        _d("d"),
        subject,
        o.RegisteredSiblingConflictEvidence(relation.relation_id, sibling_witness),
    )
    raw = o.RawFeatureMeasurement(
        target,
        inventory,
        _d("d"),
        _d("7"),
        o.RawMeasurementState.REGISTERED_SIBLING_CONFLICT,
        local_conflict=rejection,
    )
    assert raw.state is o.RawMeasurementState.REGISTERED_SIBLING_CONFLICT
    marker_a = _first_spec(o.FeatureFamily.MARKER_PATTERN)
    marker_b = o.PanelFeatureSpec(
        o.FeatureFamily.MARKER_PATTERN,
        o.SubjectScope.ONE_COHERENT_FIGURE,
        o.ReferenceFrame.NONE,
        o.MarkerPatternParameters(
            o.MarkerPrimitive.CROSS, o.ClosedCount.THREE, o.MarkerArrangement.CLUSTERED
        ),
    )
    assert o.registered_sibling_relation(marker_a, marker_b) is None
    marker_row = next(
        item
        for item in o.feature_catalog_data()["sibling_registry"]
        if item["relation_id"] == "distinct-exact-counts-v1"
    )
    assert marker_row["parameter_rule_id"] == "distinct_count_same_marker_context_v1"
    assert "note" not in marker_row


def test_calibration_is_externally_pinned_capability_specific_and_time_bounded() -> None:
    inventory = _inventory()
    witness = _point_witness(inventory)
    absence = _pair_absence(inventory)
    domain = o.FeatureDomain(
        witness.spec.family,
        witness.spec.subject_scope,
        witness.spec.reference_frame,
        (witness.spec,),
    )
    presence = o.PresenceCalibrationGrant(
        domain, _d("d"), _d("7"), _assessment(o.CalibrationRisk.FALSE_POSITIVE_CLAIM, "8"), _d("9")
    )
    absence_grant = o.AbsenceCalibrationGrant(
        domain,
        _d("d"),
        _d("7"),
        inventory.enumeration_protocol_digest,
        absence.search_protocol_digest,
        _assessment(o.CalibrationRisk.FALSE_NEGATIVE_CLAIM, "a"),
        _assessment(o.CalibrationRisk.OWNER_INVENTORY_OMISSION, "b"),
        o.EnumerationResolution.GRID16_FULL_PANEL,
        (o.RejectionKind.EXHAUSTIVE_SEARCH_NONMATCH,),
        _d("9"),
    )
    authority = o.FeatureCalibrationAuthority(
        "calibration.primary.v1", domain, _d("d"), _d("c"), _d("e"), presence, absence_grant
    )
    assert o.FeatureCalibrationAuthority.from_data(authority.to_data()) == authority
    present_raw = o.RawFeatureMeasurement(
        witness.spec,
        inventory,
        _d("d"),
        _d("7"),
        o.RawMeasurementState.WITNESS_ASSERTED,
        witness=witness,
    )
    absent_raw = o.RawFeatureMeasurement(
        absence.target_spec,
        inventory,
        _d("d"),
        _d("7"),
        o.RawMeasurementState.EXHAUSTIVE_SEARCH_NEGATIVE,
        absence=absence,
    )
    present_token = o.verify_feature_calibration_authority(
        authority,
        capability=o.CalibrationCapability.PRESENCE,
        expected_authority_digest=authority.authority_digest,
        expected_grant_digest=presence.grant_digest,
        trusted_root_digest=_d("c"),
        verifier_receipt_digest=_d("f"),
        campaign_time_unix=150,
    )
    absent_token = o.verify_feature_calibration_authority(
        authority,
        capability=o.CalibrationCapability.ABSENCE,
        expected_authority_digest=authority.authority_digest,
        expected_grant_digest=absence_grant.grant_digest,
        trusted_root_digest=_d("c"),
        verifier_receipt_digest=_d("f"),
        campaign_time_unix=150,
    )
    present_custody = o.verify_raw_measurement_custody(
        present_raw,
        expected_measurement_digest=present_raw.measurement_digest,
        expected_inventory_digest=inventory.inventory_digest,
        expected_enumeration_receipt_digest=inventory.enumeration_receipt_digest,
        expected_evidence_receipt_digest=witness.witness_receipt_digest,
        verifier_receipt_digest=_d("6"),
    )
    absent_custody = o.verify_raw_measurement_custody(
        absent_raw,
        expected_measurement_digest=absent_raw.measurement_digest,
        expected_inventory_digest=inventory.inventory_digest,
        expected_enumeration_receipt_digest=inventory.enumeration_receipt_digest,
        expected_evidence_receipt_digest=absence.search_receipt_digest,
        verifier_receipt_digest=_d("6"),
    )
    assert o.project_raw_measurement(present_raw, present_token) is Disposition.INDETERMINATE
    assert o.project_raw_measurement(present_raw, present_token, present_custody) is Disposition.INDETERMINATE
    assert o.project_raw_measurement(absent_raw, absent_token, absent_custody) is Disposition.CERTIFIED_ABSENT
    assert o.project_raw_measurement(present_raw, absent_token, present_custody) is Disposition.INDETERMINATE
    assert o.project_raw_measurement(absent_raw, present_token, absent_custody) is Disposition.INDETERMINATE
    with pytest.raises(FrozenInstanceError):
        present_token.capability = o.CalibrationCapability.ABSENCE  # type: ignore[misc]
    bad_gap_payload = o.PointContactWitnessPayload(
        o.PointContactKind.TANGENTIAL,
        witness.payload.contact_point,
        witness.payload.owner_rays,
        (_region(0, 3), _region(3, 3)),
    )
    bad_gap_witness = o.PanelFeatureWitness(
        witness.spec,
        inventory,
        witness.observer_contract_digest,
        witness.subject,
        bad_gap_payload,
        witness.witness_receipt_digest,
    )
    bad_gap_raw = o.RawFeatureMeasurement(
        witness.spec,
        inventory,
        _d("d"),
        _d("7"),
        o.RawMeasurementState.WITNESS_ASSERTED,
        witness=bad_gap_witness,
    )
    bad_gap_custody = o.verify_raw_measurement_custody(
        bad_gap_raw,
        expected_measurement_digest=bad_gap_raw.measurement_digest,
        expected_inventory_digest=inventory.inventory_digest,
        expected_enumeration_receipt_digest=inventory.enumeration_receipt_digest,
        expected_evidence_receipt_digest=bad_gap_witness.witness_receipt_digest,
        verifier_receipt_digest=_d("6"),
    )
    assert (
        o.project_raw_measurement(bad_gap_raw, present_token, bad_gap_custody)
        is Disposition.INDETERMINATE
    )
    fake_token = o._VerifiedFeatureCalibrationAuthority(
        authority,
        o.CalibrationCapability.PRESENCE,
        authority.authority_digest,
        presence.grant_digest,
        _d("c"),
        _d("f"),
        150,
        object(),
    )
    with pytest.raises(TypeError):
        o.project_raw_measurement(present_raw, fake_token, present_custody)
    unresolved = o.RawFeatureMeasurement(
        witness.spec,
        inventory,
        _d("d"),
        _d("7"),
        o.RawMeasurementState.UNRESOLVED,
        issue_code=o.MeasurementIssueCode.AMBIGUOUS_OWNER,
    )
    error = o.RawFeatureMeasurement(
        witness.spec,
        inventory,
        _d("d"),
        _d("7"),
        o.RawMeasurementState.ERROR,
        issue_code=o.MeasurementIssueCode.OBSERVER_FAILURE,
    )
    assert o.project_raw_measurement(unresolved, absent_token) is Disposition.INDETERMINATE
    assert o.project_raw_measurement(error, absent_token) is Disposition.ERROR
    with pytest.raises(TypeError):
        o.project_raw_measurement(present_raw, authority)  # type: ignore[arg-type]
    with pytest.raises(o.PanelSoftOntologyError):
        o.verify_feature_calibration_authority(
            authority,
            capability=o.CalibrationCapability.PRESENCE,
            expected_authority_digest=authority.authority_digest,
            expected_grant_digest=presence.grant_digest,
            trusted_root_digest=_d("0"),
            verifier_receipt_digest=_d("f"),
            campaign_time_unix=150,
        )
    with pytest.raises(o.PanelSoftOntologyError):
        o.verify_feature_calibration_authority(
            authority,
            capability=o.CalibrationCapability.PRESENCE,
            expected_authority_digest=authority.authority_digest,
            expected_grant_digest=presence.grant_digest,
            trusted_root_digest=_d("c"),
            verifier_receipt_digest=_d("f"),
            campaign_time_unix=201,
        )


def test_generic_unary_payload_is_diagnostic_only_and_grid16_is_unambiguous() -> None:
    inventory = _inventory()
    spec = _first_spec(o.FeatureFamily.OPEN_TRACE)
    subject = o.eligible_subject_bindings(spec, inventory)[0]
    payload = o.UnaryGeometryWitnessPayload(
        o.FeatureFamily.OPEN_TRACE,
        _region(1, 1),
        (o.QuantizedPoint(1, 1), o.QuantizedPoint(2, 2)),
        o.WitnessCoverage.LOCAL,
        _d("5"),
    )
    witness = o.PanelFeatureWitness(spec, inventory, _d("d"), subject, payload, _d("e"))
    domain = o.FeatureDomain(
        spec.family, spec.subject_scope, spec.reference_frame, (spec,)
    )
    grant = o.PresenceCalibrationGrant(
        domain,
        _d("d"),
        _d("7"),
        _assessment(o.CalibrationRisk.FALSE_POSITIVE_CLAIM, "8"),
        _d("9"),
    )
    authority = o.FeatureCalibrationAuthority(
        "calibration.open.v1", domain, _d("d"), _d("c"), _d("e"), grant
    )
    token = o.verify_feature_calibration_authority(
        authority,
        capability=o.CalibrationCapability.PRESENCE,
        expected_authority_digest=authority.authority_digest,
        expected_grant_digest=grant.grant_digest,
        trusted_root_digest=_d("c"),
        verifier_receipt_digest=_d("f"),
        campaign_time_unix=150,
    )
    raw = o.RawFeatureMeasurement(
        spec,
        inventory,
        _d("d"),
        _d("7"),
        o.RawMeasurementState.WITNESS_ASSERTED,
        witness=witness,
    )
    custody = o.verify_raw_measurement_custody(
        raw,
        expected_measurement_digest=raw.measurement_digest,
        expected_inventory_digest=inventory.inventory_digest,
        expected_enumeration_receipt_digest=inventory.enumeration_receipt_digest,
        expected_evidence_receipt_digest=witness.witness_receipt_digest,
        verifier_receipt_digest=_d("6"),
    )
    assert o.project_raw_measurement(raw, token, custody) is Disposition.INDETERMINATE
    with pytest.raises(o.PanelSoftOntologyError):
        o.UnaryGeometryWitnessPayload(
            o.FeatureFamily.OPEN_TRACE,
            _region(1, 1),
            (o.QuantizedPoint(1, 1), o.QuantizedPoint(1, 1)),
            o.WitnessCoverage.LOCAL,
            _d("5"),
        )
    assert "q16" not in repr(o.feature_catalog_data()).lower()
    assert "q16" not in o.QUANTIZED_POINT_SCHEMA.lower()
    assert "q16" not in o.QUANTIZED_REGION_SCHEMA.lower()


def test_unresolved_error_and_closed_world_presence_never_become_absence() -> None:
    inventory = _inventory()
    spec = _point_spec()
    unresolved = o.RawFeatureMeasurement(
        spec,
        inventory,
        _d("d"),
        _d("7"),
        o.RawMeasurementState.UNRESOLVED,
        issue_code=o.MeasurementIssueCode.AMBIGUOUS_OWNER,
    )
    error = o.RawFeatureMeasurement(
        spec,
        inventory,
        _d("d"),
        _d("7"),
        o.RawMeasurementState.ERROR,
        issue_code=o.MeasurementIssueCode.OBSERVER_FAILURE,
    )
    assert o.RawFeatureMeasurement.from_data(unresolved.to_data()) == unresolved
    assert o.RawFeatureMeasurement.from_data(error.to_data()) == error
    with pytest.raises(o.PanelSoftOntologyError):
        o.CalibrationAssessment(
            o.CalibrationRisk.FALSE_POSITIVE_CLAIM,
            _d("a"),
            _d("b"),
            True,
            10,
            5,
            950_000,
            1,
            2,
            _d("c"),
        )


def test_language_gap_is_closed_and_roundtrips() -> None:
    gap = o.LanguageGapArtifact(
        o.LanguageGapKind.COMPLEMENT_DERIVATION_REQUESTED,
        _d("a"),
        _d("b"),
        _d("c"),
    )
    assert o.LanguageGapArtifact.from_data(gap.to_data()) == gap
    injected = dict(gap.to_data())
    injected["negated_spec"] = _d("d")
    with pytest.raises(o.PanelSoftOntologyError):
        o.LanguageGapArtifact.from_data(injected)
