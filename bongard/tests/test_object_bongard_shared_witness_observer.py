"""Exact tests for the Python-only shared-witness panel projection."""

from __future__ import annotations

from copy import deepcopy
import hashlib

import pytest

from bongard.canonical import canonical_digest
from bongard.evidence import Disposition
from bongard.object_bongard_shared_witness import (
    ObjectBongardSharedWitnessContrast,
    ObjectBongardSharedWitnessRubricSpec,
)
from bongard import object_bongard_shared_witness_observer as observer
from bongard.prototype_scene_observer import PrototypeSceneObserverStatus
from bongard.tests.test_prototype_scene_observer import (
    EFFORT,
    LAUNCHER_DIGEST,
    MODEL,
    NO_TOOLS_KWARGS,
    _png,
    _receipt,
)
from bongard.transport import CodexStructuredResult


FREEZE_DIGEST = "a" * 64


def _spec() -> ObjectBongardSharedWitnessRubricSpec:
    contrast = ObjectBongardSharedWitnessContrast.create(
        0,
        shared_anchor="patterned loop network",
        visual_axis="junction organization",
        group_0_endpoint="shared hub",
        group_1_endpoint="distributed junction",
    )
    return ObjectBongardSharedWitnessRubricSpec.from_contrast(
        "b" * 64,
        contrast,
    )


def _entity(
    spec: ObjectBongardSharedWitnessRubricSpec,
    index: int,
    *,
    target: str,
    foil: str,
    anchor: str = "clear",
) -> dict[str, object]:
    cues = observer._neutral_endpoint_cues(spec)
    target_id, foil_id = observer._endpoint_mapping(spec, cues)
    judgments = {target_id: target, foil_id: foil}
    return {
        "entity_id": f"e{index:02d}",
        "scope": "top_level_figure",
        "bbox_q16": {
            "x0": 1000 + index * 5000,
            "y0": 2000,
            "x1": 4500 + index * 5000,
            "y1": 9000,
        },
        "locator": f"visible figure {index}",
        "anchor_support": anchor,
        "anchor_evidence": "visible patterned connected loops",
        "cue_support": [
            {
                "cue_id": cue.cue_id,
                "judgment": judgments[cue.cue_id],
                "evidence": "visible junction arrangement",
            }
            for cue in cues
        ],
    }


def _project(
    spec: ObjectBongardSharedWitnessRubricSpec,
    entities: list[dict[str, object]],
    *,
    inventory_status: str = "complete",
) -> observer.ObjectBongardSharedWitnessPanelObservation:
    cues = observer._neutral_endpoint_cues(spec)
    return observer._project_frozen_payload(
        {"inventory_status": inventory_status, "entities": entities},
        rubric_spec=spec,
        endpoint_cues=cues,
        payload_freeze_digest=FREEZE_DIGEST,
    )


def _observe_payload(payload: dict[str, object]):
    panel = _png(31)
    spec = _spec()

    def transport(prompt, paths, names, schema, **kwargs):
        return CodexStructuredResult(
            payload,
            _receipt(prompt, paths, names, schema, payload),
        )

    artifact = observer.observe_object_bongard_shared_witness_panel(
        panel,
        panel_id="bd/shared_witness_observer_fixture/0.png",
        rubric_spec=spec,
        expected_panel_sha256=hashlib.sha256(panel).hexdigest(),
        expected_rubric_spec_digest=spec.spec_digest,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        transport=transport,
        **NO_TOOLS_KWARGS,
    )
    return artifact, panel, spec


def test_python_projection_certifies_present_and_round_trips() -> None:
    spec = _spec()
    result = _project(spec, [_entity(spec, 0, target="clear", foil="none")])

    assert result.disposition is Disposition.PRESENT
    assert result.payload_freeze_digest == FREEZE_DIGEST
    assert result.entities[0].target_interval == observer.BinarySupportInterval(1, 1)
    assert result.entities[0].foil_interval == observer.BinarySupportInterval(0, 0)
    assert result.entities[0].pixel_witness_ids == ()
    assert (
        observer.ObjectBongardSharedWitnessPanelObservation.from_data(
            result.to_data()
        )
        == result
    )


def test_python_projection_certifies_symmetric_absence() -> None:
    spec = _spec()
    result = _project(spec, [_entity(spec, 0, target="none", foil="clear")])

    assert result.disposition is Disposition.CERTIFIED_ABSENT


@pytest.mark.parametrize(
    "entities,inventory_status",
    [
        ([], "complete"),
        ([], "uncertain"),
    ],
)
def test_empty_or_uncertain_inventory_is_indeterminate(
    entities: list[dict[str, object]], inventory_status: str
) -> None:
    assert _project(
        _spec(), entities, inventory_status=inventory_status
    ).disposition is Disposition.INDETERMINATE


def test_separate_target_and_foil_figures_cannot_be_pooled_or_cherry_picked() -> None:
    spec = _spec()
    result = _project(
        spec,
        [
            _entity(spec, 0, target="clear", foil="none"),
            _entity(spec, 1, target="none", foil="clear"),
        ],
    )

    assert result.disposition is Disposition.INDETERMINATE
    assert tuple(item.entity_id for item in result.entities) == ("e00", "e01")


@pytest.mark.parametrize(
    ("target", "foil", "anchor"),
    [
        ("ambiguous", "none", "clear"),
        ("clear", "ambiguous", "clear"),
        ("clear", "none", "ambiguous"),
        ("none", "none", "clear"),
    ],
)
def test_ambiguity_and_failed_fit_never_become_a_boolean_negative(
    target: str, foil: str, anchor: str
) -> None:
    spec = _spec()
    result = _project(
        spec,
        [_entity(spec, 0, target=target, foil=foil, anchor=anchor)],
    )

    assert result.disposition is Disposition.INDETERMINATE


def test_projection_rejects_noncanonical_neutral_cue_order() -> None:
    spec = _spec()
    payload = {
        "inventory_status": "complete",
        "entities": [_entity(spec, 0, target="clear", foil="none")],
    }
    malformed = deepcopy(payload)
    malformed["entities"][0]["cue_support"].reverse()  # type: ignore[index]

    with pytest.raises(
        observer.ObjectBongardSharedWitnessObserverError,
        match="cue support order",
    ):
        observer._project_frozen_payload(
            malformed,
            rubric_spec=spec,
            endpoint_cues=observer._neutral_endpoint_cues(spec),
            payload_freeze_digest=FREEZE_DIGEST,
        )


def test_serialized_policy_makes_lean_optional_and_decision_inert() -> None:
    spec = _spec()
    result = _project(spec, [_entity(spec, 0, target="clear", foil="none")])
    data = result.to_data()

    assert "lean" not in repr(result).casefold()
    assert data["projection_id"] == "all-entities-anchor-endpoint-meet-v1"
    protocol = observer._authority_data()
    assert protocol["python_is_canonical_authority"] is True
    assert protocol["lean_present"] is False
    assert protocol["lean_required"] is False
    assert protocol["lean_removable"] is True
    assert protocol["lean_defines_identity_or_decision"] is False


def test_full_panel_scope_is_unrepresentable_and_projection_rejects_it() -> None:
    spec = _spec()
    schema = observer.object_bongard_shared_witness_panel_output_schema(spec)
    scope_schema = schema["properties"]["entities"]["items"]["properties"][  # type: ignore[index]
        "scope"
    ]

    assert scope_schema["enum"] == ["top_level_figure"]  # type: ignore[index]
    assert "full_panel" not in repr(schema)

    payload_entity = _entity(spec, 0, target="clear", foil="none")
    payload_entity["scope"] = "full_panel"
    payload_entity["bbox_q16"] = {
        "x0": 0,
        "y0": 0,
        "x1": 65535,
        "y1": 65535,
    }
    with pytest.raises(
        observer.ObjectBongardSharedWitnessObserverError,
        match="entity scope differs",
    ):
        _project(spec, [payload_entity])


def test_prompt_requires_uncertain_inventory_on_eight_entity_overflow() -> None:
    prompt = observer.object_bongard_shared_witness_panel_prompt(_spec())

    assert "if more than eight exist" in prompt
    assert "return those first eight" in prompt
    assert "set inventory_status to uncertain" in prompt
    assert observer._authority_data()["overflow_requires_uncertain_inventory"] is True


def test_parser_error_replay_requires_a_real_exact_deterministic_failure() -> None:
    spec = _spec()
    valid_payload = {
        "inventory_status": "complete",
        "entities": [_entity(spec, 0, target="clear", foil="none")],
    }
    success, _, _ = _observe_payload(valid_payload)
    assert success.status is PrototypeSceneObserverStatus.SUCCESS
    assert success.observation.disposition is Disposition.PRESENT

    forged = deepcopy(success.to_data())
    forged["status"] = PrototypeSceneObserverStatus.PARSER_ERROR.value
    forged["failure_code"] = "observer_payload_rejected"
    forged["failure_type"] = "ForgedParserError"
    forged["observation"] = (
        observer.ObjectBongardSharedWitnessPanelObservation.error(
            spec.spec_digest,
            "observer_payload_rejected",
            "ForgedParserError",
        ).to_data()
    )
    forged["artifact_digest"] = canonical_digest(
        {key: value for key, value in forged.items() if key != "artifact_digest"}
    )
    with pytest.raises(
        observer.ObjectBongardSharedWitnessObserverError,
        match="parser failure payload projects successfully",
    ):
        observer.ObjectBongardSharedWitnessPanelArtifact.from_data(forged)

    malformed_payload = {
        "inventory_status": "complete",
        "entities": [],
        "unexpected": "field",
    }
    parser_error, _, malformed_spec = _observe_payload(malformed_payload)
    assert parser_error.status is PrototypeSceneObserverStatus.PARSER_ERROR
    assert parser_error.observation.disposition is Disposition.ERROR
    assert (
        observer.ObjectBongardSharedWitnessPanelArtifact.from_data(
            parser_error.to_data()
        )
        == parser_error
    )

    wrong_type = deepcopy(parser_error.to_data())
    wrong_type["failure_type"] = "ValueError"
    wrong_type["observation"] = (
        observer.ObjectBongardSharedWitnessPanelObservation.error(
            malformed_spec.spec_digest,
            "observer_payload_rejected",
            "ValueError",
        ).to_data()
    )
    wrong_type["artifact_digest"] = canonical_digest(
        {key: value for key, value in wrong_type.items() if key != "artifact_digest"}
    )
    with pytest.raises(
        observer.ObjectBongardSharedWitnessObserverError,
        match="parser failure type differs from deterministic replay",
    ):
        observer.ObjectBongardSharedWitnessPanelArtifact.from_data(wrong_type)
