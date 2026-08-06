from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
from PIL import Image
import pytest

from bongard.evidence import Disposition
from bongard.ir import Atom, Relation
from bongard.proposer import (
    HYBRID_EPISTEMIC_STATUS,
    HeadlessCodexEpisode,
    HybridClaim,
    HybridCue,
    ProposalError,
    _parse_hybrid_observation,
    hybrid_observer_prompt,
    observe_hybrid_panel,
    parse_hybrid_observation_or_error,
    parse_rule_proposal,
    propose_pure_rule,
    proposer_prompt,
    propose_rule,
    pure_only_rule_proposal_schema,
    pure_proposer_prompt,
)
from bongard.benchmark import SupportInput
from bongard.synthesis import compile_hybrid_proposal
from bongard.transport import CloudPolicyCacheSnapshot, semantic_panel_set_digest


@dataclass(frozen=True)
class FakeReceipt:
    receipt_digest: str = "receipt-digest"
    input_digest: str = "input-digest"
    thread_id: str = "thread-1"
    requested_model: str = "gpt-test"
    requested_reasoning_effort: str = "medium"
    codex_launcher_digest: str = "launcher-digest"
    codex_cli_version: str = "codex-cli test"
    cloud_config_bundle_cache_binding: str = "absent"
    isolation_policy: str = "fixture-isolation"

    def to_dict(self) -> dict[str, str]:
        return {
            "receipt_digest": self.receipt_digest,
            "input_digest": self.input_digest,
            "thread_id": self.thread_id,
            "requested_model": self.requested_model,
            "requested_reasoning_effort": self.requested_reasoning_effort,
            "codex_launcher_digest": self.codex_launcher_digest,
            "codex_cli_version": self.codex_cli_version,
            "cloud_config_bundle_cache_binding": self.cloud_config_bundle_cache_binding,
            "isolation_policy": self.isolation_policy,
        }


def hybrid_payload() -> dict[str, Any]:
    return {
        "positive_description": "one closed form has a distinct inward notch",
        "panel_descriptions": {
            **{f"pos_{i}": f"notched form {i}" for i in range(6)},
            **{f"neg_{i}": f"other form {i}" for i in range(6)},
        },
        "view": "carrier_shape",
        "observable_requests": [],
        "formula_template": {"kind": "all", "atoms": ["hybrid_claim"]},
        "hybrid_claim": {
            "epistemic_status": HYBRID_EPISTEMIC_STATUS,
            "phrase": "notched trapezoid-like form",
            "operational_definition": (
                "a single closed polygon with a shallow inward V-shaped notch "
                "on one otherwise long side"
            ),
            "required_visual_cues": [
                {
                    "cue_id": "closed_polygon",
                    "positive_description": "one closed polygonal boundary",
                },
                {
                    "cue_id": "inward_v_notch",
                    "positive_description": "one inward V-shaped notch",
                },
            ],
        },
        "confidence": "medium",
    }


def pure_payload(observable_id: str = "prototype.topology") -> dict[str, Any]:
    payload = hybrid_payload()
    payload["hybrid_claim"] = None
    payload["observable_requests"] = [
        {
            "observable_id": observable_id,
            "affirmative_interpretation": (
                "the selected component-and-hole topology is present"
            ),
            "arguments": {},
        }
    ]
    payload["formula_template"] = {"kind": "all", "atoms": [observable_id]}
    return payload


def present_observation_payload() -> dict[str, Any]:
    return {
        "epistemic_status": HYBRID_EPISTEMIC_STATUS,
        "disposition": "present",
        "observed_cue_ids": ["closed_polygon", "inward_v_notch"],
        "missing_cue_ids": [],
        "missing_cue_reasons": [],
        "visibility_certificate": None,
        "reason": None,
        "error_type": None,
    }


def absent_observation_payload() -> dict[str, Any]:
    return {
        "epistemic_status": HYBRID_EPISTEMIC_STATUS,
        "disposition": "nonmatch",
        "observed_cue_ids": ["closed_polygon"],
        "missing_cue_ids": ["inward_v_notch"],
        "missing_cue_reasons": [
            {
                "cue_id": "inward_v_notch",
                "finding": "the fully visible boundary turns outward at every corner",
            }
        ],
        "visibility_certificate": "the entire polygon boundary is visible",
        "reason": "the visible boundary has the other fully visible corner pattern",
        "error_type": None,
    }


def write_png(path: Path, marker: int, *, rgb: bool = False) -> bytes:
    panel = np.full((12, 12), 255, dtype=np.uint8)
    panel[marker % 10 + 1, 1:11] = 0
    image = np.repeat(panel[..., None], 3, axis=2) if rgb else panel
    Image.fromarray(image, mode="RGB" if rgb else "L").save(path, format="PNG")
    return path.read_bytes()


def test_hybrid_proposal_is_grounded_in_all_twelve_descriptions() -> None:
    proposal = parse_rule_proposal(
        hybrid_payload(), receipt=FakeReceipt(), observable_catalog={}  # type: ignore[arg-type]
    )
    assert proposal.is_hybrid
    assert len(proposal.panel_descriptions) == 12
    assert proposal.formula_atoms == ("hybrid_claim",)
    assert proposal.hybrid_claim is not None
    assert "inward V-shaped notch" in proposal.hybrid_claim.operational_definition
    assert proposal.hybrid_claim.epistemic_status == HYBRID_EPISTEMIC_STATUS
    assert proposal.hybrid_claim.required_cue_ids == (
        "closed_polygon",
        "inward_v_notch",
    )
    assert proposal.hybrid_claim.to_dict()["required_visual_cues"][0] == {
        "cue_id": "closed_polygon",
        "positive_description": "one closed polygonal boundary",
    }
    assert proposal.content_dict()["schema"].endswith(".v3")


@pytest.mark.parametrize(
    "negative_language",
    (
        "a form with no wings",
        "a form that is not round",
        "a form without a tail",
        "a form that lacks a tail",
        "the absence of a tail",
        "a form with an absent tail",
        "a form with a missing tail",
        "neither a circle nor a square",
        "none of the corners are sharp",
        "a form that never closes",
        "a form that cannot rotate",
        "a form that isn't round",
        "a form that doesnt close",
        "a non-bird form",
        "a wingless form",
        "a form free of corners",
        "a form devoid of corners",
        "an empty circle",
        "an isolated circle",
        "a form omitting one corner",
        "a form excluding triangles",
        "every shape except circles",
        "a form avoiding acute angles",
        "a form that fails to close",
        "a form with fewer corners",
        "a form with less ink",
        "a form with zero crossings",
        "a form with only one corner",
        "a form with at most three corners",
        "a form with at‑most three corners",
        "a form not as tall as it is wide",
        "a form unlike the negative panels",
        "a form matching the positive examples",
        "a circle rather than a square",
        "a circle instead of a square",
        "every shape other than triangles",
        "a circle different from its frame",
        "a circle outside a square",
        "a form opposite of a triangle",
        "a form compared with the negative examples",
        "a circle in contrast to a square",
        "a circle versus a square",
        "a form smaller in area than its frame",
        "a form with a reduced number of corners",
        "a form with angle < 90 degrees",
        "a form with angle ≤ 90 degrees",
        "a form with angle != 90 degrees",
        "a form with angle ≠ 90 degrees",
    ),
)
def test_hybrid_claim_rejects_explicit_and_hidden_semantic_negation(
    negative_language: str,
) -> None:
    payload = hybrid_payload()
    payload["hybrid_claim"]["phrase"] = negative_language
    with pytest.raises(ProposalError, match="semantic negation"):
        parse_rule_proposal(
            payload,
            receipt=FakeReceipt(),  # type: ignore[arg-type]
            observable_catalog={},
        )


@pytest.mark.parametrize(
    ("location", "value"),
    (
        ("definition", "one body without a beak"),
        ("cue_description", "a body lacking a beak"),
        ("cue_id", "missing_beak"),
    ),
)
def test_negation_filter_covers_definition_cue_description_and_cue_id(
    location: str, value: str
) -> None:
    payload = hybrid_payload()
    claim = payload["hybrid_claim"]
    if location == "definition":
        claim["operational_definition"] = value
    elif location == "cue_description":
        claim["required_visual_cues"][0]["positive_description"] = value
    else:
        claim["required_visual_cues"][0]["cue_id"] = value
    with pytest.raises(ProposalError, match="semantic negation"):
        parse_rule_proposal(
            payload,
            receipt=FakeReceipt(),  # type: ignore[arg-type]
            observable_catalog={},
        )


@pytest.mark.parametrize(
    ("descriptor", "definition", "cue_id", "cue_description"),
    (
        (
            "asymmetric form",
            "one closed form with visibly distinct left and right extents",
            "distinct_side_extents",
            "left and right extents with visibly distinct lengths",
        ),
        (
            "irregular polygon",
            "one polygon with a visibly varied sequence of edge lengths and angles",
            "varied_edges_angles",
            "a varied sequence of visible edge lengths and corner angles",
        ),
        (
            "unbalanced two-lobed form",
            "one central form with two visibly different lobe areas",
            "different_lobe_areas",
            "two visible lobes with distinctly different areas",
        ),
        (
            "unequal paired circles",
            "two visible circles with distinctly different diameters",
            "different_circle_diameters",
            "two visible circular diameters with distinctly different lengths",
        ),
    ),
)
def test_intrinsic_constructive_descriptors_compile_to_positive_witness_atoms(
    descriptor: str,
    definition: str,
    cue_id: str,
    cue_description: str,
) -> None:
    payload = hybrid_payload()
    payload["hybrid_claim"] = {
        "epistemic_status": HYBRID_EPISTEMIC_STATUS,
        "phrase": descriptor,
        "operational_definition": definition,
        "required_visual_cues": [
            {"cue_id": cue_id, "positive_description": cue_description}
        ],
    }
    proposal = parse_rule_proposal(
        payload,
        receipt=FakeReceipt(),  # type: ignore[arg-type]
        observable_catalog={},
    )
    compiled = compile_hybrid_proposal(proposal)
    assert isinstance(compiled.formula, Atom)
    assert compiled.formula.relation is Relation.PRESENT
    assert compiled.formula.to_data()["type"] == "atom"


@pytest.mark.parametrize(
    "forbidden",
    (
        "a form that is not symmetric",
        "a form without bilateral symmetry",
        "asymmetric compared with the negative panels",
        "irregular unlike the positive examples",
        "unbalanced on the positive support side",
        "unequal across the negative support images",
        "asymmetric in the positive support",
        "an unfilled circle",
        "an unclosed loop",
        "a form whose symmetry is false",
        "a form that fails symmetry",
        "asymmetric on positives but symmetric on negatives",
        "asymmetric relative to the other class",
        "irregular when set against class B",
    ),
)
def test_constructive_descriptors_do_not_open_a_negation_or_support_comparison_escape(
    forbidden: str,
) -> None:
    payload = hybrid_payload()
    payload["hybrid_claim"]["phrase"] = forbidden
    with pytest.raises(ProposalError, match="semantic negation"):
        parse_rule_proposal(
            payload,
            receipt=FakeReceipt(),  # type: ignore[arg-type]
            observable_catalog={},
        )


def test_positive_soft_bird_like_and_oblique_claims_remain_representable() -> None:
    payload = hybrid_payload()
    payload["hybrid_claim"] = {
        "epistemic_status": HYBRID_EPISTEMIC_STATUS,
        "phrase": "bird-like object with oblique angles",
        "operational_definition": (
            "one central body with two lateral wing-like lobes, a forward "
            "beak-like point, and visibly oblique boundary junctions"
        ),
        "required_visual_cues": [
            {
                "cue_id": "bird_like_body",
                "positive_description": "a central body with two lateral wing-like lobes",
            },
            {
                "cue_id": "beak_like_point",
                "positive_description": "a forward beak-like point",
            },
            {
                "cue_id": "oblique_junctions",
                "positive_description": "boundary segments meeting at oblique angles",
            },
        ],
    }
    proposal = parse_rule_proposal(
        payload,
        receipt=FakeReceipt(),  # type: ignore[arg-type]
        observable_catalog={},
    )
    assert proposal.hybrid_claim is not None
    assert proposal.hybrid_claim.required_cue_ids == (
        "bird_like_body",
        "beak_like_point",
        "oblique_junctions",
    )


def test_v1_free_text_cues_and_duplicate_or_invalid_v2_ids_fail_closed() -> None:
    old = hybrid_payload()
    old["hybrid_claim"].pop("epistemic_status")
    old["hybrid_claim"]["required_visual_cues"] = ["closed polygon", "notch"]
    with pytest.raises(ProposalError):
        parse_rule_proposal(
            old,
            receipt=FakeReceipt(),  # type: ignore[arg-type]
            observable_catalog={},
        )

    duplicate = hybrid_payload()
    duplicate["hybrid_claim"]["required_visual_cues"][1]["cue_id"] = "closed_polygon"
    with pytest.raises(ProposalError, match="duplicate cue IDs"):
        parse_rule_proposal(
            duplicate,
            receipt=FakeReceipt(),  # type: ignore[arg-type]
            observable_catalog={},
        )

    duplicate_description = hybrid_payload()
    duplicate_description["hybrid_claim"]["required_visual_cues"][1][
        "positive_description"
    ] = "ONE CLOSED POLYGONAL BOUNDARY"
    with pytest.raises(ProposalError, match="duplicate positive cue descriptions"):
        parse_rule_proposal(
            duplicate_description,
            receipt=FakeReceipt(),  # type: ignore[arg-type]
            observable_catalog={},
        )

    invalid = hybrid_payload()
    invalid["hybrid_claim"]["required_visual_cues"][0]["cue_id"] = "Closed_polygon"
    with pytest.raises(ProposalError, match="invalid cue ID"):
        parse_rule_proposal(
            invalid,
            receipt=FakeReceipt(),  # type: ignore[arg-type]
            observable_catalog={},
        )

    unicode_id = hybrid_payload()
    unicode_id["hybrid_claim"]["required_visual_cues"][0]["cue_id"] = "ｃlosed_polygon"
    with pytest.raises(ProposalError, match="invalid cue ID"):
        parse_rule_proposal(
            unicode_id,
            receipt=FakeReceipt(),  # type: ignore[arg-type]
            observable_catalog={},
        )

    padded = hybrid_payload()
    padded["hybrid_claim"]["required_visual_cues"][0]["cue_id"] = " closed_polygon"
    with pytest.raises(ProposalError, match="surrounding whitespace"):
        parse_rule_proposal(
            padded,
            receipt=FakeReceipt(),  # type: ignore[arg-type]
            observable_catalog={},
        )


def test_direct_hybrid_values_cannot_bypass_affirmative_cue_contract() -> None:
    with pytest.raises(ProposalError, match="semantic negation"):
        HybridCue("missing_wing", "one wing-shaped lobe")
    cue = HybridCue("wing_lobe", "one wing-shaped lobe")
    with pytest.raises(ProposalError, match="semantic negation"):
        HybridClaim("object without a beak", "one central body", (cue,))
    with pytest.raises(ProposalError, match="duplicate cue IDs"):
        HybridClaim("bird-like object", "one central body", (cue, cue))


def test_pure_proposal_cannot_flip_or_attach_decorative_observables() -> None:
    payload = hybrid_payload()
    payload["hybrid_claim"] = None
    payload["observable_requests"] = [
        {
            "observable_id": "geometry.oblique_angle",
            "affirmative_interpretation": "contains a certified oblique angle",
            "arguments": {},
        }
    ]
    payload["formula_template"] = {
        "kind": "all",
        "atoms": ["geometry.oblique_angle", "negated_rescue"],
    }
    with pytest.raises(ProposalError, match="load-bearing"):
        parse_rule_proposal(
            payload,
            receipt=FakeReceipt(),  # type: ignore[arg-type]
            observable_catalog={"geometry.oblique_angle": "an angle in degrees"},
        )
    payload["formula_template"] = {"kind": "not", "atoms": ["geometry.oblique_angle"]}
    with pytest.raises(ProposalError, match="kind must be 'all'"):
        parse_rule_proposal(
            payload,
            receipt=FakeReceipt(),  # type: ignore[arg-type]
            observable_catalog={"geometry.oblique_angle": "an angle in degrees"},
        )


def test_pure_only_schema_and_prompt_close_the_hybrid_escape_hatch() -> None:
    catalog = {
        "prototype.global_geometry": "support-relative global geometry",
        "prototype.topology": "support-relative component and hole topology",
    }
    schema = pure_only_rule_proposal_schema(catalog)
    properties = schema["properties"]

    assert properties["hybrid_claim"] == {"type": "null"}
    assert properties["observable_requests"]["minItems"] == 1
    assert properties["observable_requests"]["maxItems"] == 1
    assert properties["observable_requests"]["items"]["properties"][
        "observable_id"
    ]["enum"] == sorted(catalog)
    assert properties["formula_template"]["properties"]["atoms"]["items"][
        "enum"
    ] == sorted(catalog)

    prompt = pure_proposer_prompt(catalog)
    assert "finite verifier-frozen catalog" in prompt
    assert "Choose exactly one catalog ID" in prompt
    assert "HYBRID and arbitrary prose predicates are unavailable" in prompt
    assert "Set `hybrid_claim` to null" in prompt
    with pytest.raises(ProposalError, match="non-empty catalog"):
        pure_only_rule_proposal_schema({})


def test_pure_interpretation_cannot_smuggle_a_polarity_flip() -> None:
    payload = pure_payload()
    payload["observable_requests"][0]["affirmative_interpretation"] = (
        "not like the negative support panels"
    )
    with pytest.raises(ProposalError, match="semantic negation"):
        parse_rule_proposal(
            payload,
            receipt=FakeReceipt(),  # type: ignore[arg-type]
            observable_catalog={
                "prototype.topology": "support-relative topology"
            },
        )


@pytest.mark.parametrize(
    "raw_atom",
    ("not_hybrid_claim", "negated_rule", "anything", "NOT(hybrid_claim)"),
)
def test_hybrid_formula_cannot_launder_an_arbitrary_or_negative_atom_label(
    raw_atom: str,
) -> None:
    payload = hybrid_payload()
    payload["formula_template"] = {"kind": "all", "atoms": [raw_atom]}
    with pytest.raises(ProposalError, match="exactly the positive hybrid_claim"):
        parse_rule_proposal(
            payload,
            receipt=FakeReceipt(),  # type: ignore[arg-type]
            observable_catalog={},
        )

def test_proposer_transport_sees_only_exact_support_bytes(tmp_path: Path) -> None:
    positives: list[Path] = []
    negatives: list[Path] = []
    expected: dict[str, bytes] = {}
    for side, collection in (("pos", positives), ("neg", negatives)):
        for index in range(6):
            path = tmp_path / f"source-{side}-{index}.png"
            data = write_png(path, index + (0 if side == "pos" else 6), rgb=True)
            collection.append(path)
            expected[f"{side}_{index}.png"] = data
    query = tmp_path / "never-open-query.png"
    write_png(query, 11, rgb=True)

    calls: list[tuple[str, ...]] = []
    schemas: list[Mapping[str, Any]] = []

    def fake_transport(prompt, paths, schema, **kwargs):
        del prompt, kwargs
        schemas.append(schema)
        canonical = tuple(paths)
        calls.append(canonical)
        assert tuple(Path(path).name for path in canonical) == tuple(expected)
        assert {Path(path).read_bytes() for path in canonical} == set(expected.values())
        assert str(query) not in canonical
        return SimpleNamespace(payload=hybrid_payload(), receipt=FakeReceipt())

    proposal = propose_rule(
        positives,
        negatives,
        observable_catalog={},
        transport=fake_transport,
    )
    assert proposal.is_hybrid
    assert len(calls) == 1
    hybrid_schema = schemas[0]
    assert hybrid_schema["properties"]["observable_requests"]["maxItems"] == 0
    assert "anyOf" not in hybrid_schema["properties"]["hybrid_claim"]
    assert hybrid_schema["properties"]["formula_template"]["properties"][
        "atoms"
    ] == {
        "type": "array",
        "items": {"type": "string", "const": "hybrid_claim"},
        "minItems": 1,
        "maxItems": 1,
    }


def test_pure_transport_is_one_support_only_closed_catalog_turn(
    tmp_path: Path,
) -> None:
    positives: list[Path] = []
    negatives: list[Path] = []
    expected: dict[str, bytes] = {}
    for side, collection in (("pos", positives), ("neg", negatives)):
        for index in range(6):
            path = tmp_path / f"pure-source-{side}-{index}.png"
            expected[f"{side}_{index}.png"] = write_png(
                path, index + (0 if side == "pos" else 6), rgb=True
            )
            collection.append(path)

    calls = 0
    catalog = {"prototype.topology": "support-relative topology"}

    def fake_transport(prompt, paths, schema, **kwargs):
        nonlocal calls
        del kwargs
        calls += 1
        assert "Set `hybrid_claim` to null" in prompt
        assert tuple(Path(path).name for path in paths) == tuple(expected)
        assert {Path(path).read_bytes() for path in paths} == set(expected.values())
        assert schema["properties"]["hybrid_claim"] == {"type": "null"}
        return SimpleNamespace(payload=pure_payload(), receipt=FakeReceipt())

    proposal = propose_pure_rule(
        positives,
        negatives,
        observable_catalog=catalog,
        transport=fake_transport,
    )
    assert calls == 1
    assert not proposal.is_hybrid
    assert proposal.formula_atoms == ("prototype.topology",)
    assert proposal.observable_requests[0].arguments == ()


def test_official_binary_rgb_panels_have_a_semantic_digest(tmp_path: Path) -> None:
    paths = []
    for side in ("pos", "neg"):
        for index in range(6):
            path = tmp_path / f"{side}_{index}.png"
            write_png(path, index, rgb=True)
            paths.append(str(path))
    digest = semantic_panel_set_digest(paths)
    assert digest.startswith("sha256:") and len(digest) == 71


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {
                "epistemic_status": HYBRID_EPISTEMIC_STATUS,
                "disposition": "present",
                "observed_cue_ids": ["closed_polygon", "inward_v_notch"],
                "missing_cue_ids": [],
                "missing_cue_reasons": [],
                "visibility_certificate": None,
                "reason": None,
                "error_type": None,
            },
            Disposition.PRESENT,
        ),
        (
            {
                "epistemic_status": HYBRID_EPISTEMIC_STATUS,
                "disposition": "nonmatch",
                "observed_cue_ids": ["closed_polygon"],
                "missing_cue_ids": ["inward_v_notch"],
                "missing_cue_reasons": [
                    {
                        "cue_id": "inward_v_notch",
                        "finding": "the fully visible boundary turns outward at every corner",
                    }
                ],
                "visibility_certificate": "the entire polygon boundary is visible",
                "reason": None,
                "error_type": None,
            },
            Disposition.CERTIFIED_ABSENT,
        ),
        (
            {
                "epistemic_status": HYBRID_EPISTEMIC_STATUS,
                "disposition": "indeterminate",
                "observed_cue_ids": [],
                "missing_cue_ids": [],
                "missing_cue_reasons": [],
                "visibility_certificate": None,
                "reason": "the bend is borderline",
                "error_type": None,
            },
            Disposition.INDETERMINATE,
        ),
        (
            {
                "epistemic_status": HYBRID_EPISTEMIC_STATUS,
                "disposition": "error",
                "observed_cue_ids": [],
                "missing_cue_ids": [],
                "missing_cue_reasons": [],
                "visibility_certificate": None,
                "reason": "the supplied image could not be decoded",
                "error_type": "ImageDecodeError",
            },
            Disposition.ERROR,
        ),
    ],
)
def test_hybrid_observer_has_four_way_nonboolean_boundary(
    tmp_path: Path, payload: dict[str, Any], expected: Disposition
) -> None:
    proposal = parse_rule_proposal(
        hybrid_payload(), receipt=FakeReceipt(), observable_catalog={}  # type: ignore[arg-type]
    )
    panel = tmp_path / "source.png"
    source_bytes = write_png(panel, 2, rgb=True)

    def fake_transport(prompt, paths, names, schema, **kwargs):
        del prompt, schema, kwargs
        assert names == ("query.png",)
        assert Path(paths[0]).read_bytes() == source_bytes
        return SimpleNamespace(payload=payload, receipt=FakeReceipt(thread_id="observer"))

    observed = observe_hybrid_panel(proposal, panel, transport=fake_transport)
    assert observed.evidence.disposition is expected
    with pytest.raises(TypeError, match="four dispositions"):
        bool(observed.evidence)


def test_present_uses_exact_declared_ids_and_canonical_claim_order() -> None:
    proposal = parse_rule_proposal(
        hybrid_payload(), receipt=FakeReceipt(), observable_catalog={}  # type: ignore[arg-type]
    )
    payload = present_observation_payload()
    payload["observed_cue_ids"] = ["inward_v_notch", "closed_polygon"]
    observation = _parse_hybrid_observation(
        proposal, payload, FakeReceipt(thread_id="observer")  # type: ignore[arg-type]
    )
    assert observation.observed_cue_ids == ("inward_v_notch", "closed_polygon")
    assert observation.missing_cue_ids == ()
    assert observation.evidence.value == ("closed_polygon", "inward_v_notch")
    assert observation.epistemic_status == HYBRID_EPISTEMIC_STATUS
    assert observation.to_dict()["epistemic_status"] == HYBRID_EPISTEMIC_STATUS


def test_observer_rejects_invented_mismatched_overlapping_or_duplicate_cue_ids() -> None:
    proposal = parse_rule_proposal(
        hybrid_payload(), receipt=FakeReceipt(), observable_catalog={}  # type: ignore[arg-type]
    )

    invented = present_observation_payload()
    invented["observed_cue_ids"] = ["closed_polygon", "invented_notch"]

    description_as_id = present_observation_payload()
    description_as_id["observed_cue_ids"] = [
        "closed_polygon",
        "one inward V-shaped notch",
    ]

    overlap = absent_observation_payload()
    overlap["observed_cue_ids"] = ["closed_polygon", "inward_v_notch"]

    duplicate = present_observation_payload()
    duplicate["observed_cue_ids"] = [
        "closed_polygon",
        "inward_v_notch",
        "inward_v_notch",
    ]

    mismatched_reason = absent_observation_payload()
    mismatched_reason["missing_cue_reasons"] = [
        {
            "cue_id": "closed_polygon",
            "finding": "the boundary is visibly open",
        }
    ]

    invented_missing = absent_observation_payload()
    invented_missing["missing_cue_ids"] = ["invented_notch"]

    padded = present_observation_payload()
    padded["observed_cue_ids"] = ["closed_polygon", " inward_v_notch"]

    wrong_case = present_observation_payload()
    wrong_case["observed_cue_ids"] = ["closed_polygon", "Inward_v_notch"]

    unicode_equivalent = present_observation_payload()
    unicode_equivalent["observed_cue_ids"] = ["closed_polygon", "ｉnward_v_notch"]

    cases = (
        (invented, "undeclared cue ID"),
        (description_as_id, "invalid cue ID"),
        (overlap, "must be disjoint"),
        (duplicate, "contains duplicates"),
        (mismatched_reason, "exactly matched"),
        (invented_missing, "undeclared cue ID"),
        (padded, "surrounding whitespace"),
        (wrong_case, "invalid cue ID"),
        (unicode_equivalent, "invalid cue ID"),
    )
    for payload, match in cases:
        with pytest.raises(ProposalError, match=match):
            _parse_hybrid_observation(
                proposal,
                payload,
                FakeReceipt(thread_id="observer"),  # type: ignore[arg-type]
            )


def test_present_iff_all_required_cues_observed_and_absence_is_structured() -> None:
    proposal = parse_rule_proposal(
        hybrid_payload(), receipt=FakeReceipt(), observable_catalog={}  # type: ignore[arg-type]
    )

    incomplete_present = present_observation_payload()
    incomplete_present["observed_cue_ids"] = ["closed_polygon"]

    all_observed_but_indeterminate = present_observation_payload()
    all_observed_but_indeterminate["disposition"] = "indeterminate"
    all_observed_but_indeterminate["reason"] = "claimed ambiguity"

    absence_without_missing = absent_observation_payload()
    absence_without_missing["missing_cue_ids"] = []
    absence_without_missing["missing_cue_reasons"] = []

    absence_without_reason = absent_observation_payload()
    absence_without_reason["missing_cue_reasons"] = []

    absence_without_visibility = absent_observation_payload()
    absence_without_visibility["visibility_certificate"] = None

    reason_subset = absent_observation_payload()
    reason_subset["observed_cue_ids"] = []
    reason_subset["missing_cue_ids"] = ["closed_polygon", "inward_v_notch"]

    reason_superset = absent_observation_payload()
    reason_superset["missing_cue_reasons"] = [
        *reason_superset["missing_cue_reasons"],
        {
            "cue_id": "closed_polygon",
            "finding": "the fully visible boundary is open",
        },
    ]

    indeterminate_with_missing = present_observation_payload()
    indeterminate_with_missing.update(
        {
            "disposition": "indeterminate",
            "observed_cue_ids": [],
            "missing_cue_ids": ["inward_v_notch"],
            "reason": "the image is partly occluded",
        }
    )

    cases = (
        (incomplete_present, "present requires all"),
        (all_observed_but_indeterminate, "empty observed/missing"),
        (absence_without_missing, "nonmatch requires"),
        (absence_without_reason, "exactly matched"),
        (absence_without_visibility, "visibility statement"),
        (reason_subset, "exactly matched"),
        (reason_superset, "exactly matched"),
        (indeterminate_with_missing, "empty observed/missing"),
    )
    for payload, match in cases:
        with pytest.raises(ProposalError, match=match):
            _parse_hybrid_observation(
                proposal,
                payload,
                FakeReceipt(thread_id="observer"),  # type: ignore[arg-type]
            )

    valid = _parse_hybrid_observation(
        proposal,
        absent_observation_payload(),
        FakeReceipt(thread_id="observer"),  # type: ignore[arg-type]
    )
    assert valid.evidence.disposition is Disposition.CERTIFIED_ABSENT
    certificate = json.loads(valid.evidence.certificate)
    assert certificate["certificate_semantics"] == (
        "archived_model_nonmatch_for_frozen_operational_claim"
    )
    assert certificate["reason"] == (
        "the visible boundary has the other fully visible corner pattern"
    )
    assert certificate["missing_cue_reasons"] == [
        {
            "cue_id": "inward_v_notch",
            "finding": "the fully visible boundary turns outward at every corner",
        }
    ]
    assert certificate["visibility_certificate"] == (
        "the entire polygon boundary is visible"
    )

    without_summary = absent_observation_payload()
    without_summary["reason"] = None
    optional = _parse_hybrid_observation(
        proposal,
        without_summary,
        FakeReceipt(thread_id="observer-without-summary"),  # type: ignore[arg-type]
    )
    assert json.loads(optional.evidence.certificate)["reason"] is None

    changed_summary = absent_observation_payload()
    changed_summary["reason"] = "a different overall nonmatch summary"
    changed = _parse_hybrid_observation(
        proposal,
        changed_summary,
        FakeReceipt(thread_id="observer-changed-summary"),  # type: ignore[arg-type]
    )
    assert changed.evidence.certificate != valid.evidence.certificate


def test_wrong_epistemic_marker_and_old_observation_keys_fail_closed() -> None:
    proposal = parse_rule_proposal(
        hybrid_payload(), receipt=FakeReceipt(), observable_catalog={}  # type: ignore[arg-type]
    )
    wrong = present_observation_payload()
    wrong["epistemic_status"] = "pixel_truth"
    with pytest.raises(ProposalError, match="epistemic_status"):
        _parse_hybrid_observation(
            proposal, wrong, FakeReceipt(thread_id="observer")  # type: ignore[arg-type]
        )

    old = {
        "disposition": "present",
        "observed_cues": ["one closed polygon", "one inward notch"],
        "missing_required_cues": [],
        "certificate": None,
        "reason": None,
        "error_type": None,
    }
    with pytest.raises(ProposalError, match="fields differ from schema"):
        _parse_hybrid_observation(
            proposal, old, FakeReceipt(thread_id="observer")  # type: ignore[arg-type]
        )


def test_runtime_archives_semantically_invalid_observation_as_error() -> None:
    proposal = parse_rule_proposal(
        hybrid_payload(), receipt=FakeReceipt(), observable_catalog={}  # type: ignore[arg-type]
    )
    malformed = present_observation_payload()
    malformed["observed_cue_ids"] = ["closed_polygon"]
    receipt = FakeReceipt(thread_id="malformed-observer")
    observation = parse_hybrid_observation_or_error(
        proposal, malformed, receipt  # type: ignore[arg-type]
    )
    assert observation.payload == malformed
    assert observation.receipt is receipt
    assert observation.evidence.disposition is Disposition.ERROR
    assert observation.evidence.error_type == "ProposalError"
    assert "present requires" in (observation.evidence.reason or "")


def test_observer_prompt_declares_empirical_status_and_exact_cue_protocol() -> None:
    proposal = parse_rule_proposal(
        hybrid_payload(), receipt=FakeReceipt(), observable_catalog={}  # type: ignore[arg-type]
    )
    prompt = hybrid_observer_prompt(proposal)
    assert HYBRID_EPISTEMIC_STATUS in prompt
    assert "not pixel\ntruth" in prompt
    assert '"cue_id": "closed_polygon"' in prompt
    assert "Copy cue IDs exactly" in prompt
    assert "overall summary of why the image is a nonmatch" in prompt
    assert "never substitutes for cue-keyed findings" in prompt


def test_proposer_prompt_requires_conjunctive_near_miss_coverage() -> None:
    prompt = proposer_prompt({})
    assert "every positive panel must visibly satisfy every" in prompt
    assert "every negative panel must visibly fail at least one" in prompt
    assert "different near-miss subgroups" in prompt
    assert "collapsing them into one vague word" in prompt


def test_episode_adapter_keeps_raw_proposal_and_two_observer_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    positive: list[Path] = []
    negative: list[Path] = []
    for side, collection in (("p", positive), ("n", negative)):
        for index in range(6):
            path = tmp_path / f"{side}-{index}.png"
            write_png(path, index, rgb=True)
            collection.append(path)
    proposer_calls = 0
    observer_calls = 0
    snapshot_calls = 0
    received_snapshots: list[CloudPolicyCacheSnapshot] = []

    def freeze_policy_once() -> CloudPolicyCacheSnapshot:
        nonlocal snapshot_calls
        snapshot_calls += 1
        return CloudPolicyCacheSnapshot(None)

    monkeypatch.setattr(
        "bongard.proposer.snapshot_cloud_policy_cache", freeze_policy_once
    )

    def proposer_transport(prompt, paths, schema, **kwargs):
        nonlocal proposer_calls
        del prompt, paths, schema
        proposer_calls += 1
        received_snapshots.append(kwargs.pop("cloud_policy_cache_snapshot"))
        del kwargs
        return SimpleNamespace(payload=hybrid_payload(), receipt=FakeReceipt())

    def observer_transport(prompt, paths, names, schema, **kwargs):
        nonlocal observer_calls
        del prompt, paths, names, schema
        observer_calls += 1
        received_snapshots.append(kwargs.pop("cloud_policy_cache_snapshot"))
        del kwargs
        return SimpleNamespace(
            payload={
                "epistemic_status": HYBRID_EPISTEMIC_STATUS,
                "disposition": "present",
                "observed_cue_ids": ["closed_polygon", "inward_v_notch"],
                "missing_cue_ids": [],
                "missing_cue_reasons": [],
                "visibility_certificate": None,
                "reason": None,
                "error_type": None,
            },
            receipt=FakeReceipt(thread_id=f"observer-{observer_calls}"),
        )

    session = HeadlessCodexEpisode(
        proposer_transport=proposer_transport,
        observer_transport=observer_transport,
    )
    proposed = session.propose(
        SupportInput(tuple(positive), tuple(negative))  # type: ignore[arg-type]
    )
    query_path = tmp_path / "query.png"
    write_png(query_path, 9, rgb=True)
    for query_id in ("query-0", "query-1"):
        evidence = session.observe(
            SimpleNamespace(
                query_id=query_id,
                panel_path=query_path,
                freeze=SimpleNamespace(
                    proposer_digest=proposed.proposer_digest,
                    formula=proposed.formula,
                ),
                registry=proposed.registry,
            )
        )
        assert evidence[()].disposition is Disposition.PRESENT
    assert proposer_calls == 1
    assert observer_calls == 2
    assert snapshot_calls == 1
    assert len(received_snapshots) == 3
    assert all(item is received_snapshots[0] for item in received_snapshots)
    artifacts = session.artifact_data()
    assert artifacts["schema"].endswith(".v4")
    assert artifacts["rejected_proposal_attempt"] is None
    assert artifacts["proposal"]["hybrid_claim"]["phrase"] == "notched trapezoid-like form"
    assert sorted(artifacts["observations"]) == ["query-0", "query-1"]
