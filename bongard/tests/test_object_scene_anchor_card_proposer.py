from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Mapping, Any

from PIL import Image, ImageDraw
import pytest

from bongard.canonical import canonical_digest
from bongard.object_scene_anchor_card_proposer import (
    ObjectSceneAnchorCardProposerArtifact,
    ObjectSceneAnchorCardProposerError,
    ObjectSceneAnchorCardProposerInput,
    ObjectSceneAnchorCardProposerPanelInput,
    freeze_object_scene_anchor_card_proposer_input,
    object_scene_anchor_card_proposer_output_schema,
    object_scene_anchor_card_proposer_prompt,
    propose_object_scene_anchor_cards,
    verify_object_scene_anchor_card_proposer_artifact,
)
from bongard.object_scene_anchor_catalog import extract_object_scene_anchor_catalog
from bongard.object_scene_anchor_panel_manifest import (
    build_object_scene_anchor_panel_decision_manifest,
)
from bongard.object_scene_anchor_support_sheet import (
    build_object_scene_anchor_support_sheet,
)
from bongard.object_scene_visual_frontend import extract_object_scene_proposal_inventory
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
LAUNCHER = "c" * 64
MODEL_CATALOG, ATTESTATION = canonical_no_tools_runtime(LAUNCHER)


def _panel_png(index: int) -> bytes:
    image = Image.new("RGB", (104, 72), "white")
    draw = ImageDraw.Draw(image)
    dx = index % 4
    dy = index // 4
    draw.line(
        (8 + dx, 34 + dy, 21 + dx, 8 + dy, 35 + dx, 34 + dy, 8 + dx, 34 + dy),
        fill="black",
        width=3,
    )
    draw.line(
        (61 - dx, 48 - dy, 74 - dx, 20 - dy, 91 - dx, 48 - dy, 61 - dx, 48 - dy),
        fill="black",
        width=3,
    )
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


@lru_cache(maxsize=1)
def _inputs():
    rows = []
    for index in range(12):
        panel = _panel_png(index)
        inventory = extract_object_scene_proposal_inventory(panel)
        catalog = extract_object_scene_anchor_catalog(panel, inventory)
        manifest = build_object_scene_anchor_panel_decision_manifest(
            catalog, panel, inventory
        )
        sheet, sheet_png = build_object_scene_anchor_support_sheet(
            panel, inventory, catalog, manifest
        )
        rows.append(ObjectSceneAnchorCardProposerPanelInput(sheet, sheet_png, manifest))
    return tuple(rows[:6]), tuple(rows[6:])


def _raw_payload() -> dict[str, object]:
    def card(phrase: str, statement: str, aliases):
        return {
            "phrase": phrase,
            "anchor_kind": "entity",
            "frame_lower": 0,
            "frame_upper": 0,
            "required_witnesses": [
                {"kind": "shape_appearance", "statement": statement}
            ],
            "accepted_variants": [],
            "near_miss_boundaries": [],
            "positive_support_citations": [
                {
                    "panel_alias": alias,
                    "object_id": "object_0000",
                    "anchor_id": "entity",
                }
                for alias in aliases
            ],
        }

    return {
        "side0_positive": [
            card(
                "angular peaked form",
                "the bound form has one angular peaked contour",
                tuple(f"panel_{index:03d}" for index in range(6)),
            ),
            card(
                "pointed upper contour",
                "the bound form carries a sharply pointed upper contour",
                tuple(f"panel_{index:03d}" for index in range(6)),
            ),
            card(
                "wide angular base",
                "the bound form extends along one wide angular base",
                tuple(f"panel_{index:03d}" for index in range(6)),
            ),
            card(
                "paired angular lobes",
                "the bound form contains paired angular outer lobes",
                tuple(f"panel_{index:03d}" for index in range(6)),
            ),
        ],
        "side1_positive": [
            card(
                "compact enclosed form",
                "the bound form carries a compact enclosed outline",
                tuple(f"panel_{index:03d}" for index in range(6, 12)),
            ),
            card(
                "rounded upper contour",
                "the bound form has one smoothly rounded upper contour",
                tuple(f"panel_{index:03d}" for index in range(6, 12)),
            ),
            card(
                "narrow tapered base",
                "the bound form extends into one narrow tapered base",
                tuple(f"panel_{index:03d}" for index in range(6, 12)),
            ),
            card(
                "paired curved lobes",
                "the bound form contains paired curved outer lobes",
                tuple(f"panel_{index:03d}" for index in range(6, 12)),
            ),
        ],
    }


class _Transport:
    def __init__(self, payload=None, failure: Exception | None = None):
        self.payload = _raw_payload() if payload is None else payload
        self.failure = failure
        self.calls = 0
        self.paths: tuple[str, ...] = ()

    def __call__(self, prompt, paths, schema, **kwargs):
        self.calls += 1
        self.paths = tuple(paths)
        if self.failure is not None:
            raise self.failure
        assert tuple(Path(path).name for path in paths) == tuple(
            [f"pos_{index}.png" for index in range(6)]
            + [f"neg_{index}.png" for index in range(6)]
        )
        receipt = canonical_codex_receipt(
            prompt,
            paths,
            schema,
            self.payload,
            launcher_digest=LAUNCHER,
            reasoning_effort=EFFORT,
            model=MODEL,
            command_fixture="anchor card proposer",
        )
        return CodexStructuredResult(dict(self.payload), receipt)


def _call(transport):
    side0, side1 = _inputs()
    frozen = freeze_object_scene_anchor_card_proposer_input(side0, side1)
    artifact = propose_object_scene_anchor_cards(
        side0,
        side1,
        proposer_input=frozen,
        expected_input_digest=frozen.input_digest,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER,
        cloud_policy_cache_snapshot=CloudPolicyCacheSnapshot(None),
        model_catalog_snapshot=MODEL_CATALOG,
        no_tools_attestation=ATTESTATION,
        transport=transport,
    )
    return artifact, frozen


def test_one_call_builds_committed_proposal_and_cold_replays() -> None:
    transport = _Transport()
    artifact, frozen = _call(transport)
    side0, side1 = _inputs()

    assert transport.calls == artifact.physical_call_count == 1
    assert artifact.status == "success"
    assert artifact.proposal is not None
    assert len(artifact.proposal.side0_positive) == 4
    assert len(artifact.proposal.side1_positive) == 4
    assert artifact.proposal.dropped_cards == ()
    assert ObjectSceneAnchorCardProposerArtifact.from_data(artifact.to_data()) == artifact
    assert verify_object_scene_anchor_card_proposer_artifact(
        artifact,
        side0,
        side1,
        expected_artifact_digest=artifact.artifact_digest,
        expected_input_digest=frozen.input_digest,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER,
        cloud_policy_cache_snapshot=CloudPolicyCacheSnapshot(None),
        model_catalog_snapshot=MODEL_CATALOG,
        no_tools_attestation=ATTESTATION,
    ) == artifact
    assert transport.calls == 1


def test_prompt_has_full_layout_and_every_object_anchor_legend() -> None:
    side0, side1 = _inputs()
    frozen = freeze_object_scene_anchor_card_proposer_input(side0, side1)
    prompt = object_scene_anchor_card_proposer_prompt(frozen)

    for panel in frozen.panels:
        assert panel.panel_alias in prompt
        assert panel.standard_filename in prompt
        assert f"objects={panel.sheet.proposal_count}" in prompt
        for obj in panel.sheet.objects:
            assert obj.object_id in prompt
            assert f"{obj.crop_width_pixels},{obj.crop_height_pixels}" in prompt
            for slot in obj.atlas_slots:
                assert f"anchor_id={slot.anchor_id}" in prompt
                assert f"atlas_tile={slot.slot_id}" in prompt
                assert f"sheet_rect=({slot.sheet_x_pixels},{slot.sheet_y_pixels}" in prompt
    assert "query" not in prompt.casefold()
    assert "lean" not in prompt.casefold()
    assert "transport filenames only" in prompt
    assert "they do not express truth or logical polarity" in prompt
    assert "side1_positive must state an affirmative property visibly present" in prompt
    assert "opposite orientation as visual contrast" in prompt
    assert "unlikely to be visible on any opposite-orientation sheet" in prompt
    assert "must all hold on one same cited binding" in prompt
    assert "bird-like silhouette or strongly oblique edges" in prompt
    assert "it is never a negation, absence, failure, or complement" in prompt
    assert "Produce exactly four distinct cards per orientation" in prompt
    assert "distinct locally decidable witness bundles" in prompt
    schema = object_scene_anchor_card_proposer_output_schema(frozen)
    assert set(schema["properties"]) == {"side0_positive", "side1_positive"}


def test_redundant_raw_cards_allow_committed_builder_drops() -> None:
    payload = _raw_payload()
    for card in payload["side0_positive"][:3]:  # type: ignore[index]
        card["required_witnesses"][0]["statement"] = (  # type: ignore[index]
            "the bound form is not locally decidable"
        )
    transport = _Transport(payload)
    artifact, _ = _call(transport)

    assert artifact.status == "success"
    assert artifact.proposal is not None
    assert len(artifact.proposal.side0_positive) == 1
    assert len(artifact.proposal.side1_positive) == 4
    assert len(artifact.proposal.dropped_cards) == 3
    assert {item.reason_code for item in artifact.proposal.dropped_cards} == {
        "witness_policy"
    }


def test_raw_card_buckets_are_exactly_four_and_need_one_usable_card() -> None:
    underfilled = _raw_payload()
    underfilled["side0_positive"].pop()  # type: ignore[union-attr]
    underfilled_artifact, _ = _call(_Transport(underfilled))
    assert underfilled_artifact.status == "parser_error"
    assert underfilled_artifact.proposal is None

    unusable = _raw_payload()
    for card in unusable["side0_positive"]:  # type: ignore[union-attr]
        card["required_witnesses"][0]["statement"] = (  # type: ignore[index]
            "the bound form is not locally decidable"
        )
    unusable_artifact, _ = _call(_Transport(unusable))
    assert unusable_artifact.status == "parser_error"
    assert unusable_artifact.proposal is None


def test_parser_and_transport_failures_are_typed_and_select_no_proposal() -> None:
    malformed = _raw_payload()
    malformed["side0_positive"][0]["positive_support_citations"].pop()  # type: ignore[index]
    parser_transport = _Transport(malformed)
    parser, _ = _call(parser_transport)
    assert parser_transport.calls == 1
    assert parser.status == "parser_error"
    assert parser.receipt is not None and parser.model_payload is not None
    assert parser.proposal is None
    assert ObjectSceneAnchorCardProposerArtifact.from_data(parser.to_data()) == parser

    failed_transport = _Transport(failure=RuntimeError("offline"))
    failed, _ = _call(failed_transport)
    assert failed_transport.calls == 1
    assert failed.status == "transport_error"
    assert failed.receipt is failed.model_payload is failed.proposal is None
    assert ObjectSceneAnchorCardProposerArtifact.from_data(failed.to_data()) == failed


def test_input_and_artifact_tamper_or_reordering_fail_closed() -> None:
    artifact, frozen = _call(_Transport())
    assert ObjectSceneAnchorCardProposerInput.from_data(frozen.to_data()) == frozen

    reordered = deepcopy(frozen.to_data())
    reordered["panels"].reverse()
    reordered["input_digest"] = canonical_digest(
        {key: value for key, value in reordered.items() if key != "input_digest"}
    )
    with pytest.raises(ObjectSceneAnchorCardProposerError, match="canonical support"):
        ObjectSceneAnchorCardProposerInput.from_data(reordered)

    tampered = deepcopy(artifact.to_data())
    tampered["status"] = "parser_error"
    tampered["failure_code"] = "payload_rejected"
    tampered["failure_type"] = "ForgedError"
    tampered["proposal"] = None
    tampered["artifact_digest"] = canonical_digest(
        {key: value for key, value in tampered.items() if key != "artifact_digest"}
    )
    with pytest.raises(
        ObjectSceneAnchorCardProposerError,
        match="parser-error payload now succeeds",
    ):
        ObjectSceneAnchorCardProposerArtifact.from_data(tampered)
