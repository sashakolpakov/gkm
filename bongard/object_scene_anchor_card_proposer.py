"""One-call support-sheet proposer for exact affirmative anchor cards."""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Mapping, Sequence

import bongard.transport as _transport_runtime
from bongard.canonical import canonical_digest, canonical_json
from bongard.codex_no_tools_preflight import (
    CodexNoToolsAttestation,
    validate_codex_no_tools_attestation,
)
from bongard.object_scene_anchor_bindings import ObjectSceneAnchorBindingSpec
from bongard.object_scene_anchor_cards import (
    OBJECT_SCENE_ANCHOR_CARD_WITNESS_KINDS,
    OBJECT_SCENE_ANCHOR_MAX_CARDS_PER_ORIENTATION,
    OBJECT_SCENE_ANCHOR_MAX_NEAR_MISS_BOUNDARIES,
    OBJECT_SCENE_ANCHOR_MAX_VARIANTS,
    OBJECT_SCENE_ANCHOR_MAX_WITNESSES_PER_CARD,
    ObjectSceneAnchorCardProposal,
    build_object_scene_anchor_card_proposal,
)
from bongard.object_scene_anchor_panel_manifest import (
    ObjectSceneAnchorPanelDecisionManifest,
)
from bongard.object_scene_anchor_support_sheet import (
    ObjectSceneAnchorSupportSheet,
    object_scene_anchor_support_sheet_renderer_digest,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.transport import (
    CODEX_ISOLATION_POLICY,
    CODEX_RECEIPT_SCHEMA,
    REASONING_EFFORTS,
    STRUCTURED_INPUT_DIGEST_SCHEMA,
    CloudPolicyCacheSnapshot,
    CodexModelCatalogSnapshot,
    CodexProposerFailure,
    CodexReceipt,
    CodexStructuredResult,
    ordered_panel_view_digest,
    run_codex_structured,
    semantic_panel_set_digest,
    validate_codex_receipt,
    validate_codex_strict_output_schema,
)


OBJECT_SCENE_ANCHOR_CARD_PROPOSER_PROTOCOL_ID = (
    "bongard.object-scene-anchor-card-proposer/support-sheet-one-call-v1"
)
OBJECT_SCENE_ANCHOR_CARD_PROPOSER_PANEL_SCHEMA = (
    "gkm.object-scene-anchor-card-proposer-panel.v1"
)
OBJECT_SCENE_ANCHOR_CARD_PROPOSER_INPUT_SCHEMA = (
    "gkm.object-scene-anchor-card-proposer-input.v1"
)
OBJECT_SCENE_ANCHOR_CARD_PROPOSER_ARTIFACT_SCHEMA = (
    "gkm.object-scene-anchor-card-proposer-artifact.v1"
)
OBJECT_SCENE_ANCHOR_CARD_PROPOSER_PROTOCOL_SCHEMA = (
    "gkm.object-scene-anchor-card-proposer-protocol.v1"
)

SUPPORT_PANELS_PER_SIDE = 6
SUPPORT_PANEL_COUNT = 12
MAX_PROMPT_UTF8_BYTES = 96_000

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_PANEL_ALIAS = re.compile(r"panel_[0-9]{3}\Z")
_OBJECT_ID = re.compile(r"object_[0-9]{4}\Z")
_ANCHOR_ID = re.compile(r"(?:entity|part-[0-9]{8}|compact-[0-9]{8}|frame-[0-9]{8})\Z")


class ObjectSceneAnchorCardProposerError(RuntimeError):
    """A proposer input, payload, receipt, outcome, or replay is invalid."""


StructuredTransport = Callable[..., CodexStructuredResult]


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "support_sheet_count": SUPPORT_PANEL_COUNT,
        "query_material_admitted": False,
        "single_no_tools_call": True,
        "raw_bindings_resolved_by_python": True,
        "final_proposal_built_by_committed_builder": True,
    }


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != expected
    ):
        raise ObjectSceneAnchorCardProposerError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ObjectSceneAnchorCardProposerError(f"{label} must be a lowercase SHA-256")
    return value


def _address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise ObjectSceneAnchorCardProposerError(f"{label} must be a sha256: address")
    return value


def _canonical_payload(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ObjectSceneAnchorCardProposerError("proposer payload must be a JSON object")
    try:
        decoded = json.loads(canonical_json(dict(value)).decode("utf-8"))
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectSceneAnchorCardProposerError(
            "proposer payload is not finite canonical JSON"
        ) from exc
    if not isinstance(decoded, dict):
        raise ObjectSceneAnchorCardProposerError("proposer payload must be an object")
    return decoded


def _exception_type(exc: BaseException) -> str:
    name = type(exc).__name__
    return name if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", name) else "UnclassifiedError"


def object_scene_anchor_card_proposer_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def object_scene_anchor_card_proposer_transport_source_digest() -> str:
    source = getattr(_transport_runtime, "__file__", None)
    if not isinstance(source, str) or not source:
        raise ObjectSceneAnchorCardProposerError("structured transport source is unavailable")
    return hashlib.sha256(Path(source).read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorCardProposerPanelInput:
    sheet: ObjectSceneAnchorSupportSheet
    sheet_png_bytes: bytes
    panel_manifest: ObjectSceneAnchorPanelDecisionManifest

    def __post_init__(self) -> None:
        if type(self.sheet) is not ObjectSceneAnchorSupportSheet:
            raise TypeError("sheet must be exact ObjectSceneAnchorSupportSheet")
        if type(self.panel_manifest) is not ObjectSceneAnchorPanelDecisionManifest:
            raise TypeError("panel_manifest must be exact decision manifest")
        sheet = ObjectSceneAnchorSupportSheet.from_data(self.sheet.to_data())
        manifest = ObjectSceneAnchorPanelDecisionManifest.from_data(
            self.panel_manifest.to_data()
        )
        if not isinstance(self.sheet_png_bytes, bytes):
            raise TypeError("sheet PNG must be exact bytes")
        if (
            sheet.panel_manifest_digest != manifest.manifest_digest
            or sheet.panel_digest != manifest.panel_digest
            or sheet.inventory_digest != manifest.inventory_digest
            or sheet.proposal_count != manifest.proposal_count
            or sheet.object_ids != manifest.object_ids
            or sheet.sheet_png_byte_count != len(self.sheet_png_bytes)
            or sheet.sheet_png_digest != hashlib.sha256(self.sheet_png_bytes).hexdigest()
        ):
            raise ObjectSceneAnchorCardProposerError(
                "support sheet, PNG, and panel manifest differ"
            )


def _panel_content(value: "ObjectSceneAnchorCardProposerPanel") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_CARD_PROPOSER_PANEL_SCHEMA,
        "panel_alias": value.panel_alias,
        "orientation": value.orientation,
        "side_index": value.side_index,
        "standard_filename": value.standard_filename,
        "sheet": value.sheet.to_data(),
        "sheet_artifact_digest": value.sheet_artifact_digest,
        "sheet_png_byte_count": value.sheet_png_byte_count,
        "sheet_png_digest": value.sheet_png_digest,
        "panel_manifest": value.panel_manifest.to_data(),
        "panel_manifest_digest": value.panel_manifest_digest,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorCardProposerPanel:
    panel_alias: str
    orientation: str
    side_index: int
    standard_filename: str
    sheet: ObjectSceneAnchorSupportSheet
    sheet_artifact_digest: str
    sheet_png_byte_count: int
    sheet_png_digest: str
    panel_manifest: ObjectSceneAnchorPanelDecisionManifest
    panel_manifest_digest: str
    panel_digest: str

    def __post_init__(self) -> None:
        if _PANEL_ALIAS.fullmatch(self.panel_alias) is None:
            raise ObjectSceneAnchorCardProposerError("proposer panel alias differs")
        global_index = int(self.panel_alias[-3:])
        expected_orientation = "side0_positive" if global_index < 6 else "side1_positive"
        expected_side_index = global_index if global_index < 6 else global_index - 6
        expected_filename = f"{'pos' if global_index < 6 else 'neg'}_{expected_side_index}.png"
        if (
            self.orientation != expected_orientation
            or self.side_index != expected_side_index
            or self.standard_filename != expected_filename
            or type(self.sheet) is not ObjectSceneAnchorSupportSheet
            or type(self.panel_manifest) is not ObjectSceneAnchorPanelDecisionManifest
            or type(self.sheet_png_byte_count) is not int
            or self.sheet_png_byte_count < 1
        ):
            raise ObjectSceneAnchorCardProposerError("proposer panel position differs")
        for label, item in (
            ("sheet artifact digest", self.sheet_artifact_digest),
            ("sheet PNG digest", self.sheet_png_digest),
            ("panel manifest digest", self.panel_manifest_digest),
            ("proposer panel digest", self.panel_digest),
        ):
            _digest(item, label)
        if (
            self.sheet.artifact_digest != self.sheet_artifact_digest
            or self.sheet.sheet_png_byte_count != self.sheet_png_byte_count
            or self.sheet.sheet_png_digest != self.sheet_png_digest
            or self.panel_manifest.manifest_digest != self.panel_manifest_digest
            or self.sheet.panel_manifest_digest != self.panel_manifest_digest
            or self.panel_digest != canonical_digest(_panel_content(self))
        ):
            raise ObjectSceneAnchorCardProposerError("proposer panel binding differs")

    @classmethod
    def create(
        cls, index: int, source: ObjectSceneAnchorCardProposerPanelInput
    ) -> "ObjectSceneAnchorCardProposerPanel":
        values = {
            "panel_alias": f"panel_{index:03d}",
            "orientation": "side0_positive" if index < 6 else "side1_positive",
            "side_index": index if index < 6 else index - 6,
            "standard_filename": f"{'pos' if index < 6 else 'neg'}_{index if index < 6 else index - 6}.png",
            "sheet": source.sheet,
            "sheet_artifact_digest": source.sheet.artifact_digest,
            "sheet_png_byte_count": len(source.sheet_png_bytes),
            "sheet_png_digest": hashlib.sha256(source.sheet_png_bytes).hexdigest(),
            "panel_manifest": source.panel_manifest,
            "panel_manifest_digest": source.panel_manifest.manifest_digest,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, panel_digest=canonical_digest(_panel_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_panel_content(self), "panel_digest": self.panel_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorCardProposerPanel":
        raw = _fields(
            value,
            {
                "schema", "panel_alias", "orientation", "side_index", "standard_filename",
                "sheet", "sheet_artifact_digest", "sheet_png_byte_count", "sheet_png_digest",
                "panel_manifest", "panel_manifest_digest", *_authority_data(), "panel_digest",
            },
            "proposer panel",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_CARD_PROPOSER_PANEL_SCHEMA
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["sheet"], Mapping)
            or not isinstance(raw["panel_manifest"], Mapping)
        ):
            raise ObjectSceneAnchorCardProposerError("proposer panel policy differs")
        result = cls(
            raw["panel_alias"], raw["orientation"], raw["side_index"], raw["standard_filename"],
            ObjectSceneAnchorSupportSheet.from_data(raw["sheet"]), raw["sheet_artifact_digest"],
            raw["sheet_png_byte_count"], raw["sheet_png_digest"],
            ObjectSceneAnchorPanelDecisionManifest.from_data(raw["panel_manifest"]),
            raw["panel_manifest_digest"], raw["panel_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorCardProposerError("proposer panel is not canonical")
        return result


def _input_content(value: "ObjectSceneAnchorCardProposerInput") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_CARD_PROPOSER_INPUT_SCHEMA,
        "panels": [item.to_data() for item in value.panels],
        "panel_aliases": list(value.panel_aliases),
        "standard_filenames": list(value.standard_filenames),
        "ordered_panel_view_digest": value.ordered_panel_view_digest,
        "semantic_panel_set_digest": value.semantic_panel_set_digest,
        "support_sheet_renderer_digest": value.support_sheet_renderer_digest,
        "side_partition": "panel_000..005-side0;panel_006..011-side1",
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorCardProposerInput:
    panels: tuple[ObjectSceneAnchorCardProposerPanel, ...]
    panel_aliases: tuple[str, ...]
    standard_filenames: tuple[str, ...]
    ordered_panel_view_digest: str
    semantic_panel_set_digest: str
    support_sheet_renderer_digest: str
    input_digest: str

    def __post_init__(self) -> None:
        expected_aliases = tuple(f"panel_{index:03d}" for index in range(12))
        expected_names = tuple(
            [f"pos_{index}.png" for index in range(6)]
            + [f"neg_{index}.png" for index in range(6)]
        )
        if (
            type(self.panels) is not tuple
            or len(self.panels) != SUPPORT_PANEL_COUNT
            or any(type(item) is not ObjectSceneAnchorCardProposerPanel for item in self.panels)
            or self.panel_aliases != expected_aliases
            or tuple(item.panel_alias for item in self.panels) != expected_aliases
            or self.standard_filenames != expected_names
            or tuple(item.standard_filename for item in self.panels) != expected_names
            or len({item.sheet_artifact_digest for item in self.panels}) != SUPPORT_PANEL_COUNT
            or len({item.panel_manifest.panel_digest for item in self.panels}) != SUPPORT_PANEL_COUNT
        ):
            raise ObjectSceneAnchorCardProposerError(
                "proposer input must contain twelve distinct canonical support panels"
            )
        if self.support_sheet_renderer_digest != object_scene_anchor_support_sheet_renderer_digest():
            raise ObjectSceneAnchorCardProposerError("support-sheet renderer binding differs")
        _digest(self.ordered_panel_view_digest, "ordered panel view digest")
        _address(self.semantic_panel_set_digest, "semantic panel set digest")
        _digest(self.input_digest, "proposer input digest")
        identities = [
            {
                "name": item.standard_filename,
                "byte_count": item.sheet_png_byte_count,
                "content_digest": item.sheet_png_digest,
            }
            for item in self.panels
        ]
        if (
            self.ordered_panel_view_digest != canonical_digest(identities)
            or self.input_digest != canonical_digest(_input_content(self))
        ):
            raise ObjectSceneAnchorCardProposerError("proposer input digest differs")

    @property
    def side0_panel_manifests(self) -> dict[str, ObjectSceneAnchorPanelDecisionManifest]:
        return {item.panel_alias: item.panel_manifest for item in self.panels[:6]}

    @property
    def side1_panel_manifests(self) -> dict[str, ObjectSceneAnchorPanelDecisionManifest]:
        return {item.panel_alias: item.panel_manifest for item in self.panels[6:]}

    def to_data(self) -> dict[str, object]:
        return {**_input_content(self), "input_digest": self.input_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorCardProposerInput":
        raw = _fields(
            value,
            {
                "schema", "panels", "panel_aliases", "standard_filenames",
                "ordered_panel_view_digest", "semantic_panel_set_digest",
                "support_sheet_renderer_digest", "side_partition",
                *_authority_data(), "input_digest",
            },
            "proposer input",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_CARD_PROPOSER_INPUT_SCHEMA
            or raw["side_partition"] != "panel_000..005-side0;panel_006..011-side1"
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["panels"], list)
            or not isinstance(raw["panel_aliases"], list)
            or not isinstance(raw["standard_filenames"], list)
        ):
            raise ObjectSceneAnchorCardProposerError("proposer input policy differs")
        result = cls(
            tuple(ObjectSceneAnchorCardProposerPanel.from_data(item) for item in raw["panels"]),
            tuple(raw["panel_aliases"]), tuple(raw["standard_filenames"]),
            raw["ordered_panel_view_digest"], raw["semantic_panel_set_digest"],
            raw["support_sheet_renderer_digest"], raw["input_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorCardProposerError("proposer input is not canonical")
        return result


def _source_rows(
    side0: Sequence[ObjectSceneAnchorCardProposerPanelInput],
    side1: Sequence[ObjectSceneAnchorCardProposerPanelInput],
) -> tuple[ObjectSceneAnchorCardProposerPanelInput, ...]:
    if any(isinstance(value, (str, bytes)) or not isinstance(value, Sequence) for value in (side0, side1)):
        raise TypeError("proposer sides must be finite sequences")
    rows = tuple(side0) + tuple(side1)
    if len(side0) != 6 or len(side1) != 6 or any(
        type(item) is not ObjectSceneAnchorCardProposerPanelInput for item in rows
    ):
        raise ObjectSceneAnchorCardProposerError("proposer requires exactly six sheets per side")
    return rows


def _write_standard_view(
    directory: Path,
    rows: Sequence[ObjectSceneAnchorCardProposerPanelInput],
) -> tuple[str, ...]:
    names = [f"pos_{index}.png" for index in range(6)] + [f"neg_{index}.png" for index in range(6)]
    paths: list[str] = []
    for name, row in zip(names, rows, strict=True):
        target = directory / name
        target.write_bytes(row.sheet_png_bytes)
        paths.append(str(target.resolve()))
    return tuple(paths)


def freeze_object_scene_anchor_card_proposer_input(
    side0: Sequence[ObjectSceneAnchorCardProposerPanelInput],
    side1: Sequence[ObjectSceneAnchorCardProposerPanelInput],
) -> ObjectSceneAnchorCardProposerInput:
    rows = _source_rows(side0, side1)
    panels = tuple(
        ObjectSceneAnchorCardProposerPanel.create(index, item)
        for index, item in enumerate(rows)
    )
    with tempfile.TemporaryDirectory(prefix="bongard-anchor-card-input-") as raw:
        paths = _write_standard_view(Path(raw), rows)
        view_digest = ordered_panel_view_digest(paths)
        set_digest = semantic_panel_set_digest(paths)
    values = {
        "panels": panels,
        "panel_aliases": tuple(item.panel_alias for item in panels),
        "standard_filenames": tuple(item.standard_filename for item in panels),
        "ordered_panel_view_digest": view_digest,
        "semantic_panel_set_digest": set_digest,
        "support_sheet_renderer_digest": object_scene_anchor_support_sheet_renderer_digest(),
    }
    provisional = object.__new__(ObjectSceneAnchorCardProposerInput)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorCardProposerInput(
        **values, input_digest=canonical_digest(_input_content(provisional))
    )


def object_scene_anchor_card_proposer_prompt(
    proposer_input: ObjectSceneAnchorCardProposerInput,
) -> str:
    if type(proposer_input) is not ObjectSceneAnchorCardProposerInput:
        raise TypeError("proposer_input must be exact proposer input")
    frozen = ObjectSceneAnchorCardProposerInput.from_data(proposer_input.to_data())
    legends: list[str] = []
    for panel in frozen.panels:
        sheet = panel.sheet
        legends.append(
            f"{panel.panel_alias}: file={panel.standard_filename}; orientation={panel.orientation}; "
            f"original_panel_rect=({sheet.panel_x_pixels},{sheet.panel_y_pixels},"
            f"{sheet.panel_width_pixels},{sheet.panel_height_pixels}); objects={sheet.proposal_count}"
        )
        for obj in sheet.objects:
            legends.append(
                f"  {obj.object_id}: crop_rect=({obj.crop_x_pixels},{obj.crop_y_pixels},"
                f"{obj.crop_width_pixels},{obj.crop_height_pixels}); atlas_rect=("
                f"{obj.atlas_x_pixels},{obj.atlas_y_pixels},{obj.atlas_width_pixels},"
                f"{obj.atlas_height_pixels})"
            )
            for slot in obj.atlas_slots:
                legends.append(
                    f"    {slot.binding_alias}: anchor_kind={slot.anchor_kind}; "
                    f"anchor_id={slot.anchor_id}; atlas_tile={slot.slot_id}; "
                    f"atlas_row={slot.atlas_row_index}; atlas_column={slot.atlas_column_index}; "
                    f"sheet_rect=({slot.sheet_x_pixels},{slot.sheet_y_pixels},"
                    f"{slot.width_pixels},{slot.height_pixels})"
                )
    prompt = (
        "Propose affirmative anchor-local visual concept cards in both explicit "
        "orientations. Each grayscale support sheet places the complete original "
        "panel first, then one row per object; each row places the full-style "
        "object crop before its exhaustive anchor atlas. Use the legend below to "
        "cite one exact object and anchor in every support sheet of that card's "
        "orientation. The pos_*.png and neg_*.png names are transport filenames "
        "only; they do not express truth or logical polarity. side0_positive must "
        "state an affirmative property visibly present in every pos_* sheet. "
        "side1_positive must state an affirmative property visibly present in "
        "every neg_* sheet; it is never a negation, absence, failure, or complement "
        "of a side0 property. Use the six sheets in the opposite orientation as "
        "visual contrast when choosing each card: prefer a concrete affirmative "
        "property that is visible on all six sheets of its own orientation and is "
        "unlikely to be visible on any opposite-orientation sheet. Do not spend a "
        "card on a generic property that is obvious in both orientations. Each "
        "required_witness statement becomes a separately registered visual test, "
        "and the registered tests in one eventual conjunction must all hold on one "
        "same cited binding. Make every statement decidable from that local anchor "
        "tile, with no whole-sheet counts or cross-object comparison. Soft but "
        "visually grounded descriptions such as bird-like silhouette or strongly "
        "oblique edges are allowed when the cited tiles support them consistently. "
        "Produce exactly four distinct cards per orientation and exactly six "
        "citations per card in panel-alias order. The four cards are redundant "
        "fallbacks, so give them distinct locally decidable witness bundles. A "
        "card uses one anchor_kind. "
        "For entity or part set frame_lower=frame_upper=0. For frame use an "
        "inclusive interval within 3..8. Write lowercase affirmative atomic prose; "
        "use empty accepted_variants and near_miss_boundaries unless their strict "
        "formats are necessary. Do not reverse an orientation, write code, or "
        "refer to any unseen material.\n\nComplete sheet and anchor legend:\n"
        + "\n".join(legends)
    )
    if len(prompt.encode("utf-8")) > MAX_PROMPT_UTF8_BYTES:
        raise ObjectSceneAnchorCardProposerError("proposer prompt exceeds fixed bound")
    if re.search(r"\b(?:query|lean)\b", prompt, re.IGNORECASE):
        raise ObjectSceneAnchorCardProposerError("proposer prompt crosses sealed boundary")
    return prompt


def object_scene_anchor_card_proposer_output_schema(
    proposer_input: ObjectSceneAnchorCardProposerInput,
) -> dict[str, object]:
    if type(proposer_input) is not ObjectSceneAnchorCardProposerInput:
        raise TypeError("proposer_input must be exact proposer input")
    aliases = list(proposer_input.panel_aliases)
    object_ids = sorted({item for panel in proposer_input.panels for item in panel.sheet.object_ids})
    anchor_ids = sorted(
        {
            slot.anchor_id
            for panel in proposer_input.panels
            for obj in panel.sheet.objects
            for slot in obj.atlas_slots
        }
    )
    witness = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": list(OBJECT_SCENE_ANCHOR_CARD_WITNESS_KINDS)},
            "statement": {"type": "string"},
        },
        "required": ["kind", "statement"],
        "additionalProperties": False,
    }
    citation = {
        "type": "object",
        "properties": {
            "panel_alias": {"type": "string", "enum": aliases},
            "object_id": {"type": "string", "enum": object_ids},
            "anchor_id": {"type": "string", "enum": anchor_ids},
        },
        "required": ["panel_alias", "object_id", "anchor_id"],
        "additionalProperties": False,
    }
    card_properties: dict[str, object] = {
        "phrase": {"type": "string"},
        "anchor_kind": {"type": "string", "enum": ["entity", "part", "frame"]},
        "frame_lower": {"type": "integer", "enum": list(range(9))},
        "frame_upper": {"type": "integer", "enum": list(range(9))},
        "required_witnesses": {"type": "array", "items": witness},
        "accepted_variants": {"type": "array", "items": {"type": "string"}},
        "near_miss_boundaries": {"type": "array", "items": {"type": "string"}},
        "positive_support_citations": {"type": "array", "items": citation},
    }
    card = {
        "type": "object",
        "properties": card_properties,
        "required": list(card_properties),
        "additionalProperties": False,
    }
    schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "side0_positive": {"type": "array", "items": card},
            "side1_positive": {"type": "array", "items": card},
        },
        "required": ["side0_positive", "side1_positive"],
        "additionalProperties": False,
    }
    validate_codex_strict_output_schema(schema)
    return schema


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ObjectSceneAnchorCardProposerError(f"{label} must be a JSON list")
    return value


def _translate_raw_payload(
    payload: Mapping[str, Any], proposer_input: ObjectSceneAnchorCardProposerInput
) -> ObjectSceneAnchorCardProposal:
    raw = _fields(payload, {"side0_positive", "side1_positive"}, "raw card payload")
    translated: dict[str, list[dict[str, object]]] = {}
    card_fields = {
        "phrase", "anchor_kind", "frame_lower", "frame_upper", "required_witnesses",
        "accepted_variants", "near_miss_boundaries", "positive_support_citations",
    }
    for orientation, expected_aliases in (
        ("side0_positive", proposer_input.panel_aliases[:6]),
        ("side1_positive", proposer_input.panel_aliases[6:]),
    ):
        rows = _list(raw[orientation], f"{orientation} cards")
        if len(rows) != OBJECT_SCENE_ANCHOR_MAX_CARDS_PER_ORIENTATION:
            raise ObjectSceneAnchorCardProposerError("raw card bucket capacity differs")
        output: list[dict[str, object]] = []
        for row in rows:
            card = _fields(row, card_fields, "raw card")
            kind, lower, upper = card["anchor_kind"], card["frame_lower"], card["frame_upper"]
            if kind not in ("entity", "part", "frame") or type(lower) is not int or type(upper) is not int:
                raise ObjectSceneAnchorCardProposerError("raw binding fields differ")
            if kind == "entity":
                spec = ObjectSceneAnchorBindingSpec.entity()
            elif kind == "part":
                spec = ObjectSceneAnchorBindingSpec.part()
            elif not 3 <= lower <= upper <= 8:
                raise ObjectSceneAnchorCardProposerError("raw frame interval differs")
            else:
                spec = ObjectSceneAnchorBindingSpec.frame(lower, upper)
            if kind != "frame" and (lower, upper) != (0, 0):
                raise ObjectSceneAnchorCardProposerError("non-frame raw interval must be zero")
            witnesses = _list(card["required_witnesses"], "raw witnesses")
            variants = _list(card["accepted_variants"], "raw variants")
            boundaries = _list(card["near_miss_boundaries"], "raw boundaries")
            citations = _list(card["positive_support_citations"], "raw citations")
            if (
                not 1 <= len(witnesses) <= OBJECT_SCENE_ANCHOR_MAX_WITNESSES_PER_CARD
                or len(variants) > OBJECT_SCENE_ANCHOR_MAX_VARIANTS
                or len(boundaries) > OBJECT_SCENE_ANCHOR_MAX_NEAR_MISS_BOUNDARIES
                or len(citations) != 6
            ):
                raise ObjectSceneAnchorCardProposerError("raw card list bounds differ")
            normalized_witnesses = []
            for item in witnesses:
                witness_raw = _fields(item, {"kind", "statement"}, "raw witness")
                normalized_witnesses.append(dict(witness_raw))
            normalized_citations = []
            for item in citations:
                citation_raw = _fields(
                    item, {"panel_alias", "object_id", "anchor_id"}, "raw citation"
                )
                if (
                    not isinstance(citation_raw["panel_alias"], str)
                    or not isinstance(citation_raw["object_id"], str)
                    or not isinstance(citation_raw["anchor_id"], str)
                    or _OBJECT_ID.fullmatch(citation_raw["object_id"]) is None
                    or _ANCHOR_ID.fullmatch(citation_raw["anchor_id"]) is None
                ):
                    raise ObjectSceneAnchorCardProposerError("raw citation identity differs")
                normalized_citations.append(dict(citation_raw))
            if tuple(item["panel_alias"] for item in normalized_citations) != expected_aliases:
                raise ObjectSceneAnchorCardProposerError(
                    "raw citations must be the exact six orientation panels in order"
                )
            output.append(
                {
                    "phrase": card["phrase"],
                    "binding_spec": spec.to_data(),
                    "required_witnesses": normalized_witnesses,
                    "accepted_variants": variants,
                    "near_miss_boundaries": boundaries,
                    "positive_support_citations": normalized_citations,
                }
            )
        translated[orientation] = output
    proposal = build_object_scene_anchor_card_proposal(
        translated,
        side0_panel_manifests=proposer_input.side0_panel_manifests,
        side1_panel_manifests=proposer_input.side1_panel_manifests,
    )
    return proposal


def object_scene_anchor_card_proposer_protocol_digest() -> str:
    return canonical_digest(
        {
            "schema": OBJECT_SCENE_ANCHOR_CARD_PROPOSER_PROTOCOL_SCHEMA,
            "protocol_id": OBJECT_SCENE_ANCHOR_CARD_PROPOSER_PROTOCOL_ID,
            "source_digest": object_scene_anchor_card_proposer_source_digest(),
            "transport_source_digest": object_scene_anchor_card_proposer_transport_source_digest(),
            "transport_entrypoint": "run_codex_structured",
            "receipt_schema": CODEX_RECEIPT_SCHEMA,
            "input_digest_schema": STRUCTURED_INPUT_DIGEST_SCHEMA,
            "isolation_policy": CODEX_ISOLATION_POLICY,
            "sheet_renderer_digest": object_scene_anchor_support_sheet_renderer_digest(),
            "panel_names": [f"pos_{i}.png" for i in range(6)] + [f"neg_{i}.png" for i in range(6)],
            "raw_binding_translation": "entity-or-part-zero-interval;frame-inclusive-3-through-8",
            **_authority_data(),
        }
    )


def _model_digest(model: str, reasoning_effort: str) -> str:
    if not isinstance(model, str) or _MODEL.fullmatch(model) is None:
        raise ObjectSceneAnchorCardProposerError("proposer model differs")
    if reasoning_effort not in REASONING_EFFORTS:
        raise ObjectSceneAnchorCardProposerError("proposer reasoning effort differs")
    return canonical_digest(
        {
            "schema": "gkm.object-scene-anchor-card-proposer-model.v1",
            "model": model,
            "reasoning_effort": reasoning_effort,
        }
    )


def _runtime_from_pins(
    *, model: str, reasoning_effort: str, launcher: str, policy: str,
    catalog: str, attestation: str,
) -> str:
    _digest(launcher, "launcher digest")
    _digest(catalog, "model catalog digest")
    _digest(attestation, "no-tools digest")
    if policy != "absent":
        _address(policy, "policy-cache binding")
    return canonical_digest(
        {
            "schema": "gkm.object-scene-anchor-card-proposer-runtime.v1",
            "model_digest": _model_digest(model, reasoning_effort),
            "launcher_digest": launcher,
            "policy_cache_binding": policy,
            "model_catalog_digest": catalog,
            "no_tools_attestation_digest": attestation,
            "protocol_digest": object_scene_anchor_card_proposer_protocol_digest(),
            **_authority_data(),
        }
    )


def _runtime(
    *, model: str, reasoning_effort: str, launcher: str,
    policy_snapshot: CloudPolicyCacheSnapshot,
    catalog_snapshot: CodexModelCatalogSnapshot,
    attestation: CodexNoToolsAttestation,
) -> str:
    if not isinstance(policy_snapshot, CloudPolicyCacheSnapshot) or not isinstance(
        catalog_snapshot, CodexModelCatalogSnapshot
    ):
        raise ObjectSceneAnchorCardProposerError("exact proposer runtime snapshots are required")
    try:
        checked = validate_codex_no_tools_attestation(
            attestation,
            expected_launcher_digest=_digest(launcher, "launcher digest"),
            expected_model_catalog_digest=catalog_snapshot.raw_digest,
            expected_cloud_policy_cache_binding=policy_snapshot.binding,
        )
    except (CodexProposerFailure, TypeError, ValueError) as exc:
        raise ObjectSceneAnchorCardProposerError("proposer runtime attestation differs") from exc
    return _runtime_from_pins(
        model=model, reasoning_effort=reasoning_effort, launcher=launcher,
        policy=policy_snapshot.binding, catalog=catalog_snapshot.raw_digest,
        attestation=checked.attestation_digest,
    )


def _validate_receipt(
    receipt: Mapping[str, Any], proposer_input: ObjectSceneAnchorCardProposerInput,
    payload: Mapping[str, Any], prompt: str, schema: Mapping[str, Any],
    *, model: str, reasoning_effort: str, launcher: str, policy: str,
    catalog: str, attestation: str,
) -> None:
    try:
        validate_codex_receipt(receipt)
    except (CodexProposerFailure, TypeError, ValueError) as exc:
        raise ObjectSceneAnchorCardProposerError("proposer receipt is invalid") from exc
    expected = {
        "input_digest_schema": STRUCTURED_INPUT_DIGEST_SCHEMA,
        "prompt_digest": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "task_digest": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "output_schema_digest": canonical_digest(schema),
        "panel_view_digest": proposer_input.ordered_panel_view_digest,
        "panel_set_digest": proposer_input.semantic_panel_set_digest,
        "structured_output_digest": canonical_digest(dict(payload)),
        "requested_model": model,
        "requested_reasoning_effort": reasoning_effort,
        "codex_launcher_digest": launcher,
        "cloud_config_bundle_cache_binding": policy,
        "model_catalog_digest": catalog,
        "tool_surface_attestation_digest": attestation,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ObjectSceneAnchorCardProposerError("proposer receipt bindings differ")


def _artifact_content(value: "ObjectSceneAnchorCardProposerArtifact") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_CARD_PROPOSER_ARTIFACT_SCHEMA,
        "proposer_input": value.proposer_input.to_data(),
        "input_digest": value.input_digest,
        "status": value.status,
        "physical_call_count": value.physical_call_count,
        "model_payload": value.model_payload,
        "receipt": value.receipt,
        "proposal": None if value.proposal is None else value.proposal.to_data(),
        "failure_code": value.failure_code,
        "failure_type": value.failure_type,
        "protocol_digest": value.protocol_digest,
        "source_digest": value.source_digest,
        "transport_source_digest": value.transport_source_digest,
        "model": value.model,
        "reasoning_effort": value.reasoning_effort,
        "model_digest": value.model_digest,
        "launcher_digest": value.launcher_digest,
        "policy_cache_binding": value.policy_cache_binding,
        "model_catalog_digest": value.model_catalog_digest,
        "no_tools_attestation_digest": value.no_tools_attestation_digest,
        "runtime_digest": value.runtime_digest,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorCardProposerArtifact:
    proposer_input: ObjectSceneAnchorCardProposerInput
    input_digest: str
    status: str
    physical_call_count: int
    model_payload: Mapping[str, Any] | None
    receipt: Mapping[str, Any] | None
    proposal: ObjectSceneAnchorCardProposal | None
    failure_code: str | None
    failure_type: str | None
    protocol_digest: str
    source_digest: str
    transport_source_digest: str
    model: str
    reasoning_effort: str
    model_digest: str
    launcher_digest: str
    policy_cache_binding: str
    model_catalog_digest: str
    no_tools_attestation_digest: str
    runtime_digest: str
    artifact_digest: str

    def __post_init__(self) -> None:
        if type(self.proposer_input) is not ObjectSceneAnchorCardProposerInput:
            raise TypeError("artifact proposer input has the wrong type")
        for label, item in (
            ("input digest", self.input_digest), ("protocol digest", self.protocol_digest),
            ("source digest", self.source_digest), ("transport source digest", self.transport_source_digest),
            ("model digest", self.model_digest), ("launcher digest", self.launcher_digest),
            ("model catalog digest", self.model_catalog_digest),
            ("no-tools digest", self.no_tools_attestation_digest),
            ("runtime digest", self.runtime_digest), ("artifact digest", self.artifact_digest),
        ):
            _digest(item, label)
        if self.policy_cache_binding != "absent":
            _address(self.policy_cache_binding, "policy-cache binding")
        if (
            self.proposer_input.input_digest != self.input_digest
            or self.status not in ("success", "parser_error", "transport_error")
            or self.physical_call_count != 1
            or self.protocol_digest != object_scene_anchor_card_proposer_protocol_digest()
            or self.source_digest != object_scene_anchor_card_proposer_source_digest()
            or self.transport_source_digest != object_scene_anchor_card_proposer_transport_source_digest()
            or self.model_digest != _model_digest(self.model, self.reasoning_effort)
            or self.runtime_digest != _runtime_from_pins(
                model=self.model, reasoning_effort=self.reasoning_effort,
                launcher=self.launcher_digest, policy=self.policy_cache_binding,
                catalog=self.model_catalog_digest, attestation=self.no_tools_attestation_digest,
            )
        ):
            raise ObjectSceneAnchorCardProposerError("proposer artifact protocol differs")
        prompt = object_scene_anchor_card_proposer_prompt(self.proposer_input)
        schema = object_scene_anchor_card_proposer_output_schema(self.proposer_input)
        if self.status == "transport_error":
            if (
                self.model_payload is not None or self.receipt is not None or self.proposal is not None
                or self.failure_code != "transport_failed" or not isinstance(self.failure_type, str)
            ):
                raise ObjectSceneAnchorCardProposerError("transport-error outcome differs")
        else:
            if not isinstance(self.model_payload, Mapping) or not isinstance(self.receipt, Mapping):
                raise ObjectSceneAnchorCardProposerError("receipted outcome differs")
            payload = _canonical_payload(self.model_payload)
            receipt = _canonical_payload(self.receipt)
            object.__setattr__(self, "model_payload", payload)
            object.__setattr__(self, "receipt", receipt)
            _validate_receipt(
                receipt, self.proposer_input, payload, prompt, schema,
                model=self.model, reasoning_effort=self.reasoning_effort,
                launcher=self.launcher_digest, policy=self.policy_cache_binding,
                catalog=self.model_catalog_digest, attestation=self.no_tools_attestation_digest,
            )
            if self.status == "success":
                if (
                    type(self.proposal) is not ObjectSceneAnchorCardProposal
                    or self.failure_code is not None or self.failure_type is not None
                    or _translate_raw_payload(payload, self.proposer_input) != self.proposal
                ):
                    raise ObjectSceneAnchorCardProposerError("successful proposer outcome differs")
            else:
                if self.proposal is not None or self.failure_code != "payload_rejected" or not isinstance(self.failure_type, str):
                    raise ObjectSceneAnchorCardProposerError("parser-error outcome differs")
                try:
                    _translate_raw_payload(payload, self.proposer_input)
                except Exception as exc:
                    if _exception_type(exc) != self.failure_type:
                        raise ObjectSceneAnchorCardProposerError("parser failure replay differs") from exc
                else:
                    raise ObjectSceneAnchorCardProposerError("parser-error payload now succeeds")
        if self.artifact_digest != canonical_digest(_artifact_content(self)):
            raise ObjectSceneAnchorCardProposerError("proposer artifact digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_artifact_content(self), "artifact_digest": self.artifact_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorCardProposerArtifact":
        raw = _fields(
            value,
            {
                "schema", "proposer_input", "input_digest", "status", "physical_call_count",
                "model_payload", "receipt", "proposal", "failure_code", "failure_type",
                "protocol_digest", "source_digest", "transport_source_digest", "model",
                "reasoning_effort", "model_digest", "launcher_digest", "policy_cache_binding",
                "model_catalog_digest", "no_tools_attestation_digest", "runtime_digest",
                *_authority_data(), "artifact_digest",
            },
            "proposer artifact",
        )
        if raw["schema"] != OBJECT_SCENE_ANCHOR_CARD_PROPOSER_ARTIFACT_SCHEMA or any(
            raw[key] != item for key, item in _authority_data().items()
        ) or not isinstance(raw["proposer_input"], Mapping):
            raise ObjectSceneAnchorCardProposerError("proposer artifact policy differs")
        result = cls(
            ObjectSceneAnchorCardProposerInput.from_data(raw["proposer_input"]), raw["input_digest"],
            raw["status"], raw["physical_call_count"],
            None if raw["model_payload"] is None else dict(raw["model_payload"]),
            None if raw["receipt"] is None else dict(raw["receipt"]),
            None if raw["proposal"] is None else ObjectSceneAnchorCardProposal.from_data(raw["proposal"]),
            raw["failure_code"], raw["failure_type"], raw["protocol_digest"], raw["source_digest"],
            raw["transport_source_digest"], raw["model"], raw["reasoning_effort"], raw["model_digest"],
            raw["launcher_digest"], raw["policy_cache_binding"], raw["model_catalog_digest"],
            raw["no_tools_attestation_digest"], raw["runtime_digest"], raw["artifact_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorCardProposerError("proposer artifact is not canonical")
        return result


def _seal(
    *, proposer_input: ObjectSceneAnchorCardProposerInput, status: str,
    payload: Mapping[str, Any] | None, receipt: Mapping[str, Any] | None,
    proposal: ObjectSceneAnchorCardProposal | None,
    failure_code: str | None, failure_type: str | None,
    model: str, reasoning_effort: str, launcher: str, policy: str,
    catalog: str, attestation: str,
) -> ObjectSceneAnchorCardProposerArtifact:
    values = {
        "proposer_input": proposer_input,
        "input_digest": proposer_input.input_digest,
        "status": status,
        "physical_call_count": 1,
        "model_payload": None if payload is None else _canonical_payload(payload),
        "receipt": None if receipt is None else _canonical_payload(receipt),
        "proposal": proposal,
        "failure_code": failure_code,
        "failure_type": failure_type,
        "protocol_digest": object_scene_anchor_card_proposer_protocol_digest(),
        "source_digest": object_scene_anchor_card_proposer_source_digest(),
        "transport_source_digest": object_scene_anchor_card_proposer_transport_source_digest(),
        "model": model,
        "reasoning_effort": reasoning_effort,
        "model_digest": _model_digest(model, reasoning_effort),
        "launcher_digest": launcher,
        "policy_cache_binding": policy,
        "model_catalog_digest": catalog,
        "no_tools_attestation_digest": attestation,
        "runtime_digest": _runtime_from_pins(
            model=model, reasoning_effort=reasoning_effort, launcher=launcher,
            policy=policy, catalog=catalog, attestation=attestation,
        ),
    }
    provisional = object.__new__(ObjectSceneAnchorCardProposerArtifact)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorCardProposerArtifact(
        **values, artifact_digest=canonical_digest(_artifact_content(provisional))
    )


def propose_object_scene_anchor_cards(
    side0: Sequence[ObjectSceneAnchorCardProposerPanelInput],
    side1: Sequence[ObjectSceneAnchorCardProposerPanelInput],
    *, proposer_input: ObjectSceneAnchorCardProposerInput,
    expected_input_digest: str,
    model: str,
    reasoning_effort: str,
    expected_launcher_digest: str,
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
    minutes: int = 15,
    verbose: bool = False,
    executable: str = "codex",
    transport: StructuredTransport = run_codex_structured,
) -> ObjectSceneAnchorCardProposerArtifact:
    frozen = freeze_object_scene_anchor_card_proposer_input(side0, side1)
    if type(proposer_input) is not ObjectSceneAnchorCardProposerInput or frozen != proposer_input:
        raise ObjectSceneAnchorCardProposerError("runtime sheets differ from frozen proposer input")
    if frozen.input_digest != _digest(expected_input_digest, "expected input digest"):
        raise ObjectSceneAnchorCardProposerError("proposer input differs from commitment")
    runtime_digest = _runtime(
        model=model, reasoning_effort=reasoning_effort, launcher=expected_launcher_digest,
        policy_snapshot=cloud_policy_cache_snapshot, catalog_snapshot=model_catalog_snapshot,
        attestation=no_tools_attestation,
    )
    del runtime_digest
    if not callable(transport):
        raise TypeError("transport must be callable")
    prompt = object_scene_anchor_card_proposer_prompt(frozen)
    schema = object_scene_anchor_card_proposer_output_schema(frozen)
    rows = _source_rows(side0, side1)
    pins = {
        "model": model, "reasoning_effort": reasoning_effort,
        "launcher": expected_launcher_digest, "policy": cloud_policy_cache_snapshot.binding,
        "catalog": model_catalog_snapshot.raw_digest,
        "attestation": no_tools_attestation.attestation_digest,
    }
    with tempfile.TemporaryDirectory(prefix="bongard-anchor-card-call-") as raw:
        paths = _write_standard_view(Path(raw), rows)
        try:
            result = transport(
                prompt, paths, schema, model=model, reasoning_effort=reasoning_effort,
                minutes=minutes, verbose=verbose, executable=executable,
                cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
                model_catalog_snapshot=model_catalog_snapshot,
                expected_launcher_digest=expected_launcher_digest,
                tool_surface_attestation=no_tools_attestation,
                expected_tool_surface_attestation_digest=no_tools_attestation.attestation_digest,
            )
            if not isinstance(result, CodexStructuredResult):
                raise ObjectSceneAnchorCardProposerError("structured transport returned wrong type")
            payload = _canonical_payload(result.payload)
            if not isinstance(result.receipt, CodexReceipt):
                raise ObjectSceneAnchorCardProposerError("structured transport returned no receipt")
            receipt = result.receipt.to_dict()
            _validate_receipt(receipt, frozen, payload, prompt, schema, **pins)
        except Exception as exc:
            return _seal(
                proposer_input=frozen, status="transport_error", payload=None, receipt=None,
                proposal=None, failure_code="transport_failed", failure_type=_exception_type(exc),
                **pins,
            )
    try:
        proposal = _translate_raw_payload(payload, frozen)
    except Exception as exc:
        return _seal(
            proposer_input=frozen, status="parser_error", payload=payload, receipt=receipt,
            proposal=None, failure_code="payload_rejected", failure_type=_exception_type(exc),
            **pins,
        )
    return _seal(
        proposer_input=frozen, status="success", payload=payload, receipt=receipt,
        proposal=proposal, failure_code=None, failure_type=None, **pins,
    )


def verify_object_scene_anchor_card_proposer_artifact(
    artifact: ObjectSceneAnchorCardProposerArtifact,
    side0: Sequence[ObjectSceneAnchorCardProposerPanelInput],
    side1: Sequence[ObjectSceneAnchorCardProposerPanelInput],
    *, expected_artifact_digest: str, expected_input_digest: str,
    model: str, reasoning_effort: str, expected_launcher_digest: str,
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
) -> ObjectSceneAnchorCardProposerArtifact:
    if type(artifact) is not ObjectSceneAnchorCardProposerArtifact:
        raise TypeError("artifact must be exact proposer artifact")
    restored = ObjectSceneAnchorCardProposerArtifact.from_data(artifact.to_data())
    if restored.artifact_digest != _digest(expected_artifact_digest, "expected artifact digest"):
        raise ObjectSceneAnchorCardProposerError("proposer artifact differs from commitment")
    replayed_input = freeze_object_scene_anchor_card_proposer_input(side0, side1)
    if (
        replayed_input.input_digest != _digest(expected_input_digest, "expected input digest")
        or replayed_input != restored.proposer_input
    ):
        raise ObjectSceneAnchorCardProposerError("proposer input bytes differ on cold replay")
    expected_runtime = _runtime(
        model=model, reasoning_effort=reasoning_effort, launcher=expected_launcher_digest,
        policy_snapshot=cloud_policy_cache_snapshot, catalog_snapshot=model_catalog_snapshot,
        attestation=no_tools_attestation,
    )
    if (
        restored.model != model or restored.reasoning_effort != reasoning_effort
        or restored.launcher_digest != expected_launcher_digest
        or restored.policy_cache_binding != cloud_policy_cache_snapshot.binding
        or restored.model_catalog_digest != model_catalog_snapshot.raw_digest
        or restored.no_tools_attestation_digest != no_tools_attestation.attestation_digest
        or restored.runtime_digest != expected_runtime
    ):
        raise ObjectSceneAnchorCardProposerError("proposer runtime differs on cold replay")
    return restored


__all__ = (
    "OBJECT_SCENE_ANCHOR_CARD_PROPOSER_ARTIFACT_SCHEMA",
    "OBJECT_SCENE_ANCHOR_CARD_PROPOSER_INPUT_SCHEMA",
    "OBJECT_SCENE_ANCHOR_CARD_PROPOSER_PANEL_SCHEMA",
    "OBJECT_SCENE_ANCHOR_CARD_PROPOSER_PROTOCOL_ID",
    "ObjectSceneAnchorCardProposerArtifact",
    "ObjectSceneAnchorCardProposerError",
    "ObjectSceneAnchorCardProposerInput",
    "ObjectSceneAnchorCardProposerPanel",
    "ObjectSceneAnchorCardProposerPanelInput",
    "freeze_object_scene_anchor_card_proposer_input",
    "object_scene_anchor_card_proposer_output_schema",
    "object_scene_anchor_card_proposer_prompt",
    "object_scene_anchor_card_proposer_protocol_digest",
    "object_scene_anchor_card_proposer_source_digest",
    "object_scene_anchor_card_proposer_transport_source_digest",
    "propose_object_scene_anchor_cards",
    "verify_object_scene_anchor_card_proposer_artifact",
)
