"""Query-free affirmative atom slate, independent panel scores, and Python search.

One support-only proposer sees exactly six Group A and six Group B drawings and
returns eight bounded affirmative visual-predicate phrases.  A separate one-panel
observer rates every frozen atom independently.  Only after all twelve rows
exist does deterministic Python enumerate the eight singletons and all
twenty-eight two-atom conjunctions.  No query API, negative concept, ``Not``,
polarity choice, threshold choice, executable prose, or Lean dependency exists
in this module.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

from bongard import prototype_scene_observer as _scene_runtime
from bongard.canonical import canonical_digest, canonical_json
from bongard.codex_no_tools_preflight import CodexNoToolsAttestation
from bongard.evidence import Disposition
from bongard.object_bongard_turn_journal import (
    ObjectBongardNamedImageTurnJournalTransport,
    ObjectBongardTurnJournalSummary,
    object_bongard_turn_journal_source_digest,
)
from bongard.panel_typed_codex_observer import (
    TypedCodexRuntimeBinding,
    _bind_runtime,
    _digest,
    _exact_png,
    _receipt_from_data,
    _validate_receipt_binding,
)
from bongard.prototype_scene_observer import PrototypeImageIdentity
from bongard.transport import (
    CloudPolicyCacheSnapshot,
    CodexModelCatalogSnapshot,
    CodexReceipt,
    run_codex_named_images_structured,
    validate_codex_named_image_receipt,
    validate_codex_strict_output_schema,
)


ATOM_COUNT = 8
GROUP_SIZE = 6
SUPPORT_PANEL_COUNT = 12
FORMULA_COUNT = ATOM_COUNT + ATOM_COUNT * (ATOM_COUNT - 1) // 2
ATOM_IDS = tuple(f"atom_{index:02d}" for index in range(ATOM_COUNT))
PRESENT_LOWER_BOUND = 3
ABSENT_UPPER_BOUND = 1
MINIMUM_DECISIVE_PER_SIDE = 5

ATOM_SLATE_PROTOCOL_ID = "bongard.positive-atom-slate/support-only-eight-v1"
ATOM_SLATE_SCHEMA = "gkm.bongard-positive-atom-slate.v1"
ATOM_SLATE_REQUEST_SCHEMA = "gkm.bongard-positive-atom-slate-request.v1"
ATOM_SLATE_ARTIFACT_SCHEMA = "gkm.bongard-positive-atom-slate-artifact.v1"
ATOM_PANEL_REQUEST_SCHEMA = "gkm.bongard-positive-atom-panel-request.v1"
ATOM_PANEL_ROW_SCHEMA = "gkm.bongard-positive-atom-panel-row.v1"
ATOM_PANEL_ARTIFACT_SCHEMA = "gkm.bongard-positive-atom-panel-artifact.v1"
ATOM_FORMULA_SCHEMA = "gkm.bongard-positive-atom-formula.v1"
ATOM_PROFILE_SCHEMA = "gkm.bongard-positive-atom-support-profile.v1"
ATOM_INVENTORY_SCHEMA = "gkm.bongard-positive-atom-inventory.v1"
ATOM_GAP_SCHEMA = "gkm.bongard-positive-atom-gap.v1"
ATOM_TRANSPORT_SCHEMA = "gkm.bongard-positive-atom-transport.v1"

PROPOSER_IMAGE_NAMES = tuple(
    [f"group_a_{index:02d}.png" for index in range(GROUP_SIZE)]
    + [f"group_b_{index:02d}.png" for index in range(GROUP_SIZE)]
)
PANEL_IMAGE_NAME = "panel.png"
MAX_ATOM_BYTES = 240
MAX_RESPONSE_BYTES = 8 * 1024

_RAW_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ERROR_CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_FORBIDDEN_ATOM = re.compile(
    r"(?:```|`|[{};]|[<>]=?|[≤≥]|%|"
    r"\b(?:and|or|not|no|without|lack|lacks|lacking|absence|absent|"
    r"negative|complement|foil|opposite|fails|failure|exclude|excludes|"
    r"threshold|cutoff|score|probability|confidence|percent|"
    r"at\s+least|at\s+most|more\s+than|less\s+than|above|below|"
    r"def|lambda|import|exec|eval|return|function|python|javascript|sql|regex|"
    r"task|phase|side|class|support|query|group|panel|image|candidate|formula)\b)",
    re.IGNORECASE,
)


class PositiveAtomSlateError(ValueError):
    """An atom slate, panel score, transport, or deterministic inventory differs."""


def panel_positive_atom_slate_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected
    ):
        raise PositiveAtomSlateError(f"{label} fields differ")
    return value


def _raw_digest(value: object, label: str) -> str:
    if type(value) is not str or _RAW_DIGEST.fullmatch(value) is None:
        raise PositiveAtomSlateError(f"{label} must be a raw SHA-256")
    return value


def _address(value: object, label: str) -> str:
    if type(value) is not str or _ADDRESS.fullmatch(value) is None:
        raise PositiveAtomSlateError(f"{label} must be a sha256: address")
    return value


def _canonical_payload(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise PositiveAtomSlateError(f"{label} must be an object")
    try:
        result = json.loads(canonical_json(dict(value)).decode("utf-8"))
    except Exception as exc:
        raise PositiveAtomSlateError(f"{label} is not canonical JSON") from exc
    if type(result) is not dict:
        raise PositiveAtomSlateError(f"{label} must be an object")
    return result


def _atom_text(value: object, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise PositiveAtomSlateError(f"{label} must be nonempty trimmed prose")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError as exc:
        raise PositiveAtomSlateError(f"{label} must be UTF-8") from exc
    if len(encoded) > MAX_ATOM_BYTES or any(ord(char) < 32 for char in value):
        raise PositiveAtomSlateError(f"{label} exceeds its prose bound")
    if _FORBIDDEN_ATOM.search(value) is not None:
        raise PositiveAtomSlateError(
            f"{label} contains composition, negation, policy, identifiers, or code"
        )
    return value


def _freeze_group(value: Sequence[bytes], label: str) -> tuple[bytes, ...]:
    if isinstance(value, (bytes, bytearray, str)) or len(value) != GROUP_SIZE:
        raise PositiveAtomSlateError(f"{label} must contain exactly six PNGs")
    return tuple(_exact_png(item, f"{label} PNG {index}") for index, item in enumerate(value))


@dataclass(frozen=True, slots=True)
class AffirmativeAtomSlate:
    atoms: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.atoms) is not tuple or len(self.atoms) != ATOM_COUNT:
            raise PositiveAtomSlateError("affirmative slate must contain exactly eight atoms")
        checked = tuple(
            _atom_text(item, f"affirmative atom {index}")
            for index, item in enumerate(self.atoms)
        )
        if len({item.casefold() for item in checked}) != ATOM_COUNT:
            raise PositiveAtomSlateError("affirmative slate atoms must be distinct")

    @property
    def slate_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": ATOM_SLATE_SCHEMA,
            "atom_ids": list(ATOM_IDS),
            "atoms": list(self.atoms),
            "atom_count": ATOM_COUNT,
            "atom_contract": "opaque_affirmative_visual_predicate_prose",
            "explicit_boolean_composition_lexically_rejected": True,
            "explicit_negation_lexically_rejected": True,
            "semantic_atomicity_mechanically_proven": False,
            "logical_negation_operator_present": False,
            "threshold_or_polarity_fields_present": False,
            "lean_present": False,
        }

    @classmethod
    def from_data(cls, value: object) -> "AffirmativeAtomSlate":
        raw = _fields(
            value,
            {
                "schema", "atom_ids", "atoms", "atom_count",
                "atom_contract", "explicit_boolean_composition_lexically_rejected",
                "explicit_negation_lexically_rejected",
                "semantic_atomicity_mechanically_proven",
                "logical_negation_operator_present",
                "threshold_or_polarity_fields_present", "lean_present",
            },
            "affirmative atom slate",
        )
        if (
            raw["schema"] != ATOM_SLATE_SCHEMA
            or raw["atom_ids"] != list(ATOM_IDS)
            or type(raw["atoms"]) is not list
            or raw["atom_count"] != ATOM_COUNT
            or raw["atom_contract"] != "opaque_affirmative_visual_predicate_prose"
            or raw["explicit_boolean_composition_lexically_rejected"] is not True
            or raw["explicit_negation_lexically_rejected"] is not True
            or raw["semantic_atomicity_mechanically_proven"] is not False
            or raw["logical_negation_operator_present"] is not False
            or raw["threshold_or_polarity_fields_present"] is not False
            or raw["lean_present"] is not False
        ):
            raise PositiveAtomSlateError("affirmative atom slate policy differs")
        result = cls(tuple(raw["atoms"]))
        if result.to_data() != dict(raw):
            raise PositiveAtomSlateError("affirmative atom slate is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class AtomSlateProposerRequest:
    runtime: TypedCodexRuntimeBinding
    presentation: tuple[PrototypeImageIdentity, ...]

    def __post_init__(self) -> None:
        if type(self.runtime) is not TypedCodexRuntimeBinding:
            raise TypeError("atom slate request needs TypedCodexRuntimeBinding")
        if (
            type(self.presentation) is not tuple
            or len(self.presentation) != SUPPORT_PANEL_COUNT
            or any(type(item) is not PrototypeImageIdentity for item in self.presentation)
            or tuple(item.name for item in self.presentation) != PROPOSER_IMAGE_NAMES
        ):
            raise PositiveAtomSlateError("atom slate request presentation differs")

    @classmethod
    def build(
        cls,
        group_a_pngs: Sequence[bytes],
        group_b_pngs: Sequence[bytes],
        *,
        runtime: TypedCodexRuntimeBinding,
    ) -> "AtomSlateProposerRequest":
        first = _freeze_group(group_a_pngs, "group A")
        second = _freeze_group(group_b_pngs, "group B")
        presentation = tuple(
            PrototypeImageIdentity(name, len(raw), hashlib.sha256(raw).hexdigest())
            for name, raw in zip(PROPOSER_IMAGE_NAMES, (*first, *second), strict=True)
        )
        return cls(runtime, presentation)

    @property
    def request_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": ATOM_SLATE_REQUEST_SCHEMA,
            "protocol_id": ATOM_SLATE_PROTOCOL_ID,
            "runtime": self.runtime.to_data(),
            "presentation": [item.to_data() for item in self.presentation],
            "group_sizes": [GROUP_SIZE, GROUP_SIZE],
            "atom_slots": ATOM_COUNT,
            "query_image_count": 0,
            "model_call_count": 1,
            "group_b_may_be_heterogeneous": True,
            "dataset_ids_model_visible": False,
            "positive_vs_contrast_role_model_visible": True,
            "negative_class_description_output_field_present": False,
        }

    @classmethod
    def from_data(cls, value: object) -> "AtomSlateProposerRequest":
        raw = _fields(
            value,
            {
                "schema", "protocol_id", "runtime", "presentation", "group_sizes",
                "atom_slots", "query_image_count", "model_call_count",
                "group_b_may_be_heterogeneous", "dataset_ids_model_visible",
                "positive_vs_contrast_role_model_visible",
                "negative_class_description_output_field_present",
            },
            "atom slate proposer request",
        )
        if (
            raw["schema"] != ATOM_SLATE_REQUEST_SCHEMA
            or raw["protocol_id"] != ATOM_SLATE_PROTOCOL_ID
            or type(raw["presentation"]) is not list
            or raw["group_sizes"] != [GROUP_SIZE, GROUP_SIZE]
            or raw["atom_slots"] != ATOM_COUNT
            or raw["query_image_count"] != 0
            or raw["model_call_count"] != 1
            or raw["group_b_may_be_heterogeneous"] is not True
            or raw["dataset_ids_model_visible"] is not False
            or raw["positive_vs_contrast_role_model_visible"] is not True
            or raw["negative_class_description_output_field_present"] is not False
        ):
            raise PositiveAtomSlateError("atom slate request policy differs")
        result = cls(
            TypedCodexRuntimeBinding.from_data(raw["runtime"]),
            tuple(PrototypeImageIdentity.from_data(item) for item in raw["presentation"]),
        )
        if result.to_data() != dict(raw):
            raise PositiveAtomSlateError("atom slate request is not canonical")
        return result


def atom_slate_proposer_prompt(request: AtomSlateProposerRequest) -> str:
    if type(request) is not AtomSlateProposerRequest:
        raise TypeError("atom slate prompt needs AtomSlateProposerRequest")
    first = ", ".join(PROPOSER_IMAGE_NAMES[:GROUP_SIZE])
    second = ", ".join(PROPOSER_IMAGE_NAMES[GROUP_SIZE:])
    return (
        "Inspect exactly twelve complete drawings in two disclosed support groups. "
        f"Group A contains {first}. Group B contains {second}. Group B may be "
        "heterogeneous; never infer or describe one shared Group B concept. Return "
        "exactly eight distinct affirmative visual atoms in atom_00 through atom_07. "
        "Each atom must name one visibly testable property that may help describe "
        "Group A. Include alternative structural hypotheses when uncertain. Describe "
        "latent carrier geometry rather than counting zigzags, dots, circles, squares, "
        "triangles, or rendering-style transitions as structural parts. Each returned "
        "atom must be bounded plain prose with no conjunction, disjunction, negation, "
        "absence, foil, negative-class description, threshold, score, polarity, code, "
        "or dataset identifier. Do not select or combine atoms. Python will score every "
        "atom independently on every support drawing and only then enumerate all "
        "singletons and two-atom conjunctions. No query drawing exists in this call."
    )


def atom_slate_proposer_output_schema(
    request: AtomSlateProposerRequest | None = None,
) -> dict[str, object]:
    if request is not None and type(request) is not AtomSlateProposerRequest:
        raise TypeError("atom slate schema request differs")
    properties = {name: {"type": "string"} for name in ATOM_IDS}
    schema: dict[str, object] = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    validate_codex_strict_output_schema(schema)
    return schema


def _parse_slate_payload(value: object) -> AffirmativeAtomSlate:
    raw = _fields(value, set(ATOM_IDS), "atom slate payload")
    return AffirmativeAtomSlate(tuple(raw[name] for name in ATOM_IDS))


def _transport_source_binding(kind: str) -> str:
    if kind == "production_direct":
        body: dict[str, object] = {
            "kind": kind,
            "transport_source_digest": _scene_runtime.prototype_scene_transport_source_digest(),
        }
    elif kind == "production_exactly_once_journal":
        body = {
            "kind": kind,
            "journal_source_digest": object_bongard_turn_journal_source_digest(),
            "transport_source_digest": _scene_runtime.prototype_scene_transport_source_digest(),
        }
    elif kind == "injected_unverified":
        body = {"kind": kind, "callable_source_identity_verified": False}
    else:
        raise PositiveAtomSlateError("atom slate transport kind differs")
    return "sha256:" + canonical_digest(
        {"schema": "gkm.bongard-positive-atom-transport-source.v1", **body}
    )


@dataclass(frozen=True, slots=True)
class AtomTransportCustody:
    kind: str
    source_binding: str
    production_transport_chain_verified: bool
    benchmark_sealable: bool
    journal_terminal_status: str | None = None
    journal_manifest_digest: str | None = None
    journal_turn_key: str | None = None
    journal_claim_digest: str | None = None
    journal_result_digest: str | None = None
    journal_outcome_digest: str | None = None
    journal_terminal_record_digest: str | None = None

    def __post_init__(self) -> None:
        journal = self.kind == "production_exactly_once_journal"
        production = self.kind in {"production_direct", "production_exactly_once_journal"}
        if (
            self.kind not in {
                "production_direct", "production_exactly_once_journal", "injected_unverified"
            }
            or self.source_binding != _transport_source_binding(self.kind)
            or self.production_transport_chain_verified is not production
            or (not journal and self.benchmark_sealable is not False)
        ):
            raise PositiveAtomSlateError("atom slate transport custody differs")
        pins = (
            self.journal_manifest_digest, self.journal_turn_key,
            self.journal_claim_digest, self.journal_result_digest,
            self.journal_outcome_digest, self.journal_terminal_record_digest,
        )
        if journal:
            if (
                self.journal_terminal_status not in {"success", "failure"}
                or any(type(item) is not str or _ADDRESS.fullmatch(item) is None for item in pins)
                or self.benchmark_sealable is not (
                    self.journal_terminal_status == "success"
                )
            ):
                raise PositiveAtomSlateError("atom slate journal pins differ")
        elif self.journal_terminal_status is not None or any(item is not None for item in pins):
            raise PositiveAtomSlateError("non-journal atom transport names journal pins")

    @classmethod
    def create(
        cls,
        kind: str,
        *,
        journal_summary: ObjectBongardTurnJournalSummary | None = None,
    ) -> "AtomTransportCustody":
        if kind == "production_exactly_once_journal":
            if type(journal_summary) is not ObjectBongardTurnJournalSummary:
                raise PositiveAtomSlateError("atom slate journal terminal is not exact")
            return cls(
                kind, _transport_source_binding(kind), True,
                journal_summary.terminal_status == "success",
                journal_summary.terminal_status,
                journal_summary.manifest_digest, journal_summary.turn_key,
                journal_summary.claim_digest, journal_summary.result_digest,
                journal_summary.outcome_digest, journal_summary.record_digest,
            )
        if journal_summary is not None:
            raise PositiveAtomSlateError("non-journal atom transport received a terminal")
        return cls(kind, _transport_source_binding(kind), kind == "production_direct", False)

    def to_data(self) -> dict[str, object]:
        return {
            "schema": ATOM_TRANSPORT_SCHEMA,
            "kind": self.kind,
            "source_binding": self.source_binding,
            "production_transport_chain_verified": self.production_transport_chain_verified,
            "benchmark_sealable": self.benchmark_sealable,
            "journal_terminal_status": self.journal_terminal_status,
            "journal_manifest_digest": self.journal_manifest_digest,
            "journal_turn_key": self.journal_turn_key,
            "journal_claim_digest": self.journal_claim_digest,
            "journal_result_digest": self.journal_result_digest,
            "journal_outcome_digest": self.journal_outcome_digest,
            "journal_terminal_record_digest": self.journal_terminal_record_digest,
            "external_terminal_required_for_cold_replay": True,
        }

    @classmethod
    def from_data(cls, value: object) -> "AtomTransportCustody":
        raw = _fields(
            value,
            {
                "schema", "kind", "source_binding", "production_transport_chain_verified",
                "benchmark_sealable", "journal_terminal_status", "journal_manifest_digest",
                "journal_turn_key", "journal_claim_digest", "journal_result_digest",
                "journal_outcome_digest", "journal_terminal_record_digest",
                "external_terminal_required_for_cold_replay",
            },
            "atom transport custody",
        )
        if (
            raw["schema"] != ATOM_TRANSPORT_SCHEMA
            or raw["external_terminal_required_for_cold_replay"] is not True
        ):
            raise PositiveAtomSlateError("atom transport custody policy differs")
        result = cls(
            raw["kind"], raw["source_binding"],
            raw["production_transport_chain_verified"], raw["benchmark_sealable"],
            raw["journal_terminal_status"],
            raw["journal_manifest_digest"], raw["journal_turn_key"],
            raw["journal_claim_digest"], raw["journal_result_digest"],
            raw["journal_outcome_digest"], raw["journal_terminal_record_digest"],
        )
        if result.to_data() != dict(raw):
            raise PositiveAtomSlateError("atom transport custody is not canonical")
        return result


def _transport_custody(transport: object) -> AtomTransportCustody:
    if transport is run_codex_named_images_structured:
        return AtomTransportCustody.create("production_direct")
    if (
        type(transport) is ObjectBongardNamedImageTurnJournalTransport
        and getattr(transport, "_underlying_transport", None)
        is run_codex_named_images_structured
        and transport.runtime.transport_source_digest
        == _scene_runtime.prototype_scene_transport_source_digest()
    ):
        return AtomTransportCustody.create(
            "production_exactly_once_journal", journal_summary=transport.verify()
        )
    return AtomTransportCustody.create("injected_unverified")


def _verify_external_terminal(
    custody: AtomTransportCustody,
    summary: ObjectBongardTurnJournalSummary | None,
) -> None:
    if custody.kind != "production_exactly_once_journal":
        if summary is not None:
            raise PositiveAtomSlateError("non-journal atom artifact received a terminal")
        return
    if (
        type(summary) is not ObjectBongardTurnJournalSummary
        or summary.terminal_status != custody.journal_terminal_status
        or (
            summary.manifest_digest, summary.turn_key, summary.claim_digest,
            summary.result_digest, summary.outcome_digest, summary.record_digest,
        )
        != (
            custody.journal_manifest_digest, custody.journal_turn_key,
            custody.journal_claim_digest, custody.journal_result_digest,
            custody.journal_outcome_digest, custody.journal_terminal_record_digest,
        )
    ):
        raise PositiveAtomSlateError("external atom journal terminal differs")


def _slate_contract_digest(request: AtomSlateProposerRequest) -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-positive-atom-slate-contract.v1",
            "protocol_id": ATOM_SLATE_PROTOCOL_ID,
            "source_digest": panel_positive_atom_slate_source_digest(),
            "request_digest": request.request_digest,
            "atom_slots": ATOM_COUNT,
            "support_panels": SUPPORT_PANEL_COUNT,
            "query_panels": 0,
            "negative_class_description_output_field_allowed": False,
            "logical_negation_or_disjunction_operator_allowed": False,
            "composition_by_model_allowed": False,
            "threshold_or_polarity_selection_allowed": False,
            "lean_present": False,
        }
    )


@dataclass(frozen=True, slots=True)
class AtomSlateProposerArtifact:
    runtime: TypedCodexRuntimeBinding
    request: AtomSlateProposerRequest
    transport_custody: AtomTransportCustody
    request_digest: str
    source_digest: str
    contract_digest: str
    prompt_digest: str
    output_schema_digest: str
    payload_digest: str
    model_payload: Mapping[str, Any]
    receipt: CodexReceipt
    slate: AffirmativeAtomSlate

    def __post_init__(self) -> None:
        if type(self.runtime) is not TypedCodexRuntimeBinding:
            raise TypeError("atom slate artifact needs typed runtime")
        if type(self.request) is not AtomSlateProposerRequest:
            raise TypeError("atom slate artifact needs exact request")
        if type(self.transport_custody) is not AtomTransportCustody:
            raise TypeError("atom slate artifact needs transport custody")
        if (
            self.transport_custody.kind == "production_exactly_once_journal"
            and self.transport_custody.journal_terminal_status != "success"
        ):
            raise PositiveAtomSlateError("successful atom slate lacks success terminal")
        if type(self.slate) is not AffirmativeAtomSlate:
            raise TypeError("atom slate artifact needs exact slate")
        for label, value in (
            ("request digest", self.request_digest),
            ("source digest", self.source_digest),
            ("contract digest", self.contract_digest),
            ("prompt digest", self.prompt_digest),
            ("output schema digest", self.output_schema_digest),
            ("payload digest", self.payload_digest),
        ):
            _raw_digest(value, label)
        payload = _canonical_payload(self.model_payload, "atom slate model payload")
        object.__setattr__(self, "model_payload", payload)
        prompt = atom_slate_proposer_prompt(self.request)
        schema = atom_slate_proposer_output_schema(self.request)
        if (
            self.runtime != self.request.runtime
            or self.request_digest != self.request.request_digest
            or self.source_digest != panel_positive_atom_slate_source_digest()
            or self.contract_digest != _slate_contract_digest(self.request)
            or self.prompt_digest != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            or self.output_schema_digest != canonical_digest(schema)
            or self.payload_digest != canonical_digest(payload)
            or self.slate != _parse_slate_payload(payload)
        ):
            raise PositiveAtomSlateError("atom slate artifact envelope differs")
        _validate_receipt_binding(
            self.receipt,
            runtime=self.runtime,
            prompt_digest=self.prompt_digest,
            output_schema_digest=self.output_schema_digest,
            payload_digest=self.payload_digest,
            presentation=self.request.presentation,
        )

    @property
    def benchmark_sealable(self) -> bool:
        return self.transport_custody.benchmark_sealable

    @property
    def artifact_digest(self) -> str:
        return canonical_digest(self.content_data())

    def content_data(self) -> dict[str, object]:
        return {
            "schema": ATOM_SLATE_ARTIFACT_SCHEMA,
            "protocol_id": ATOM_SLATE_PROTOCOL_ID,
            "runtime": self.runtime.to_data(),
            "request": self.request.to_data(),
            "transport_custody": self.transport_custody.to_data(),
            "benchmark_sealable": self.benchmark_sealable,
            "request_digest": self.request_digest,
            "source_digest": self.source_digest,
            "contract_digest": self.contract_digest,
            "prompt_digest": self.prompt_digest,
            "output_schema_digest": self.output_schema_digest,
            "payload_digest": self.payload_digest,
            "model_payload": dict(self.model_payload),
            "receipt": self.receipt.to_dict(),
            "receipt_digest": self.receipt.receipt_digest,
            "slate": self.slate.to_data(),
            "slate_digest": self.slate.slate_digest,
            "support_image_count": SUPPORT_PANEL_COUNT,
            "query_image_count": 0,
            "model_call_count": 1,
            "model_selected_formula": False,
            "python_composition_occurs_after_panel_observations": True,
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "artifact_digest": self.artifact_digest}

    @classmethod
    def from_data(cls, value: object) -> "AtomSlateProposerArtifact":
        raw = _fields(
            value,
            {
                "schema", "protocol_id", "runtime", "request", "transport_custody",
                "benchmark_sealable", "request_digest", "source_digest", "contract_digest",
                "prompt_digest", "output_schema_digest", "payload_digest", "model_payload",
                "receipt", "receipt_digest", "slate", "slate_digest", "support_image_count",
                "query_image_count", "model_call_count", "model_selected_formula",
                "python_composition_occurs_after_panel_observations", "artifact_digest",
            },
            "atom slate proposer artifact",
        )
        if (
            raw["schema"] != ATOM_SLATE_ARTIFACT_SCHEMA
            or raw["protocol_id"] != ATOM_SLATE_PROTOCOL_ID
            or raw["support_image_count"] != SUPPORT_PANEL_COUNT
            or raw["query_image_count"] != 0
            or raw["model_call_count"] != 1
            or raw["model_selected_formula"] is not False
            or raw["python_composition_occurs_after_panel_observations"] is not True
        ):
            raise PositiveAtomSlateError("atom slate artifact policy differs")
        result = cls(
            TypedCodexRuntimeBinding.from_data(raw["runtime"]),
            AtomSlateProposerRequest.from_data(raw["request"]),
            AtomTransportCustody.from_data(raw["transport_custody"]),
            raw["request_digest"], raw["source_digest"], raw["contract_digest"],
            raw["prompt_digest"], raw["output_schema_digest"], raw["payload_digest"],
            _canonical_payload(raw["model_payload"], "archived atom slate payload"),
            _receipt_from_data(raw["receipt"]), AffirmativeAtomSlate.from_data(raw["slate"]),
        )
        if (
            raw["benchmark_sealable"] is not result.benchmark_sealable
            or raw["receipt_digest"] != result.receipt.receipt_digest
            or raw["slate_digest"] != result.slate.slate_digest
            or raw["artifact_digest"] != result.artifact_digest
            or result.to_data() != dict(raw)
        ):
            raise PositiveAtomSlateError("atom slate artifact digest differs")
        return result


def propose_affirmative_atom_slate(
    group_a_pngs: Sequence[bytes],
    group_b_pngs: Sequence[bytes],
    *,
    request: AtomSlateProposerRequest,
    model: str,
    reasoning_effort: str,
    minutes: int = 15,
    verbose: bool = False,
    executable: str = "codex",
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None = None,
    expected_launcher_digest: str,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
    transport=run_codex_named_images_structured,
) -> AtomSlateProposerArtifact:
    if type(request) is not AtomSlateProposerRequest:
        raise TypeError("atom slate call needs AtomSlateProposerRequest")
    first = _freeze_group(group_a_pngs, "group A")
    second = _freeze_group(group_b_pngs, "group B")
    rebuilt = AtomSlateProposerRequest.build(first, second, runtime=request.runtime)
    if rebuilt != request:
        raise PositiveAtomSlateError("atom slate request belongs to other pixels")
    runtime = _bind_runtime(
        model=model,
        reasoning_effort=reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
        model_catalog_snapshot=model_catalog_snapshot,
        no_tools_attestation=no_tools_attestation,
    )
    if runtime != request.runtime or type(minutes) is not int or minutes <= 0:
        raise PositiveAtomSlateError("atom slate call runtime or capacity differs")
    prompt = atom_slate_proposer_prompt(request)
    schema = atom_slate_proposer_output_schema(request)
    presentation = tuple(zip(PROPOSER_IMAGE_NAMES, (*first, *second), strict=True))
    try:
        payload, receipt = _scene_runtime._stage_and_call(
            presentation,
            prompt=prompt,
            schema=schema,
            model=model,
            reasoning_effort=reasoning_effort,
            minutes=minutes,
            verbose=verbose,
            executable=executable,
            cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
            expected_launcher_digest=expected_launcher_digest,
            model_catalog_snapshot=model_catalog_snapshot,
            no_tools_attestation=no_tools_attestation,
            transport=transport,
        )
        frozen = _canonical_payload(payload, "atom slate payload")
        if len(canonical_json(frozen)) > MAX_RESPONSE_BYTES:
            raise PositiveAtomSlateError("atom slate payload exceeds capacity")
        slate = _parse_slate_payload(frozen)
        custody = _transport_custody(transport)
        return AtomSlateProposerArtifact(
            runtime, request, custody, request.request_digest,
            panel_positive_atom_slate_source_digest(), _slate_contract_digest(request),
            hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            canonical_digest(schema), canonical_digest(frozen), frozen, receipt, slate,
        )
    except PositiveAtomSlateError:
        raise
    except Exception as exc:
        raise PositiveAtomSlateError("atom slate proposer failed closed") from exc


def verify_atom_slate_proposer_artifact(
    artifact: AtomSlateProposerArtifact,
    group_a_pngs: Sequence[bytes],
    group_b_pngs: Sequence[bytes],
    *,
    expected_artifact_digest: str,
    proposer_journal_terminal: ObjectBongardTurnJournalSummary | None = None,
) -> AtomSlateProposerArtifact:
    if type(artifact) is not AtomSlateProposerArtifact:
        raise TypeError("atom slate replay needs AtomSlateProposerArtifact")
    restored = AtomSlateProposerArtifact.from_data(artifact.to_data())
    if restored.artifact_digest != _raw_digest(expected_artifact_digest, "expected artifact"):
        raise PositiveAtomSlateError("atom slate artifact differs from commitment")
    _verify_external_terminal(restored.transport_custody, proposer_journal_terminal)
    first = _freeze_group(group_a_pngs, "group A")
    second = _freeze_group(group_b_pngs, "group B")
    rebuilt = AtomSlateProposerRequest.build(first, second, runtime=restored.runtime)
    if rebuilt != restored.request:
        raise PositiveAtomSlateError("atom slate replay pixels differ")
    prompt = atom_slate_proposer_prompt(rebuilt)
    schema = atom_slate_proposer_output_schema(rebuilt)
    with tempfile.TemporaryDirectory(prefix="bongard-atom-slate-replay-") as raw:
        directory = Path(raw)
        paths: list[str] = []
        for name, data in zip(PROPOSER_IMAGE_NAMES, (*first, *second), strict=True):
            target = directory / name
            target.write_bytes(data)
            paths.append(str(target.resolve()))
        try:
            validate_codex_named_image_receipt(
                restored.receipt, prompt, tuple(paths), PROPOSER_IMAGE_NAMES,
                schema, dict(restored.model_payload),
            )
        except Exception as exc:
            raise PositiveAtomSlateError("atom slate receipt replay failed") from exc
    return restored


@dataclass(frozen=True, order=True, slots=True)
class AtomScoreInterval:
    lower: int
    upper: int

    def __post_init__(self) -> None:
        if (
            type(self.lower) is not int
            or type(self.upper) is not int
            or not 0 <= self.lower <= self.upper <= 4
        ):
            raise PositiveAtomSlateError("atom score interval must lie in 0..4")

    @property
    def disposition(self) -> Disposition:
        if self.lower >= PRESENT_LOWER_BOUND:
            return Disposition.PRESENT
        if self.upper <= ABSENT_UPPER_BOUND:
            return Disposition.CERTIFIED_ABSENT
        return Disposition.INDETERMINATE

    def to_data(self) -> dict[str, object]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "disposition": self.disposition.value,
        }

    @classmethod
    def from_data(cls, value: object) -> "AtomScoreInterval":
        raw = _fields(value, {"lower", "upper", "disposition"}, "atom score interval")
        result = cls(raw["lower"], raw["upper"])
        if raw["disposition"] != result.disposition.value or result.to_data() != dict(raw):
            raise PositiveAtomSlateError("atom score interval disposition differs")
        return result


@dataclass(frozen=True, slots=True)
class AtomPanelScoreRow:
    """All eight independent atom judgments for one exposed support panel."""

    panel_ordinal: int
    slate_digest: str
    intervals: tuple[AtomScoreInterval | None, ...]
    error_code: str | None = None

    def __post_init__(self) -> None:
        if type(self.panel_ordinal) is not int or not 0 <= self.panel_ordinal < SUPPORT_PANEL_COUNT:
            raise PositiveAtomSlateError("atom panel ordinal differs")
        _raw_digest(self.slate_digest, "atom panel slate digest")
        if type(self.intervals) is not tuple or len(self.intervals) != ATOM_COUNT:
            raise PositiveAtomSlateError("atom panel row must contain eight slots")
        if self.error_code is None:
            if any(type(item) is not AtomScoreInterval for item in self.intervals):
                raise PositiveAtomSlateError("successful atom panel row lacks intervals")
        elif (
            type(self.error_code) is not str
            or _ERROR_CODE.fullmatch(self.error_code) is None
            or any(item is not None for item in self.intervals)
        ):
            raise PositiveAtomSlateError("failed atom panel row differs")

    @classmethod
    def from_payload(
        cls,
        panel_ordinal: int,
        slate: AffirmativeAtomSlate,
        payload: object,
    ) -> "AtomPanelScoreRow":
        if type(slate) is not AffirmativeAtomSlate:
            raise TypeError("atom panel payload needs exact slate")
        expected = {
            f"{atom_id}_{bound}" for atom_id in ATOM_IDS for bound in ("lower", "upper")
        }
        raw = _fields(payload, expected, "atom panel score payload")
        intervals = tuple(
            AtomScoreInterval(raw[f"{atom_id}_lower"], raw[f"{atom_id}_upper"])
            for atom_id in ATOM_IDS
        )
        return cls(panel_ordinal, slate.slate_digest, intervals)

    @classmethod
    def error(
        cls,
        panel_ordinal: int,
        slate_digest: str,
        error_code: str,
    ) -> "AtomPanelScoreRow":
        return cls(panel_ordinal, slate_digest, (None,) * ATOM_COUNT, error_code)

    @property
    def dispositions(self) -> tuple[Disposition, ...]:
        if self.error_code is not None:
            return (Disposition.ERROR,) * ATOM_COUNT
        return tuple(item.disposition for item in self.intervals if item is not None)

    @property
    def row_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": ATOM_PANEL_ROW_SCHEMA,
            "panel_ordinal": self.panel_ordinal,
            "slate_digest": self.slate_digest,
            "intervals": [None if item is None else item.to_data() for item in self.intervals],
            "dispositions": [item.value for item in self.dispositions],
            "error_code": self.error_code,
            "failed_fit_counts_as_absence": False,
        }

    @classmethod
    def from_data(cls, value: object) -> "AtomPanelScoreRow":
        raw = _fields(
            value,
            {
                "schema", "panel_ordinal", "slate_digest", "intervals",
                "dispositions", "error_code", "failed_fit_counts_as_absence",
            },
            "atom panel score row",
        )
        if (
            raw["schema"] != ATOM_PANEL_ROW_SCHEMA
            or type(raw["intervals"]) is not list
            or len(raw["intervals"]) != ATOM_COUNT
            or raw["failed_fit_counts_as_absence"] is not False
        ):
            raise PositiveAtomSlateError("atom panel score row policy differs")
        intervals = tuple(
            None if item is None else AtomScoreInterval.from_data(item)
            for item in raw["intervals"]
        )
        result = cls(raw["panel_ordinal"], raw["slate_digest"], intervals, raw["error_code"])
        if raw["dispositions"] != [item.value for item in result.dispositions]:
            raise PositiveAtomSlateError("atom panel row dispositions differ")
        if result.to_data() != dict(raw):
            raise PositiveAtomSlateError("atom panel score row is not canonical")
        return result


def _panel_request_content(value: "AtomPanelScoreRequest") -> dict[str, object]:
    return {
        "schema": ATOM_PANEL_REQUEST_SCHEMA,
        "protocol_id": ATOM_SLATE_PROTOCOL_ID,
        "runtime": value.runtime.to_data(),
        "panel_ordinal": value.panel_ordinal,
        "panel_identity": value.panel_identity.to_data(),
        "slate": value.slate.to_data(),
        "slate_digest": value.slate.slate_digest,
        "source_proposer_artifact_digest": value.source_proposer_artifact_digest,
        "source_proposer_request_digest": value.source_proposer_request_digest,
        "source_proposer_benchmark_sealable": value.source_proposer_benchmark_sealable,
        "model_visible_image_names": [PANEL_IMAGE_NAME],
        "model_visible_panel_ordinal": False,
        "model_returns_all_atom_intervals_in_one_batch": True,
        "model_selects_or_combines_atoms": False,
        "negative_class_description_field_present": False,
        "logical_negation_operator_present": False,
        "query_image_count": 0,
    }


@dataclass(frozen=True, slots=True)
class AtomPanelScoreRequest:
    runtime: TypedCodexRuntimeBinding
    panel_ordinal: int
    panel_identity: PrototypeImageIdentity
    slate: AffirmativeAtomSlate
    source_proposer_artifact_digest: str
    source_proposer_request_digest: str
    source_proposer_benchmark_sealable: bool
    request_digest: str

    def __post_init__(self) -> None:
        if type(self.runtime) is not TypedCodexRuntimeBinding:
            raise TypeError("atom panel request needs typed runtime")
        if type(self.panel_ordinal) is not int or not 0 <= self.panel_ordinal < SUPPORT_PANEL_COUNT:
            raise PositiveAtomSlateError("atom panel request ordinal differs")
        if (
            type(self.panel_identity) is not PrototypeImageIdentity
            or self.panel_identity.name != PANEL_IMAGE_NAME
            or type(self.slate) is not AffirmativeAtomSlate
            or type(self.source_proposer_benchmark_sealable) is not bool
        ):
            raise PositiveAtomSlateError("atom panel request lineage differs")
        _raw_digest(self.source_proposer_artifact_digest, "source proposer artifact digest")
        _raw_digest(self.source_proposer_request_digest, "source proposer request digest")
        _raw_digest(self.request_digest, "atom panel request digest")
        if self.request_digest != canonical_digest(_panel_request_content(self)):
            raise PositiveAtomSlateError("atom panel request digest differs")

    @classmethod
    def build_from_proposer(
        cls,
        panel_png: bytes,
        panel_ordinal: int,
        proposer_artifact: AtomSlateProposerArtifact,
        *,
        expected_proposer_artifact_digest: str,
    ) -> "AtomPanelScoreRequest":
        panel = _exact_png(panel_png, "atom support panel PNG")
        if type(panel_ordinal) is not int or not 0 <= panel_ordinal < SUPPORT_PANEL_COUNT:
            raise PositiveAtomSlateError("atom support panel ordinal differs")
        proposer = _restore_slate_proposer(
            proposer_artifact,
            expected_artifact_digest=expected_proposer_artifact_digest,
            proposer_journal_terminal=None,
            verify_terminal=False,
        )
        identity = PrototypeImageIdentity(
            PANEL_IMAGE_NAME, len(panel), hashlib.sha256(panel).hexdigest()
        )
        exposed = proposer.request.presentation[panel_ordinal]
        if (
            identity.byte_count != exposed.byte_count
            or identity.content_digest != exposed.content_digest
        ):
            raise PositiveAtomSlateError(
                "atom panel is not the exact exposed support at its ordinal"
            )
        provisional = object.__new__(cls)
        values = {
            "runtime": proposer.runtime,
            "panel_ordinal": panel_ordinal,
            "panel_identity": identity,
            "slate": proposer.slate,
            "source_proposer_artifact_digest": proposer.artifact_digest,
            "source_proposer_request_digest": proposer.request_digest,
            "source_proposer_benchmark_sealable": proposer.benchmark_sealable,
        }
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, request_digest=canonical_digest(_panel_request_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_panel_request_content(self), "request_digest": self.request_digest}

    @classmethod
    def from_data(cls, value: object) -> "AtomPanelScoreRequest":
        expected = set(_panel_request_content_fields()) | {"request_digest"}
        raw = _fields(value, expected, "atom panel score request")
        if (
            raw["schema"] != ATOM_PANEL_REQUEST_SCHEMA
            or raw["protocol_id"] != ATOM_SLATE_PROTOCOL_ID
            or raw["slate_digest"] != canonical_digest(raw["slate"])
            or raw["model_visible_image_names"] != [PANEL_IMAGE_NAME]
            or raw["model_visible_panel_ordinal"] is not False
            or raw["model_returns_all_atom_intervals_in_one_batch"] is not True
            or raw["model_selects_or_combines_atoms"] is not False
            or raw["negative_class_description_field_present"] is not False
            or raw["logical_negation_operator_present"] is not False
            or raw["query_image_count"] != 0
        ):
            raise PositiveAtomSlateError("atom panel request policy differs")
        result = cls(
            TypedCodexRuntimeBinding.from_data(raw["runtime"]), raw["panel_ordinal"],
            PrototypeImageIdentity.from_data(raw["panel_identity"]),
            AffirmativeAtomSlate.from_data(raw["slate"]),
            raw["source_proposer_artifact_digest"],
            raw["source_proposer_request_digest"],
            raw["source_proposer_benchmark_sealable"], raw["request_digest"],
        )
        if result.to_data() != dict(raw):
            raise PositiveAtomSlateError("atom panel request is not canonical")
        return result


def _panel_request_content_fields() -> tuple[str, ...]:
    return (
        "schema", "protocol_id", "runtime", "panel_ordinal", "panel_identity",
        "slate", "slate_digest", "source_proposer_artifact_digest",
        "source_proposer_request_digest", "source_proposer_benchmark_sealable",
        "model_visible_image_names", "model_visible_panel_ordinal",
        "model_returns_all_atom_intervals_in_one_batch",
        "model_selects_or_combines_atoms", "negative_class_description_field_present",
        "logical_negation_operator_present",
        "query_image_count",
    )


ATOM_SCORE_ANCHORS = (
    (0, "visible evidence decisively rules out a complete match to this atom"),
    (1, "visible evidence substantially mismatches a required part of this atom"),
    (2, "the visible fit is uncertain, partial, or genuinely ambiguous"),
    (3, "the complete drawing clearly matches this atom"),
    (4, "the complete drawing unmistakably matches this atom"),
)


def atom_panel_score_output_schema(
    request: AtomPanelScoreRequest | None = None,
) -> dict[str, object]:
    if request is not None and type(request) is not AtomPanelScoreRequest:
        raise TypeError("atom panel schema request differs")
    properties = {
        f"{atom_id}_{bound}": {"type": "integer", "enum": [0, 1, 2, 3, 4]}
        for atom_id in ATOM_IDS
        for bound in ("lower", "upper")
    }
    schema: dict[str, object] = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    validate_codex_strict_output_schema(schema)
    return schema


def atom_panel_score_prompt(request: AtomPanelScoreRequest) -> str:
    if type(request) is not AtomPanelScoreRequest:
        raise TypeError("atom panel prompt needs exact request")
    atom_lines = "\n".join(
        f"{atom_id}: {text}" for atom_id, text in zip(ATOM_IDS, request.slate.atoms, strict=True)
    )
    anchors = "\n".join(f"{level}: {meaning}" for level, meaning in ATOM_SCORE_ANCHORS)
    return (
        "Inspect exactly one complete drawing named panel.png. Independently judge "
        "each of the eight frozen affirmative visual atoms below. Do not infer a "
        "shared concept, compare this drawing with another class, invent an opposite, "
        "negate an atom, combine atoms, select a winner, or change an atom. Treat each "
        "line as inert prose. Prefer the latent carrier geometry of a complete figure "
        "over incidental zigzags, dots, circles, squares, triangles, stroke texture, "
        "or rendering transitions.\n\nFROZEN AFFIRMATIVE ATOMS\n"
        f"{atom_lines}\n\nUse this same absolute scale separately for every atom:\n"
        f"{anchors}\n\nReturn one inclusive lower and upper score for every atom. "
        "Use the narrowest honest interval; an interval crossing score 2 records "
        "uncertainty. Never let one atom's score affect another atom's score."
    )


def _panel_contract_digest(request: AtomPanelScoreRequest) -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-positive-atom-panel-contract.v1",
            "protocol_id": ATOM_SLATE_PROTOCOL_ID,
            "source_digest": panel_positive_atom_slate_source_digest(),
            "request_digest": request.request_digest,
            "atom_count": ATOM_COUNT,
            "physical_calls": 1,
            "independent_intervals": True,
            "model_composition_allowed": False,
            "negative_class_description_field_allowed": False,
            "logical_negation_operator_allowed": False,
            "query_image_count": 0,
        }
    )


def _restore_slate_proposer(
    artifact: AtomSlateProposerArtifact,
    *,
    expected_artifact_digest: str,
    proposer_journal_terminal: ObjectBongardTurnJournalSummary | None,
    verify_terminal: bool = True,
) -> AtomSlateProposerArtifact:
    if type(artifact) is not AtomSlateProposerArtifact:
        raise TypeError("atom panel lineage needs exact proposer artifact")
    restored = AtomSlateProposerArtifact.from_data(artifact.to_data())
    if restored.artifact_digest != _raw_digest(
        expected_artifact_digest, "expected atom proposer artifact digest"
    ):
        raise PositiveAtomSlateError("atom panel source proposer differs")
    if verify_terminal:
        _verify_external_terminal(restored.transport_custody, proposer_journal_terminal)
    return restored


def _verify_panel_request_lineage(
    request: AtomPanelScoreRequest,
    proposer: AtomSlateProposerArtifact,
) -> None:
    exposed = proposer.request.presentation[request.panel_ordinal]
    if (
        request.runtime != proposer.runtime
        or request.slate != proposer.slate
        or request.source_proposer_artifact_digest != proposer.artifact_digest
        or request.source_proposer_request_digest != proposer.request_digest
        or request.source_proposer_benchmark_sealable is not proposer.benchmark_sealable
        or request.panel_identity.byte_count != exposed.byte_count
        or request.panel_identity.content_digest != exposed.content_digest
    ):
        raise PositiveAtomSlateError("atom panel request differs from proposer lineage")


class AtomPanelStatus(str, Enum):
    SUCCESS = "success"
    PARSER_ERROR = "parser_error"
    TRANSPORT_ERROR = "transport_error"


def _panel_artifact_content(value: "AtomPanelScoreArtifact") -> dict[str, object]:
    return {
        "schema": ATOM_PANEL_ARTIFACT_SCHEMA,
        "protocol_id": ATOM_SLATE_PROTOCOL_ID,
        "request": value.request.to_data(),
        "request_digest": value.request.request_digest,
        "source_digest": value.source_digest,
        "contract_digest": value.contract_digest,
        "transport_custody": value.transport_custody.to_data(),
        "prompt_digest": value.prompt_digest,
        "output_schema_digest": value.output_schema_digest,
        "presentation": [item.to_data() for item in value.presentation],
        "physical_call_count": 1,
        "status": value.status.value,
        "model_payload": None if value.model_payload is None else dict(value.model_payload),
        "receipt": None if value.receipt is None else value.receipt.to_dict(),
        "row": value.row.to_data(),
        "failure_code": value.failure_code,
        "failure_type": value.failure_type,
        "benchmark_sealable": value.benchmark_sealable,
        "model_visible_image_names": [PANEL_IMAGE_NAME],
        "all_atoms_scored_in_one_batch": True,
        "model_selected_formula": False,
        "query_image_count": 0,
    }


@dataclass(frozen=True, slots=True)
class AtomPanelScoreArtifact:
    request: AtomPanelScoreRequest
    source_digest: str
    contract_digest: str
    transport_custody: AtomTransportCustody
    prompt_digest: str
    output_schema_digest: str
    presentation: tuple[PrototypeImageIdentity, ...]
    status: AtomPanelStatus
    model_payload: Mapping[str, Any] | None
    receipt: CodexReceipt | None
    row: AtomPanelScoreRow
    failure_code: str | None
    failure_type: str | None
    artifact_digest: str

    @property
    def benchmark_sealable(self) -> bool:
        return (
            self.status is AtomPanelStatus.SUCCESS
            and self.transport_custody.benchmark_sealable
            and self.request.source_proposer_benchmark_sealable
        )

    def __post_init__(self) -> None:
        if type(self.request) is not AtomPanelScoreRequest:
            raise TypeError("atom panel artifact needs exact request")
        if type(self.transport_custody) is not AtomTransportCustody:
            raise TypeError("atom panel artifact needs exact custody")
        if type(self.row) is not AtomPanelScoreRow or not isinstance(self.status, AtomPanelStatus):
            raise PositiveAtomSlateError("atom panel artifact outcome differs")
        for label, value in (
            ("source digest", self.source_digest),
            ("contract digest", self.contract_digest),
            ("prompt digest", self.prompt_digest),
            ("output schema digest", self.output_schema_digest),
            ("artifact digest", self.artifact_digest),
        ):
            _raw_digest(value, label)
        prompt = atom_panel_score_prompt(self.request)
        schema = atom_panel_score_output_schema(self.request)
        if (
            self.source_digest != panel_positive_atom_slate_source_digest()
            or self.contract_digest != _panel_contract_digest(self.request)
            or self.prompt_digest != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            or self.output_schema_digest != canonical_digest(schema)
            or type(self.presentation) is not tuple
            or len(self.presentation) != 1
            or self.presentation[0] != self.request.panel_identity
            or self.row.panel_ordinal != self.request.panel_ordinal
            or self.row.slate_digest != self.request.slate.slate_digest
        ):
            raise PositiveAtomSlateError("atom panel artifact binding differs")
        if self.model_payload is not None:
            object.__setattr__(
                self, "model_payload", _canonical_payload(self.model_payload, "atom panel payload")
            )
        if self.status is AtomPanelStatus.SUCCESS:
            if (
                self.model_payload is None
                or self.receipt is None
                or self.failure_code is not None
                or self.failure_type is not None
                or self.row != AtomPanelScoreRow.from_payload(
                    self.request.panel_ordinal, self.request.slate, self.model_payload
                )
            ):
                raise PositiveAtomSlateError("successful atom panel artifact differs")
        elif self.status is AtomPanelStatus.PARSER_ERROR:
            if (
                self.model_payload is None
                or self.receipt is None
                or self.row.error_code != self.failure_code
                or type(self.failure_code) is not str
                or type(self.failure_type) is not str
                or _ERROR_CODE.fullmatch(self.failure_code) is None
                or _ERROR_CODE.fullmatch(self.failure_type) is None
            ):
                raise PositiveAtomSlateError("atom parser failure artifact differs")
        elif self.status is AtomPanelStatus.TRANSPORT_ERROR:
            if (
                self.model_payload is not None
                or self.receipt is not None
                or self.row.error_code != self.failure_code
                or type(self.failure_code) is not str
                or type(self.failure_type) is not str
                or _ERROR_CODE.fullmatch(self.failure_code) is None
                or _ERROR_CODE.fullmatch(self.failure_type) is None
            ):
                raise PositiveAtomSlateError("atom transport failure artifact differs")
        else:
            raise PositiveAtomSlateError("atom panel status is unsupported")
        if self.receipt is not None:
            assert self.model_payload is not None
            _validate_receipt_binding(
                self.receipt,
                runtime=self.request.runtime,
                prompt_digest=self.prompt_digest,
                output_schema_digest=self.output_schema_digest,
                payload_digest=canonical_digest(dict(self.model_payload)),
                presentation=self.presentation,
            )
        if self.artifact_digest != canonical_digest(_panel_artifact_content(self)):
            raise PositiveAtomSlateError("atom panel artifact digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_panel_artifact_content(self), "artifact_digest": self.artifact_digest}

    @classmethod
    def from_data(cls, value: object) -> "AtomPanelScoreArtifact":
        raw = _fields(
            value,
            {
                "schema", "protocol_id", "request", "request_digest", "source_digest",
                "contract_digest", "transport_custody", "prompt_digest",
                "output_schema_digest", "presentation", "physical_call_count", "status",
                "model_payload", "receipt", "row", "failure_code", "failure_type",
                "benchmark_sealable", "model_visible_image_names",
                "all_atoms_scored_in_one_batch", "model_selected_formula",
                "query_image_count", "artifact_digest",
            },
            "atom panel score artifact",
        )
        if (
            raw["schema"] != ATOM_PANEL_ARTIFACT_SCHEMA
            or raw["protocol_id"] != ATOM_SLATE_PROTOCOL_ID
            or raw["physical_call_count"] != 1
            or raw["model_visible_image_names"] != [PANEL_IMAGE_NAME]
            or raw["all_atoms_scored_in_one_batch"] is not True
            or raw["model_selected_formula"] is not False
            or raw["query_image_count"] != 0
            or type(raw["presentation"]) is not list
        ):
            raise PositiveAtomSlateError("atom panel artifact policy differs")
        try:
            status = AtomPanelStatus(raw["status"])
        except Exception as exc:
            raise PositiveAtomSlateError("atom panel status is unknown") from exc
        result = cls(
            AtomPanelScoreRequest.from_data(raw["request"]), raw["source_digest"],
            raw["contract_digest"], AtomTransportCustody.from_data(raw["transport_custody"]),
            raw["prompt_digest"], raw["output_schema_digest"],
            tuple(PrototypeImageIdentity.from_data(item) for item in raw["presentation"]),
            status, raw["model_payload"],
            None if raw["receipt"] is None else _receipt_from_data(raw["receipt"]),
            AtomPanelScoreRow.from_data(raw["row"]), raw["failure_code"],
            raw["failure_type"], raw["artifact_digest"],
        )
        if (
            raw["request_digest"] != result.request.request_digest
            or raw["benchmark_sealable"] is not result.benchmark_sealable
            or result.to_data() != dict(raw)
        ):
            raise PositiveAtomSlateError("atom panel artifact is not canonical")
        return result


def _seal_panel_artifact(
    *,
    request: AtomPanelScoreRequest,
    custody: AtomTransportCustody,
    presentation: tuple[PrototypeImageIdentity, ...],
    status: AtomPanelStatus,
    payload: Mapping[str, Any] | None,
    receipt: CodexReceipt | None,
    row: AtomPanelScoreRow,
    failure_code: str | None,
    failure_type: str | None,
) -> AtomPanelScoreArtifact:
    values = {
        "request": request,
        "source_digest": panel_positive_atom_slate_source_digest(),
        "contract_digest": _panel_contract_digest(request),
        "transport_custody": custody,
        "prompt_digest": hashlib.sha256(atom_panel_score_prompt(request).encode("utf-8")).hexdigest(),
        "output_schema_digest": canonical_digest(atom_panel_score_output_schema(request)),
        "presentation": presentation,
        "status": status,
        "model_payload": None if payload is None else _canonical_payload(payload, "atom panel payload"),
        "receipt": receipt,
        "row": row,
        "failure_code": failure_code,
        "failure_type": failure_type,
    }
    provisional = object.__new__(AtomPanelScoreArtifact)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return AtomPanelScoreArtifact(
        **values, artifact_digest=canonical_digest(_panel_artifact_content(provisional))
    )


def observe_affirmative_atom_panel(
    panel_png: bytes,
    *,
    request: AtomPanelScoreRequest,
    source_proposer_artifact: AtomSlateProposerArtifact,
    expected_source_proposer_artifact_digest: str,
    proposer_journal_terminal: ObjectBongardTurnJournalSummary | None = None,
    minutes: int = 15,
    verbose: bool = False,
    executable: str = "codex",
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None,
    expected_launcher_digest: str,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
    transport=run_codex_named_images_structured,
) -> AtomPanelScoreArtifact:
    """Score all eight frozen atoms on one exposed support panel in one call."""

    if type(request) is not AtomPanelScoreRequest:
        raise TypeError("atom panel observer needs exact request")
    proposer = _restore_slate_proposer(
        source_proposer_artifact,
        expected_artifact_digest=expected_source_proposer_artifact_digest,
        proposer_journal_terminal=proposer_journal_terminal,
    )
    _verify_panel_request_lineage(request, proposer)
    panel = _exact_png(panel_png, "atom support panel PNG")
    runtime = _bind_runtime(
        model=request.runtime.model,
        reasoning_effort=request.runtime.reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
        model_catalog_snapshot=model_catalog_snapshot,
        no_tools_attestation=no_tools_attestation,
    )
    if (
        runtime != request.runtime
        or len(panel) != request.panel_identity.byte_count
        or hashlib.sha256(panel).hexdigest() != request.panel_identity.content_digest
        or type(minutes) is not int
        or minutes <= 0
    ):
        raise PositiveAtomSlateError("atom panel request belongs to other pixels or runtime")
    prompt = atom_panel_score_prompt(request)
    schema = atom_panel_score_output_schema(request)
    presentation_bytes = ((PANEL_IMAGE_NAME, panel),)
    presentation = (
        PrototypeImageIdentity(PANEL_IMAGE_NAME, len(panel), hashlib.sha256(panel).hexdigest()),
    )
    try:
        payload, receipt = _scene_runtime._stage_and_call(
            presentation_bytes,
            prompt=prompt,
            schema=schema,
            model=runtime.model,
            reasoning_effort=runtime.reasoning_effort,
            minutes=minutes,
            verbose=verbose,
            executable=executable,
            cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
            expected_launcher_digest=expected_launcher_digest,
            model_catalog_snapshot=model_catalog_snapshot,
            no_tools_attestation=no_tools_attestation,
            transport=transport,
        )
    except Exception as exc:
        failure_type = _scene_runtime._exception_type(exc)
        try:
            custody = _transport_custody(transport)
        except Exception:
            custody = AtomTransportCustody.create(
                "production_direct" if transport is run_codex_named_images_structured
                else "injected_unverified"
            )
        return _seal_panel_artifact(
            request=request, custody=custody, presentation=presentation,
            status=AtomPanelStatus.TRANSPORT_ERROR, payload=None, receipt=None,
            row=AtomPanelScoreRow.error(
                request.panel_ordinal, request.slate.slate_digest, "atom_transport_failed"
            ),
            failure_code="atom_transport_failed", failure_type=failure_type,
        )
    custody = _transport_custody(transport)
    frozen = _canonical_payload(payload, "atom panel payload")
    try:
        row = AtomPanelScoreRow.from_payload(request.panel_ordinal, request.slate, frozen)
    except Exception as exc:
        failure_type = _scene_runtime._exception_type(exc)
        return _seal_panel_artifact(
            request=request, custody=custody, presentation=presentation,
            status=AtomPanelStatus.PARSER_ERROR, payload=frozen, receipt=receipt,
            row=AtomPanelScoreRow.error(
                request.panel_ordinal, request.slate.slate_digest, "atom_payload_rejected"
            ),
            failure_code="atom_payload_rejected", failure_type=failure_type,
        )
    return _seal_panel_artifact(
        request=request, custody=custody, presentation=presentation,
        status=AtomPanelStatus.SUCCESS, payload=frozen, receipt=receipt, row=row,
        failure_code=None, failure_type=None,
    )


def verify_atom_panel_score_artifact(
    artifact: AtomPanelScoreArtifact,
    panel_png: bytes,
    *,
    expected_artifact_digest: str,
    source_proposer_artifact: AtomSlateProposerArtifact,
    expected_source_proposer_artifact_digest: str,
    proposer_journal_terminal: ObjectBongardTurnJournalSummary | None = None,
    panel_journal_terminal: ObjectBongardTurnJournalSummary | None = None,
    expected_request_digest: str | None = None,
) -> AtomPanelScoreArtifact:
    """Cold replay panel bytes, frozen slate lineage, terminal pins, and receipt."""

    if type(artifact) is not AtomPanelScoreArtifact:
        raise TypeError("atom panel replay needs exact artifact")
    restored = AtomPanelScoreArtifact.from_data(artifact.to_data())
    if restored.artifact_digest != _raw_digest(expected_artifact_digest, "expected atom panel artifact"):
        raise PositiveAtomSlateError("atom panel artifact differs from commitment")
    if expected_request_digest is not None and restored.request.request_digest != _raw_digest(
        expected_request_digest, "expected atom panel request"
    ):
        raise PositiveAtomSlateError("atom panel request differs from commitment")
    _verify_external_terminal(restored.transport_custody, panel_journal_terminal)
    proposer = _restore_slate_proposer(
        source_proposer_artifact,
        expected_artifact_digest=expected_source_proposer_artifact_digest,
        proposer_journal_terminal=proposer_journal_terminal,
    )
    _verify_panel_request_lineage(restored.request, proposer)
    panel = _exact_png(panel_png, "atom replay panel PNG")
    if (
        len(panel) != restored.request.panel_identity.byte_count
        or hashlib.sha256(panel).hexdigest() != restored.request.panel_identity.content_digest
    ):
        raise PositiveAtomSlateError("atom panel replay pixels differ")
    if restored.receipt is not None:
        assert restored.model_payload is not None
        with tempfile.TemporaryDirectory(prefix="bongard-atom-panel-replay-") as raw:
            target = Path(raw) / PANEL_IMAGE_NAME
            target.write_bytes(panel)
            try:
                validate_codex_named_image_receipt(
                    restored.receipt, atom_panel_score_prompt(restored.request),
                    (str(target.resolve()),), (PANEL_IMAGE_NAME,),
                    atom_panel_score_output_schema(restored.request),
                    dict(restored.model_payload),
                )
            except Exception as exc:
                raise PositiveAtomSlateError("atom panel receipt replay failed") from exc
    return restored


@dataclass(frozen=True, order=True, slots=True)
class AtomFormula:
    atom_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            type(self.atom_indices) is not tuple
            or len(self.atom_indices) not in {1, 2}
            or any(type(item) is not int or not 0 <= item < ATOM_COUNT for item in self.atom_indices)
            or tuple(sorted(set(self.atom_indices))) != self.atom_indices
        ):
            raise PositiveAtomSlateError("atom formula must be a sorted singleton or pair")

    @property
    def atom_ids(self) -> tuple[str, ...]:
        return tuple(ATOM_IDS[index] for index in self.atom_indices)

    @property
    def formula_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": ATOM_FORMULA_SCHEMA,
            "atom_indices": list(self.atom_indices),
            "atom_ids": list(self.atom_ids),
            "arity": len(self.atom_indices),
            "operator": "identity" if len(self.atom_indices) == 1 else "affirmative_conjunction",
            "negation_present": False,
            "polarity_choice_present": False,
        }

    @classmethod
    def from_data(cls, value: object) -> "AtomFormula":
        raw = _fields(
            value,
            {
                "schema", "atom_indices", "atom_ids", "arity", "operator",
                "negation_present", "polarity_choice_present",
            },
            "atom formula",
        )
        if raw["schema"] != ATOM_FORMULA_SCHEMA or type(raw["atom_indices"]) is not list:
            raise PositiveAtomSlateError("atom formula policy differs")
        result = cls(tuple(raw["atom_indices"]))
        if result.to_data() != dict(raw):
            raise PositiveAtomSlateError("atom formula is not canonical")
        return result


def enumerate_affirmative_atom_formulas() -> tuple[AtomFormula, ...]:
    """Return the preregistered eight singletons then 28 lexicographic pairs."""

    return tuple(
        [AtomFormula((index,)) for index in range(ATOM_COUNT)]
        + [
            AtomFormula((first, second))
            for first in range(ATOM_COUNT)
            for second in range(first + 1, ATOM_COUNT)
        ]
    )


def _conjunction_disposition(
    row: AtomPanelScoreRow,
    formula: AtomFormula,
) -> Disposition:
    parts = tuple(row.dispositions[index] for index in formula.atom_indices)
    if Disposition.ERROR in parts:
        return Disposition.ERROR
    if Disposition.CERTIFIED_ABSENT in parts:
        return Disposition.CERTIFIED_ABSENT
    if all(item is Disposition.PRESENT for item in parts):
        return Disposition.PRESENT
    return Disposition.INDETERMINATE


@dataclass(frozen=True, slots=True)
class AtomFormulaSupportProfile:
    formula: AtomFormula
    dispositions: tuple[Disposition, ...]

    def __post_init__(self) -> None:
        if (
            type(self.formula) is not AtomFormula
            or type(self.dispositions) is not tuple
            or len(self.dispositions) != SUPPORT_PANEL_COUNT
            or any(not isinstance(item, Disposition) for item in self.dispositions)
        ):
            raise PositiveAtomSlateError("atom formula support profile differs")

    def _count(self, start: int, disposition: Disposition) -> int:
        return sum(item is disposition for item in self.dispositions[start:start + GROUP_SIZE])

    @property
    def native_present(self) -> int:
        return self._count(0, Disposition.PRESENT)

    @property
    def native_absent(self) -> int:
        return self._count(0, Disposition.CERTIFIED_ABSENT)

    @property
    def native_indeterminate(self) -> int:
        return self._count(0, Disposition.INDETERMINATE)

    @property
    def native_error(self) -> int:
        return self._count(0, Disposition.ERROR)

    @property
    def contrast_present(self) -> int:
        return self._count(GROUP_SIZE, Disposition.PRESENT)

    @property
    def contrast_absent(self) -> int:
        return self._count(GROUP_SIZE, Disposition.CERTIFIED_ABSENT)

    @property
    def contrast_indeterminate(self) -> int:
        return self._count(GROUP_SIZE, Disposition.INDETERMINATE)

    @property
    def contrast_error(self) -> int:
        return self._count(GROUP_SIZE, Disposition.ERROR)

    @property
    def admitted(self) -> bool:
        return (
            self.native_present >= MINIMUM_DECISIVE_PER_SIDE
            and self.native_absent == 0
            and self.native_error == 0
            and self.contrast_absent >= MINIMUM_DECISIVE_PER_SIDE
            and self.contrast_present == 0
            and self.contrast_error == 0
        )

    def to_data(self) -> dict[str, object]:
        return {
            "schema": ATOM_PROFILE_SCHEMA,
            "formula": self.formula.to_data(),
            "dispositions": [item.value for item in self.dispositions],
            "native_counts": {
                "present": self.native_present,
                "certified_absent": self.native_absent,
                "indeterminate": self.native_indeterminate,
                "error": self.native_error,
            },
            "contrast_counts": {
                "present": self.contrast_present,
                "certified_absent": self.contrast_absent,
                "indeterminate": self.contrast_indeterminate,
                "error": self.contrast_error,
            },
            "minimum_decisive_per_side": MINIMUM_DECISIVE_PER_SIDE,
            "zero_contradictions_required": True,
            "zero_errors_required": True,
            "admitted": self.admitted,
        }

    @classmethod
    def from_data(cls, value: object) -> "AtomFormulaSupportProfile":
        raw = _fields(
            value,
            {
                "schema", "formula", "dispositions", "native_counts",
                "contrast_counts", "minimum_decisive_per_side",
                "zero_contradictions_required", "zero_errors_required", "admitted",
            },
            "atom formula support profile",
        )
        if (
            raw["schema"] != ATOM_PROFILE_SCHEMA
            or type(raw["dispositions"]) is not list
            or raw["minimum_decisive_per_side"] != MINIMUM_DECISIVE_PER_SIDE
            or raw["zero_contradictions_required"] is not True
            or raw["zero_errors_required"] is not True
        ):
            raise PositiveAtomSlateError("atom support profile policy differs")
        try:
            dispositions = tuple(Disposition(item) for item in raw["dispositions"])
        except Exception as exc:
            raise PositiveAtomSlateError("atom support profile disposition is unknown") from exc
        result = cls(AtomFormula.from_data(raw["formula"]), dispositions)
        if result.to_data() != dict(raw):
            raise PositiveAtomSlateError("atom formula support profile is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class AtomSupportGap:
    error_row_ordinals: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            type(self.error_row_ordinals) is not tuple
            or tuple(sorted(set(self.error_row_ordinals))) != self.error_row_ordinals
            or any(type(item) is not int or not 0 <= item < SUPPORT_PANEL_COUNT for item in self.error_row_ordinals)
        ):
            raise PositiveAtomSlateError("atom support gap error rows differ")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": ATOM_GAP_SCHEMA,
            "code": "no_admissible_affirmative_singleton_or_pair",
            "formula_count": FORMULA_COUNT,
            "row_count": SUPPORT_PANEL_COUNT,
            "error_row_ordinals": list(self.error_row_ordinals),
            "query_release_allowed": False,
        }

    @classmethod
    def from_data(cls, value: object) -> "AtomSupportGap":
        raw = _fields(
            value,
            {
                "schema", "code", "formula_count", "row_count",
                "error_row_ordinals", "query_release_allowed",
            },
            "atom support gap",
        )
        if (
            raw["schema"] != ATOM_GAP_SCHEMA
            or raw["code"] != "no_admissible_affirmative_singleton_or_pair"
            or raw["formula_count"] != FORMULA_COUNT
            or raw["row_count"] != SUPPORT_PANEL_COUNT
            or type(raw["error_row_ordinals"]) is not list
            or raw["query_release_allowed"] is not False
        ):
            raise PositiveAtomSlateError("atom support gap policy differs")
        result = cls(tuple(raw["error_row_ordinals"]))
        if result.to_data() != dict(raw):
            raise PositiveAtomSlateError("atom support gap is not canonical")
        return result


def _profiles_for_rows(
    rows: tuple[AtomPanelScoreRow, ...],
) -> tuple[AtomFormulaSupportProfile, ...]:
    return tuple(
        AtomFormulaSupportProfile(
            formula, tuple(_conjunction_disposition(row, formula) for row in rows)
        )
        for formula in enumerate_affirmative_atom_formulas()
    )


def _freeze_complete_rows(
    slate: AffirmativeAtomSlate,
    rows: Sequence[AtomPanelScoreRow],
) -> tuple[AtomPanelScoreRow, ...]:
    if type(slate) is not AffirmativeAtomSlate:
        raise TypeError("atom support inventory needs exact slate")
    if isinstance(rows, (bytes, bytearray, str)):
        raise TypeError("atom support inventory rows differ")
    frozen = tuple(rows)
    if (
        len(frozen) != SUPPORT_PANEL_COUNT
        or any(type(row) is not AtomPanelScoreRow for row in frozen)
        or tuple(row.panel_ordinal for row in frozen) != tuple(range(SUPPORT_PANEL_COUNT))
        or any(row.slate_digest != slate.slate_digest for row in frozen)
    ):
        raise PositiveAtomSlateError(
            "all twelve ordered support rows must exist before formula enumeration"
        )
    return frozen


@dataclass(frozen=True, slots=True)
class AtomSupportInventory:
    slate: AffirmativeAtomSlate
    rows: tuple[AtomPanelScoreRow, ...]
    profiles: tuple[AtomFormulaSupportProfile, ...]
    admitted_formulas: tuple[AtomFormula, ...]
    gap: AtomSupportGap | None

    def __post_init__(self) -> None:
        if type(self.rows) is not tuple:
            raise TypeError("atom support inventory rows must be frozen")
        _freeze_complete_rows(self.slate, self.rows)
        expected_profiles = _profiles_for_rows(self.rows)
        expected_admitted = tuple(
            profile.formula for profile in expected_profiles if profile.admitted
        )
        expected_gap = None if expected_admitted else AtomSupportGap(
            tuple(row.panel_ordinal for row in self.rows if row.error_code is not None)
        )
        if (
            self.profiles != expected_profiles
            or self.admitted_formulas != expected_admitted
            or self.gap != expected_gap
        ):
            raise PositiveAtomSlateError("atom support inventory is not deterministic")

    @classmethod
    def create(
        cls,
        slate: AffirmativeAtomSlate,
        rows: Sequence[AtomPanelScoreRow],
    ) -> "AtomSupportInventory":
        frozen = _freeze_complete_rows(slate, rows)
        profiles = _profiles_for_rows(frozen)
        admitted = tuple(profile.formula for profile in profiles if profile.admitted)
        gap = None if admitted else AtomSupportGap(
            tuple(row.panel_ordinal for row in frozen if row.error_code is not None)
        )
        return cls(slate, frozen, profiles, admitted, gap)

    @property
    def inventory_digest(self) -> str:
        return canonical_digest(self.content_data())

    @property
    def query_release_allowed(self) -> bool:
        return bool(self.admitted_formulas)

    def content_data(self) -> dict[str, object]:
        return {
            "schema": ATOM_INVENTORY_SCHEMA,
            "protocol_id": ATOM_SLATE_PROTOCOL_ID,
            "slate": self.slate.to_data(),
            "slate_digest": self.slate.slate_digest,
            "rows": [row.to_data() for row in self.rows],
            "row_digests": [row.row_digest for row in self.rows],
            "profiles": [profile.to_data() for profile in self.profiles],
            "formula_count": FORMULA_COUNT,
            "formula_order": "eight_singletons_then_twenty_eight_lexicographic_pairs",
            "composition_started_after_all_panel_rows": True,
            "admitted_formulas": [formula.to_data() for formula in self.admitted_formulas],
            "gap": None if self.gap is None else self.gap.to_data(),
            "query_release_allowed": self.query_release_allowed,
            "query_image_count": 0,
            "negative_class_description_field_present": False,
            "logical_negation_or_polarity_operator_present": False,
            "model_selected_formula": False,
            "threshold_selected_after_observations": False,
            "lean_present": False,
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "inventory_digest": self.inventory_digest}

    @classmethod
    def from_data(cls, value: object) -> "AtomSupportInventory":
        raw = _fields(
            value,
            {
                "schema", "protocol_id", "slate", "slate_digest", "rows",
                "row_digests", "profiles", "formula_count", "formula_order",
                "composition_started_after_all_panel_rows", "admitted_formulas",
                "gap", "query_release_allowed", "query_image_count",
                "negative_class_description_field_present",
                "logical_negation_or_polarity_operator_present",
                "model_selected_formula", "threshold_selected_after_observations",
                "lean_present", "inventory_digest",
            },
            "atom support inventory",
        )
        if (
            raw["schema"] != ATOM_INVENTORY_SCHEMA
            or raw["protocol_id"] != ATOM_SLATE_PROTOCOL_ID
            or type(raw["rows"]) is not list
            or type(raw["profiles"]) is not list
            or type(raw["admitted_formulas"]) is not list
            or raw["formula_count"] != FORMULA_COUNT
            or raw["formula_order"] != "eight_singletons_then_twenty_eight_lexicographic_pairs"
            or raw["composition_started_after_all_panel_rows"] is not True
            or raw["query_image_count"] != 0
            or raw["negative_class_description_field_present"] is not False
            or raw["logical_negation_or_polarity_operator_present"] is not False
            or raw["model_selected_formula"] is not False
            or raw["threshold_selected_after_observations"] is not False
            or raw["lean_present"] is not False
        ):
            raise PositiveAtomSlateError("atom support inventory policy differs")
        result = cls(
            AffirmativeAtomSlate.from_data(raw["slate"]),
            tuple(AtomPanelScoreRow.from_data(item) for item in raw["rows"]),
            tuple(AtomFormulaSupportProfile.from_data(item) for item in raw["profiles"]),
            tuple(AtomFormula.from_data(item) for item in raw["admitted_formulas"]),
            None if raw["gap"] is None else AtomSupportGap.from_data(raw["gap"]),
        )
        if (
            raw["slate_digest"] != result.slate.slate_digest
            or raw["row_digests"] != [row.row_digest for row in result.rows]
            or raw["query_release_allowed"] is not result.query_release_allowed
            or raw["inventory_digest"] != result.inventory_digest
            or result.to_data() != dict(raw)
        ):
            raise PositiveAtomSlateError("atom support inventory is not canonical")
        return result


__all__ = (
    "ABSENT_UPPER_BOUND",
    "ATOM_COUNT",
    "ATOM_IDS",
    "ATOM_SCORE_ANCHORS",
    "FORMULA_COUNT",
    "GROUP_SIZE",
    "MINIMUM_DECISIVE_PER_SIDE",
    "PRESENT_LOWER_BOUND",
    "SUPPORT_PANEL_COUNT",
    "AffirmativeAtomSlate",
    "AtomFormula",
    "AtomFormulaSupportProfile",
    "AtomPanelScoreArtifact",
    "AtomPanelScoreRequest",
    "AtomPanelScoreRow",
    "AtomPanelStatus",
    "AtomScoreInterval",
    "AtomSlateProposerArtifact",
    "AtomSlateProposerRequest",
    "AtomSupportGap",
    "AtomSupportInventory",
    "AtomTransportCustody",
    "PositiveAtomSlateError",
    "atom_panel_score_output_schema",
    "atom_panel_score_prompt",
    "atom_slate_proposer_output_schema",
    "atom_slate_proposer_prompt",
    "enumerate_affirmative_atom_formulas",
    "observe_affirmative_atom_panel",
    "panel_positive_atom_slate_source_digest",
    "propose_affirmative_atom_slate",
    "verify_atom_panel_score_artifact",
    "verify_atom_slate_proposer_artifact",
)
