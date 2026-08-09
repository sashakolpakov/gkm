"""One receipted support-only call proposing one inert positive conjunction.

Exactly six ``group_a`` and six ``group_b`` PNGs are shown.  Group B is
explicitly allowed to be heterogeneous.  The model returns only a short cue
and two affirmative component strings; Python treats all three as inert prose.
No query pixels, executable predicate, threshold, polarity, or class metadata
are admitted by this adapter.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

from bongard import prototype_scene_observer as _scene_runtime
from bongard.canonical import canonical_digest, canonical_json
from bongard.codex_no_tools_preflight import CodexNoToolsAttestation
from bongard.object_bongard_turn_journal import (
    ObjectBongardNamedImageTurnJournalTransport,
    ObjectBongardTurnJournalSummary,
    object_bongard_turn_journal_source_digest,
)
from bongard.panel_typed_codex_observer import (
    PanelTypedCodexObserverError,
    TypedCodexRuntimeBinding,
    _bind_runtime,
    _canonical_payload,
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


SUPPORT_POSITIVE_PROPOSER_PROTOCOL_ID = (
    "bongard.panel-support-positive-proposer/twelve-support-one-cue-v1"
)
SUPPORT_POSITIVE_PROPOSER_REQUEST_SCHEMA = (
    "gkm.bongard-support-positive-proposer-request.v1"
)
POSITIVE_CONJUNCTION_RUBRIC_SCHEMA = (
    "gkm.bongard-positive-conjunction-rubric.v1"
)
SUPPORT_POSITIVE_PROPOSER_ARTIFACT_SCHEMA = (
    "gkm.bongard-support-positive-proposer-artifact.v1"
)
SUPPORT_POSITIVE_TRANSPORT_PROVENANCE_SCHEMA = (
    "gkm.bongard-support-positive-transport-provenance.v1"
)

SUPPORT_GROUP_SIZE = 6
SUPPORT_IMAGE_COUNT = 12
SUPPORT_POSITIVE_PRESENTATION_NAMES = tuple(
    [f"group_a_{index:02d}.png" for index in range(SUPPORT_GROUP_SIZE)]
    + [f"group_b_{index:02d}.png" for index in range(SUPPORT_GROUP_SIZE)]
)
MAX_CUE_UTF8_BYTES = 512
MAX_COMPONENT_UTF8_BYTES = 256
MAX_PROMPT_UTF8_BYTES = 32 * 1024
MAX_RESPONSE_UTF8_BYTES = 4 * 1024
SUPPORT_POSITIVE_ESTIMATES = ("supports", "does_not_support", "unclear")
SUPPORT_POSITIVE_GAP_SCHEMA = "gkm.bongard-support-positive-proposal-gap.v1"

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_FORBIDDEN_PROSE = re.compile(
    r"(?:```|`|[{};]|[<>]=?|[≤≥]|%|"
    r"\b(?:not|no|without|lack|lacks|lacking|absence|absent|negative|"
    r"complement|foil|opposite|fails|failure|exclude|excludes|"
    r"threshold|cutoff|score|probability|confidence|percent|"
    r"at\s+least|at\s+most|more\s+than|less\s+than|above|below|"
    r"def|lambda|import|exec|eval|return|function|python|javascript|sql|regex|"
    r"task|phase|side|class|support|query|group|panel|image|candidate|formula)\b)",
    re.IGNORECASE,
)


class SupportPositiveProposerError(PanelTypedCodexObserverError):
    """The support-only proposer request, payload, custody, or replay is invalid."""


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected
    ):
        raise SupportPositiveProposerError(f"{label} fields differ")
    return value


def panel_support_positive_proposer_source_digest() -> str:
    """Return the exact authenticated proposer source loaded by Python."""

    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _inert_prose(value: object, *, label: str, maximum: int) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise SupportPositiveProposerError(f"{label} must be nonempty trimmed prose")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError as exc:
        raise SupportPositiveProposerError(f"{label} must be UTF-8") from exc
    if len(encoded) > maximum or any(ord(char) < 32 for char in value):
        raise SupportPositiveProposerError(f"{label} exceeds its inert prose bound")
    if _FORBIDDEN_PROSE.search(value) is not None:
        raise SupportPositiveProposerError(
            f"{label} contains negation, policy, identifiers, threshold, or code"
        )
    return value


@dataclass(frozen=True, slots=True)
class PositiveConjunctionRubric:
    """Three bounded inert strings learned only from the support presentation."""

    cue_text: str
    component_1: str
    component_2: str

    def __post_init__(self) -> None:
        cue = _inert_prose(
            self.cue_text, label="positive cue", maximum=MAX_CUE_UTF8_BYTES
        )
        first = _inert_prose(
            self.component_1,
            label="positive component 1",
            maximum=MAX_COMPONENT_UTF8_BYTES,
        )
        second = _inert_prose(
            self.component_2,
            label="positive component 2",
            maximum=MAX_COMPONENT_UTF8_BYTES,
        )
        if cue != f"{first} and {second}":
            raise SupportPositiveProposerError(
                "positive cue must be the exact ordered join of its two components"
            )
        if first.casefold() == second.casefold():
            raise SupportPositiveProposerError(
                "positive conjunction components must be distinct"
            )

    @property
    def rubric_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": POSITIVE_CONJUNCTION_RUBRIC_SCHEMA,
            "cue_text": self.cue_text,
            "component_1": self.component_1,
            "component_2": self.component_2,
            "prose_is_inert": True,
            "threshold_selected": False,
            "polarity_selected": False,
            "executable_code_present": False,
        }

    @classmethod
    def from_data(cls, value: object) -> "PositiveConjunctionRubric":
        raw = _fields(
            value,
            {
                "schema",
                "cue_text",
                "component_1",
                "component_2",
                "prose_is_inert",
                "threshold_selected",
                "polarity_selected",
                "executable_code_present",
            },
            "positive conjunction rubric",
        )
        if (
            raw["schema"] != POSITIVE_CONJUNCTION_RUBRIC_SCHEMA
            or raw["prose_is_inert"] is not True
            or any(
                raw[key] is not False
                for key in (
                    "threshold_selected",
                    "polarity_selected",
                    "executable_code_present",
                )
            )
        ):
            raise SupportPositiveProposerError("positive rubric policy differs")
        result = cls(raw["cue_text"], raw["component_1"], raw["component_2"])
        if result.to_data() != dict(raw):
            raise SupportPositiveProposerError("positive rubric is not canonical")
        return result


def _estimate_field(name: str) -> str:
    return name.removesuffix(".png") + "_estimate"


@dataclass(frozen=True, slots=True)
class SupportPositiveProposalGap:
    """Typed non-proposal when the one cue fails fixed contrastive admission."""

    group_a_supports: int
    group_a_does_not_support: int
    group_a_unclear: int
    group_b_supports: int
    group_b_does_not_support: int
    group_b_unclear: int

    def __post_init__(self) -> None:
        values = (
            self.group_a_supports,
            self.group_a_does_not_support,
            self.group_a_unclear,
            self.group_b_supports,
            self.group_b_does_not_support,
            self.group_b_unclear,
        )
        if any(type(item) is not int or not 0 <= item <= 6 for item in values):
            raise SupportPositiveProposerError("positive proposal gap counts differ")
        if sum(values[:3]) != 6 or sum(values[3:]) != 6:
            raise SupportPositiveProposerError("positive proposal gap arity differs")
        if _admitted_counts(*values):
            raise SupportPositiveProposerError("admitted cue cannot be a proposal gap")

    @property
    def gap_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": SUPPORT_POSITIVE_GAP_SCHEMA,
            "code": "contrastive_admission_rejected",
            "group_a_supports": self.group_a_supports,
            "group_a_does_not_support": self.group_a_does_not_support,
            "group_a_unclear": self.group_a_unclear,
            "group_b_supports": self.group_b_supports,
            "group_b_does_not_support": self.group_b_does_not_support,
            "group_b_unclear": self.group_b_unclear,
            "rubric_admitted": False,
        }

    @classmethod
    def from_data(cls, value: object) -> "SupportPositiveProposalGap":
        raw = _fields(
            value,
            {
                "schema", "code", "group_a_supports",
                "group_a_does_not_support", "group_a_unclear",
                "group_b_supports", "group_b_does_not_support",
                "group_b_unclear", "rubric_admitted",
            },
            "support positive proposal gap",
        )
        if (
            raw["schema"] != SUPPORT_POSITIVE_GAP_SCHEMA
            or raw["code"] != "contrastive_admission_rejected"
            or raw["rubric_admitted"] is not False
        ):
            raise SupportPositiveProposerError("support positive gap policy differs")
        result = cls(
            raw["group_a_supports"], raw["group_a_does_not_support"],
            raw["group_a_unclear"], raw["group_b_supports"],
            raw["group_b_does_not_support"], raw["group_b_unclear"],
        )
        if result.to_data() != dict(raw):
            raise SupportPositiveProposerError("support positive gap is not canonical")
        return result


def _freeze_group(value: Sequence[bytes], label: str) -> tuple[bytes, ...]:
    if isinstance(value, (bytes, bytearray, str)) or len(value) != SUPPORT_GROUP_SIZE:
        raise SupportPositiveProposerError(f"{label} must contain exactly six PNGs")
    return tuple(
        _exact_png(item, f"{label} PNG {index}") for index, item in enumerate(value)
    )


@dataclass(frozen=True, slots=True)
class SupportPositiveProposerRequest:
    """Frozen exact 6+6 support presentation and runtime; no query identity exists."""

    runtime: TypedCodexRuntimeBinding
    presentation: tuple[PrototypeImageIdentity, ...]

    def __post_init__(self) -> None:
        if type(self.runtime) is not TypedCodexRuntimeBinding:
            raise TypeError("support positive request needs TypedCodexRuntimeBinding")
        if (
            type(self.presentation) is not tuple
            or len(self.presentation) != SUPPORT_IMAGE_COUNT
            or any(
                type(item) is not PrototypeImageIdentity for item in self.presentation
            )
            or tuple(item.name for item in self.presentation)
            != SUPPORT_POSITIVE_PRESENTATION_NAMES
        ):
            raise SupportPositiveProposerError(
                "support positive request must bind the exact 6+6 presentation"
            )

    @classmethod
    def build(
        cls,
        group_a_pngs: Sequence[bytes],
        group_b_pngs: Sequence[bytes],
        *,
        runtime: TypedCodexRuntimeBinding,
    ) -> "SupportPositiveProposerRequest":
        first = _freeze_group(group_a_pngs, "group A")
        second = _freeze_group(group_b_pngs, "group B")
        presentation = tuple(
            PrototypeImageIdentity(name, len(raw), hashlib.sha256(raw).hexdigest())
            for name, raw in zip(
                SUPPORT_POSITIVE_PRESENTATION_NAMES, (*first, *second), strict=True
            )
        )
        return cls(runtime, presentation)

    @property
    def request_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": SUPPORT_POSITIVE_PROPOSER_REQUEST_SCHEMA,
            "protocol_id": SUPPORT_POSITIVE_PROPOSER_PROTOCOL_ID,
            "runtime": self.runtime.to_data(),
            "presentation": [item.to_data() for item in self.presentation],
            "group_sizes": [SUPPORT_GROUP_SIZE, SUPPORT_GROUP_SIZE],
            "query_image_count": 0,
            "model_call_count": 1,
            "task_phase_side_class_ids_model_visible": False,
        }

    @classmethod
    def from_data(cls, value: object) -> "SupportPositiveProposerRequest":
        raw = _fields(
            value,
            {
                "schema",
                "protocol_id",
                "runtime",
                "presentation",
                "group_sizes",
                "query_image_count",
                "model_call_count",
                "task_phase_side_class_ids_model_visible",
            },
            "support positive request",
        )
        if (
            raw["schema"] != SUPPORT_POSITIVE_PROPOSER_REQUEST_SCHEMA
            or raw["protocol_id"] != SUPPORT_POSITIVE_PROPOSER_PROTOCOL_ID
            or type(raw["presentation"]) is not list
            or raw["group_sizes"] != [SUPPORT_GROUP_SIZE, SUPPORT_GROUP_SIZE]
            or raw["query_image_count"] != 0
            or raw["model_call_count"] != 1
            or raw["task_phase_side_class_ids_model_visible"] is not False
        ):
            raise SupportPositiveProposerError("support positive request policy differs")
        result = cls(
            TypedCodexRuntimeBinding.from_data(raw["runtime"]),
            tuple(PrototypeImageIdentity.from_data(item) for item in raw["presentation"]),
        )
        if result.to_data() != dict(raw):
            raise SupportPositiveProposerError("support positive request is not canonical")
        return result


def support_positive_proposer_prompt(request: SupportPositiveProposerRequest) -> str:
    """Return the fixed one-positive, non-foil support prompt."""

    if type(request) is not SupportPositiveProposerRequest:
        raise TypeError("support positive prompt needs SupportPositiveProposerRequest")
    group_a = ", ".join(SUPPORT_POSITIVE_PRESENTATION_NAMES[:SUPPORT_GROUP_SIZE])
    group_b = ", ".join(SUPPORT_POSITIVE_PRESENTATION_NAMES[SUPPORT_GROUP_SIZE:])
    prompt = (
        "Inspect exactly twelve complete drawings in two disclosed groups. "
        f"Group A contains {group_a}. Group B contains {group_b}. "
        "Group B may be heterogeneous: do not assume, infer, or describe one shared "
        "Group B rule. Propose exactly one short affirmative visual conjunction "
        "describing Group A. Use two visibly present affirmative components. The cue "
        "must be formed byte-for-byte as component_1, then the literal lowercase "
        "text ' and ', then component_2; do not paraphrase the components in "
        "cue_text. Return only cue_text, component_1, and "
        "component_2 plus all twelve fixed per-drawing estimate fields. For the whole "
        "conjunction, mark each drawing supports, does_not_support, or unclear. Check "
        "all twelve independently. Python admits a cue only when all six Group A "
        "drawings support it, at least five Group B drawings do_not_support it, and "
        "at most one Group B drawing is either a contradiction or unclear. Do not "
        "state a foil, complement, negative predicate, absence, "
        "negation, or a description of Group B. Do not select a threshold, scoring "
        "rule, polarity, classifier, decision procedure, or executable code. Exact "
        "visible counts may be described as visual properties, but never tune a "
        "cutoff. In returned strings do not mention groups, images, panels, task, "
        "phase, side, class, support/query roles, candidates, or formulas. Returned "
        "prose is an inert frozen visual cue, never executable semantics."
    )
    if len(prompt.encode("utf-8")) > MAX_PROMPT_UTF8_BYTES:
        raise SupportPositiveProposerError("support positive prompt exceeds capacity")
    return prompt


def support_positive_proposer_output_schema(
    request: SupportPositiveProposerRequest,
) -> dict[str, object]:
    """Return the strict fixed three-string response schema."""

    if type(request) is not SupportPositiveProposerRequest:
        raise TypeError("support positive schema needs SupportPositiveProposerRequest")
    properties = {
        "cue_text": {"type": "string"},
        "component_1": {"type": "string"},
        "component_2": {"type": "string"},
    }
    for name in SUPPORT_POSITIVE_PRESENTATION_NAMES:
        properties[_estimate_field(name)] = {
            "type": "string",
            "enum": list(SUPPORT_POSITIVE_ESTIMATES),
        }
    schema: dict[str, object] = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    validate_codex_strict_output_schema(schema)
    return schema


def _estimate_counts(
    payload: Mapping[str, Any], names: Sequence[str]
) -> tuple[int, int, int]:
    values = tuple(payload[_estimate_field(name)] for name in names)
    if any(value not in SUPPORT_POSITIVE_ESTIMATES for value in values):
        raise SupportPositiveProposerError("support positive estimate differs")
    return tuple(  # type: ignore[return-value]
        values.count(item) for item in SUPPORT_POSITIVE_ESTIMATES
    )


def _admitted_counts(
    group_a_supports: int,
    group_a_does_not_support: int,
    group_a_unclear: int,
    group_b_supports: int,
    group_b_does_not_support: int,
    group_b_unclear: int,
) -> bool:
    return (
        group_a_supports == 6
        and group_a_does_not_support == 0
        and group_a_unclear == 0
        and group_b_does_not_support >= 5
        and group_b_supports + group_b_unclear <= 1
    )


def _parse_outcome(
    payload: Mapping[str, Any],
) -> tuple[PositiveConjunctionRubric | None, SupportPositiveProposalGap | None]:
    expected = {
        "cue_text",
        "component_1",
        "component_2",
        *(_estimate_field(name) for name in SUPPORT_POSITIVE_PRESENTATION_NAMES),
    }
    raw = _fields(
        payload, expected, "positive payload"
    )
    candidate = PositiveConjunctionRubric(
        raw["cue_text"], raw["component_1"], raw["component_2"]
    )
    first = _estimate_counts(
        raw, SUPPORT_POSITIVE_PRESENTATION_NAMES[:SUPPORT_GROUP_SIZE]
    )
    second = _estimate_counts(
        raw, SUPPORT_POSITIVE_PRESENTATION_NAMES[SUPPORT_GROUP_SIZE:]
    )
    counts = (*first, *second)
    if _admitted_counts(*counts):
        return candidate, None
    return None, SupportPositiveProposalGap(*counts)


def _transport_source_binding(kind: str) -> str:
    transport_source = _scene_runtime.prototype_scene_transport_source_digest()
    if kind == "production_direct":
        value: dict[str, object] = {
            "schema": "gkm.bongard-support-positive-transport-source.v1",
            "kind": kind,
            "transport_source_digest": transport_source,
        }
    elif kind == "production_exactly_once_journal":
        value = {
            "schema": "gkm.bongard-support-positive-transport-source.v1",
            "kind": kind,
            "journal_source_digest": object_bongard_turn_journal_source_digest(),
            "underlying_transport_source_digest": transport_source,
        }
    elif kind == "injected_unverified":
        value = {
            "schema": "gkm.bongard-support-positive-transport-source.v1",
            "kind": kind,
            "callable_source_identity_verified": False,
        }
    else:
        raise SupportPositiveProposerError("support positive transport kind differs")
    return "sha256:" + canonical_digest(value)


@dataclass(frozen=True, slots=True)
class SupportPositiveTransportProvenance:
    kind: str
    source_binding: str
    production_transport_chain_verified: bool
    benchmark_sealable: bool
    journal_manifest_digest: str | None = None
    journal_turn_key: str | None = None
    journal_claim_digest: str | None = None
    journal_result_digest: str | None = None
    journal_outcome_digest: str | None = None
    journal_terminal_record_digest: str | None = None

    def __post_init__(self) -> None:
        kinds = {
            "production_direct",
            "production_exactly_once_journal",
            "injected_unverified",
        }
        if self.kind not in kinds or self.source_binding != _transport_source_binding(
            self.kind
        ):
            raise SupportPositiveProposerError("support positive transport differs")
        production = self.kind != "injected_unverified"
        journal = self.kind == "production_exactly_once_journal"
        if (
            self.production_transport_chain_verified is not production
            or self.benchmark_sealable is not journal
        ):
            raise SupportPositiveProposerError("support positive sealing policy differs")
        values = (
            self.journal_manifest_digest,
            self.journal_turn_key,
            self.journal_claim_digest,
            self.journal_result_digest,
            self.journal_outcome_digest,
            self.journal_terminal_record_digest,
        )
        if journal:
            if any(type(item) is not str or _ADDRESS.fullmatch(item) is None for item in values):
                raise SupportPositiveProposerError("journal terminal provenance differs")
        elif any(item is not None for item in values):
            raise SupportPositiveProposerError("non-journal transport names a journal")

    @classmethod
    def create(
        cls,
        kind: str,
        *,
        journal_summary: ObjectBongardTurnJournalSummary | None = None,
    ) -> "SupportPositiveTransportProvenance":
        if kind == "production_exactly_once_journal":
            if (
                type(journal_summary) is not ObjectBongardTurnJournalSummary
                or journal_summary.terminal_status != "success"
            ):
                raise SupportPositiveProposerError("journal is not a durable success")
            return cls(
                kind,
                _transport_source_binding(kind),
                True,
                True,
                journal_summary.manifest_digest,
                journal_summary.turn_key,
                journal_summary.claim_digest,
                journal_summary.result_digest,
                journal_summary.outcome_digest,
                journal_summary.record_digest,
            )
        if journal_summary is not None:
            raise SupportPositiveProposerError(
                "non-journal transport received journal terminal custody"
            )
        return cls(
            kind,
            _transport_source_binding(kind),
            kind == "production_direct",
            False,
        )

    def to_data(self) -> dict[str, object]:
        return {
            "schema": SUPPORT_POSITIVE_TRANSPORT_PROVENANCE_SCHEMA,
            "kind": self.kind,
            "source_binding": self.source_binding,
            "production_transport_chain_verified": self.production_transport_chain_verified,
            "benchmark_sealable": self.benchmark_sealable,
            "journal_manifest_digest": self.journal_manifest_digest,
            "journal_turn_key": self.journal_turn_key,
            "journal_claim_digest": self.journal_claim_digest,
            "journal_result_digest": self.journal_result_digest,
            "journal_outcome_digest": self.journal_outcome_digest,
            "journal_terminal_record_digest": self.journal_terminal_record_digest,
            "physical_model_call_cold_authenticated": False,
            "benchmark_requires_external_journal_terminal": True,
        }

    @classmethod
    def from_data(cls, value: object) -> "SupportPositiveTransportProvenance":
        raw = _fields(
            value,
            {
                "schema", "kind", "source_binding",
                "production_transport_chain_verified", "benchmark_sealable",
                "journal_manifest_digest", "journal_turn_key",
                "journal_claim_digest", "journal_result_digest",
                "journal_outcome_digest",
                "journal_terminal_record_digest",
                "physical_model_call_cold_authenticated",
                "benchmark_requires_external_journal_terminal",
            },
            "support positive transport provenance",
        )
        if (
            raw["schema"] != SUPPORT_POSITIVE_TRANSPORT_PROVENANCE_SCHEMA
            or raw["physical_model_call_cold_authenticated"] is not False
            or raw["benchmark_requires_external_journal_terminal"] is not True
        ):
            raise SupportPositiveProposerError("support positive provenance policy differs")
        result = cls(
            raw["kind"], raw["source_binding"],
            raw["production_transport_chain_verified"], raw["benchmark_sealable"],
            raw["journal_manifest_digest"], raw["journal_turn_key"],
            raw["journal_claim_digest"], raw["journal_result_digest"],
            raw["journal_outcome_digest"],
            raw["journal_terminal_record_digest"],
        )
        if result.to_data() != dict(raw):
            raise SupportPositiveProposerError("support positive provenance is not canonical")
        return result


def _transport_provenance(transport: object) -> SupportPositiveTransportProvenance:
    if transport is run_codex_named_images_structured:
        return SupportPositiveTransportProvenance.create("production_direct")
    if (
        type(transport) is ObjectBongardNamedImageTurnJournalTransport
        and getattr(transport, "_underlying_transport", None)
        is run_codex_named_images_structured
        and transport.runtime.transport_source_digest
        == _scene_runtime.prototype_scene_transport_source_digest()
    ):
        return SupportPositiveTransportProvenance.create(
            "production_exactly_once_journal", journal_summary=transport.verify()
        )
    return SupportPositiveTransportProvenance.create("injected_unverified")


def _verify_external_journal_terminal(
    provenance: SupportPositiveTransportProvenance,
    summary: ObjectBongardTurnJournalSummary | None,
) -> None:
    if provenance.kind != "production_exactly_once_journal":
        if summary is not None:
            raise SupportPositiveProposerError(
                "non-journal proposer artifact received external journal custody"
            )
        return
    if (
        type(summary) is not ObjectBongardTurnJournalSummary
        or summary.terminal_status != "success"
        or (
            summary.manifest_digest,
            summary.turn_key,
            summary.claim_digest,
            summary.result_digest,
            summary.outcome_digest,
            summary.record_digest,
        )
        != (
            provenance.journal_manifest_digest,
            provenance.journal_turn_key,
            provenance.journal_claim_digest,
            provenance.journal_result_digest,
            provenance.journal_outcome_digest,
            provenance.journal_terminal_record_digest,
        )
    ):
        raise SupportPositiveProposerError(
            "external proposer journal terminal differs from artifact custody"
        )


def _contract_digest(request: SupportPositiveProposerRequest) -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-support-positive-proposer-contract.v1",
            "protocol_id": SUPPORT_POSITIVE_PROPOSER_PROTOCOL_ID,
            "source_digest": panel_support_positive_proposer_source_digest(),
            "runtime": request.runtime.to_data(),
            "request_digest": request.request_digest,
            "presentation_names": list(SUPPORT_POSITIVE_PRESENTATION_NAMES),
            "support_image_count": SUPPORT_IMAGE_COUNT,
            "group_b_may_be_heterogeneous": True,
            "positive_conjunction_count": 1,
            "positive_component_count": 2,
            "cue_text_is_exact_ordered_component_join": True,
            "component_joiner": " and ",
            "per_drawing_estimate_count": SUPPORT_IMAGE_COUNT,
            "native_supports_required": 6,
            "contrast_does_not_support_required": 5,
            "contrast_contradiction_or_unclear_allowed": 1,
            "foil_or_complement_allowed": False,
            "query_material_admitted": False,
            "prose_is_inert": True,
            "threshold_polarity_or_code_selected": False,
        }
    )


@dataclass(frozen=True, slots=True)
class SupportPositiveProposerArtifact:
    """Exact support pixels, frozen rubric, full receipt/runtime, and provenance."""

    runtime: TypedCodexRuntimeBinding
    request: SupportPositiveProposerRequest
    transport_provenance: SupportPositiveTransportProvenance
    request_digest: str
    proposer_source_digest: str
    contract_digest: str
    prompt_digest: str
    output_schema_digest: str
    payload_digest: str
    model_payload: Mapping[str, Any]
    codex_receipt: CodexReceipt
    rubric: PositiveConjunctionRubric | None
    proposal_gap: SupportPositiveProposalGap | None

    def __post_init__(self) -> None:
        if type(self.runtime) is not TypedCodexRuntimeBinding:
            raise TypeError("support positive artifact needs TypedCodexRuntimeBinding")
        if type(self.request) is not SupportPositiveProposerRequest:
            raise TypeError("support positive artifact needs its exact request")
        if type(self.transport_provenance) is not SupportPositiveTransportProvenance:
            raise TypeError("support positive artifact needs transport provenance")
        if not (
            (type(self.rubric) is PositiveConjunctionRubric and self.proposal_gap is None)
            or (self.rubric is None and type(self.proposal_gap) is SupportPositiveProposalGap)
        ):
            raise TypeError("support positive artifact needs exactly one typed outcome")
        for label, item in (
            ("request digest", self.request_digest),
            ("proposer source digest", self.proposer_source_digest),
            ("contract digest", self.contract_digest),
            ("prompt digest", self.prompt_digest),
            ("output schema digest", self.output_schema_digest),
            ("payload digest", self.payload_digest),
        ):
            _digest(item, f"support positive {label}")
        payload = _canonical_payload(self.model_payload, "support positive payload")
        object.__setattr__(self, "model_payload", payload)
        prompt = support_positive_proposer_prompt(self.request)
        schema = support_positive_proposer_output_schema(self.request)
        if (
            self.runtime != self.request.runtime
            or self.request_digest != self.request.request_digest
            or self.proposer_source_digest != panel_support_positive_proposer_source_digest()
            or self.contract_digest != _contract_digest(self.request)
            or self.prompt_digest != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            or self.output_schema_digest != canonical_digest(schema)
            or self.payload_digest != canonical_digest(payload)
        ):
            raise SupportPositiveProposerError("support positive frozen envelope differs")
        parsed_rubric, parsed_gap = _parse_outcome(payload)
        if (parsed_rubric, parsed_gap) != (self.rubric, self.proposal_gap):
            raise SupportPositiveProposerError("support positive outcome replay differs")
        _validate_receipt_binding(
            self.codex_receipt,
            runtime=self.runtime,
            prompt_digest=self.prompt_digest,
            output_schema_digest=self.output_schema_digest,
            payload_digest=self.payload_digest,
            presentation=self.request.presentation,
        )

    @property
    def benchmark_sealable(self) -> bool:
        return self.transport_provenance.benchmark_sealable

    @property
    def artifact_digest(self) -> str:
        return canonical_digest(self.content_data())

    def content_data(self) -> dict[str, object]:
        return {
            "schema": SUPPORT_POSITIVE_PROPOSER_ARTIFACT_SCHEMA,
            "protocol_id": SUPPORT_POSITIVE_PROPOSER_PROTOCOL_ID,
            "runtime": self.runtime.to_data(),
            "request": self.request.to_data(),
            "transport_provenance": self.transport_provenance.to_data(),
            "benchmark_sealable": self.benchmark_sealable,
            "request_digest": self.request_digest,
            "proposer_source_digest": self.proposer_source_digest,
            "contract_digest": self.contract_digest,
            "prompt_digest": self.prompt_digest,
            "output_schema_digest": self.output_schema_digest,
            "payload_digest": self.payload_digest,
            "model_payload": dict(self.model_payload),
            "codex_receipt": self.codex_receipt.to_dict(),
            "codex_receipt_digest": self.codex_receipt.receipt_digest,
            "rubric": None if self.rubric is None else self.rubric.to_data(),
            "rubric_digest": None if self.rubric is None else self.rubric.rubric_digest,
            "proposal_gap": (
                None if self.proposal_gap is None else self.proposal_gap.to_data()
            ),
            "proposal_gap_digest": (
                None if self.proposal_gap is None else self.proposal_gap.gap_digest
            ),
            "rubric_admitted": self.rubric is not None,
            "support_image_count": SUPPORT_IMAGE_COUNT,
            "query_image_count": 0,
            "model_call_count": 1,
            "model_visible_image_names": list(SUPPORT_POSITIVE_PRESENTATION_NAMES),
            "group_b_may_be_heterogeneous": True,
            "foil_complement_negative_predicate_allowed": False,
            "task_phase_side_class_ids_model_visible": False,
            "candidate_formula_ids_model_visible": False,
            "prose_is_inert": True,
            "threshold_polarity_or_code_selected": False,
            "temporal_freeze_before_query_requires_external_custody": True,
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "artifact_digest": self.artifact_digest}

    @classmethod
    def from_data(cls, value: object) -> "SupportPositiveProposerArtifact":
        raw = _fields(
            value,
            {
                "schema", "protocol_id", "runtime", "request",
                "transport_provenance", "benchmark_sealable", "request_digest",
                "proposer_source_digest", "contract_digest", "prompt_digest",
                "output_schema_digest", "payload_digest", "model_payload",
                "codex_receipt", "codex_receipt_digest", "rubric", "rubric_digest",
                "proposal_gap", "proposal_gap_digest", "rubric_admitted",
                "support_image_count", "query_image_count", "model_call_count",
                "model_visible_image_names", "group_b_may_be_heterogeneous",
                "foil_complement_negative_predicate_allowed",
                "task_phase_side_class_ids_model_visible",
                "candidate_formula_ids_model_visible", "prose_is_inert",
                "threshold_polarity_or_code_selected",
                "temporal_freeze_before_query_requires_external_custody",
                "artifact_digest",
            },
            "support positive proposer artifact",
        )
        if (
            raw["schema"] != SUPPORT_POSITIVE_PROPOSER_ARTIFACT_SCHEMA
            or raw["protocol_id"] != SUPPORT_POSITIVE_PROPOSER_PROTOCOL_ID
            or raw["support_image_count"] != SUPPORT_IMAGE_COUNT
            or raw["query_image_count"] != 0
            or raw["model_call_count"] != 1
            or raw["model_visible_image_names"] != list(SUPPORT_POSITIVE_PRESENTATION_NAMES)
            or raw["group_b_may_be_heterogeneous"] is not True
            or raw["prose_is_inert"] is not True
            or raw["temporal_freeze_before_query_requires_external_custody"] is not True
            or any(
                raw[key] is not False
                for key in (
                    "foil_complement_negative_predicate_allowed",
                    "task_phase_side_class_ids_model_visible",
                    "candidate_formula_ids_model_visible",
                    "threshold_polarity_or_code_selected",
                )
            )
        ):
            raise SupportPositiveProposerError("support positive artifact policy differs")
        result = cls(
            TypedCodexRuntimeBinding.from_data(raw["runtime"]),
            SupportPositiveProposerRequest.from_data(raw["request"]),
            SupportPositiveTransportProvenance.from_data(raw["transport_provenance"]),
            raw["request_digest"], raw["proposer_source_digest"],
            raw["contract_digest"], raw["prompt_digest"],
            raw["output_schema_digest"], raw["payload_digest"],
            _canonical_payload(raw["model_payload"], "archived support positive payload"),
            _receipt_from_data(raw["codex_receipt"]),
            (
                None
                if raw["rubric"] is None
                else PositiveConjunctionRubric.from_data(raw["rubric"])
            ),
            (
                None
                if raw["proposal_gap"] is None
                else SupportPositiveProposalGap.from_data(raw["proposal_gap"])
            ),
        )
        if (
            raw["benchmark_sealable"] is not result.benchmark_sealable
            or raw["codex_receipt_digest"] != result.codex_receipt.receipt_digest
            or raw["rubric_digest"]
            != (None if result.rubric is None else result.rubric.rubric_digest)
            or raw["proposal_gap_digest"]
            != (
                None
                if result.proposal_gap is None
                else result.proposal_gap.gap_digest
            )
            or raw["rubric_admitted"] is not (result.rubric is not None)
            or raw["artifact_digest"] != result.artifact_digest
            or result.to_data() != dict(raw)
        ):
            raise SupportPositiveProposerError("support positive artifact digest differs")
        return result


def propose_support_positive_rubric(
    group_a_pngs: Sequence[bytes],
    group_b_pngs: Sequence[bytes],
    *,
    request: SupportPositiveProposerRequest,
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
) -> SupportPositiveProposerArtifact:
    """Make one support-only call and freeze its inert positive rubric."""

    if type(request) is not SupportPositiveProposerRequest:
        raise TypeError("support positive call needs SupportPositiveProposerRequest")
    first = _freeze_group(group_a_pngs, "group A")
    second = _freeze_group(group_b_pngs, "group B")
    presentation = tuple(zip(SUPPORT_POSITIVE_PRESENTATION_NAMES, (*first, *second), strict=True))
    rebuilt = SupportPositiveProposerRequest.build(first, second, runtime=request.runtime)
    if rebuilt != request:
        raise SupportPositiveProposerError("support positive request belongs to other pixels")
    runtime = _bind_runtime(
        model=model,
        reasoning_effort=reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
        model_catalog_snapshot=model_catalog_snapshot,
        no_tools_attestation=no_tools_attestation,
    )
    if runtime != request.runtime:
        raise SupportPositiveProposerError("support positive request runtime differs")
    if type(minutes) is not int or minutes <= 0 or not callable(transport):
        raise SupportPositiveProposerError("support positive call configuration differs")
    prompt = support_positive_proposer_prompt(request)
    schema = support_positive_proposer_output_schema(request)
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
        frozen = _canonical_payload(payload, "support positive payload")
        if len(canonical_json(frozen)) > MAX_RESPONSE_UTF8_BYTES:
            raise SupportPositiveProposerError("support positive payload exceeds capacity")
        rubric, proposal_gap = _parse_outcome(frozen)
        provenance = _transport_provenance(transport)
        return SupportPositiveProposerArtifact(
            runtime, request, provenance, request.request_digest,
            panel_support_positive_proposer_source_digest(), _contract_digest(request),
            hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            canonical_digest(schema), canonical_digest(frozen), frozen, receipt,
            rubric, proposal_gap,
        )
    except SupportPositiveProposerError:
        raise
    except Exception as exc:
        raise SupportPositiveProposerError(
            "support positive call or parser failed closed; no artifact was produced"
        ) from exc


def verify_support_positive_proposer_artifact(
    artifact: SupportPositiveProposerArtifact,
    group_a_pngs: Sequence[bytes],
    group_b_pngs: Sequence[bytes],
    *,
    expected_artifact_digest: str,
    proposer_journal_terminal: ObjectBongardTurnJournalSummary | None = None,
) -> SupportPositiveProposerArtifact:
    """Cold replay pixels, source, journal custody, receipt, and prose parser."""

    if type(artifact) is not SupportPositiveProposerArtifact:
        raise TypeError("support positive replay needs SupportPositiveProposerArtifact")
    expected = _digest(expected_artifact_digest, "expected support positive artifact digest")
    restored = SupportPositiveProposerArtifact.from_data(artifact.to_data())
    if restored.artifact_digest != expected:
        raise SupportPositiveProposerError("support positive artifact differs from commitment")
    _verify_external_journal_terminal(
        restored.transport_provenance, proposer_journal_terminal
    )
    first = _freeze_group(group_a_pngs, "group A")
    second = _freeze_group(group_b_pngs, "group B")
    rebuilt = SupportPositiveProposerRequest.build(first, second, runtime=restored.runtime)
    if rebuilt != restored.request:
        raise SupportPositiveProposerError("support positive replay pixels differ")
    prompt = support_positive_proposer_prompt(rebuilt)
    schema = support_positive_proposer_output_schema(rebuilt)
    with tempfile.TemporaryDirectory(prefix="bongard-support-positive-replay-") as raw:
        directory = Path(raw)
        paths: list[str] = []
        for name, data in zip(SUPPORT_POSITIVE_PRESENTATION_NAMES, (*first, *second), strict=True):
            target = directory / name
            target.write_bytes(data)
            paths.append(str(target.resolve()))
        try:
            validate_codex_named_image_receipt(
                restored.codex_receipt,
                prompt,
                tuple(paths),
                SUPPORT_POSITIVE_PRESENTATION_NAMES,
                schema,
                dict(restored.model_payload),
            )
        except Exception as exc:
            raise SupportPositiveProposerError(
                "support positive receipt cold replay failed"
            ) from exc
        for path, expected_bytes in zip(paths, (*first, *second), strict=True):
            if Path(path).read_bytes() != expected_bytes:
                raise SupportPositiveProposerError("support positive replay pixels changed")
    if _parse_outcome(restored.model_payload) != (
        restored.rubric,
        restored.proposal_gap,
    ):
        raise SupportPositiveProposerError("support positive outcome replay differs")
    return restored


__all__ = [
    "PositiveConjunctionRubric",
    "SUPPORT_POSITIVE_PRESENTATION_NAMES",
    "SupportPositiveProposerArtifact",
    "SupportPositiveProposerError",
    "SupportPositiveProposalGap",
    "SupportPositiveProposerRequest",
    "SupportPositiveTransportProvenance",
    "panel_support_positive_proposer_source_digest",
    "propose_support_positive_rubric",
    "support_positive_proposer_output_schema",
    "support_positive_proposer_prompt",
    "verify_support_positive_proposer_artifact",
]
