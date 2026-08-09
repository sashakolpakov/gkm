"""One-call proposer for affirmative predicates over complete support panels.

The proposer shows Codex exactly twelve raw PNGs.  It never constructs crops,
atlases, masks, or object hypotheses.  The receipted model output is inert
prose; :class:`PanelSoftAtom` applies the closed Python lexical policy.

``raw_proposer_evidence_digest`` intentionally differs from the final artifact
digest.  It commits the raw images, prompt, schema, runtime, payload, and Codex
receipt, while excluding parsed drops, the vocabulary, and the final artifact.
Atoms bind that acyclic evidence digest through their historical
``proposer_artifact_digest`` field.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.object_bongard_soft_cues import object_bongard_soft_cue_grammar_digest
from bongard.panel_soft_predicate import (
    PANEL_SOFT_ORIENTATIONS,
    PanelSoftAtom,
    PanelSoftPredicateError,
    PanelSoftVocabulary,
    panel_soft_atom_text_grammar_digest,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from bongard import prototype_scene_observer as _scene_runtime
from bongard.prototype_scene_observer import (
    CloudPolicyCacheSnapshot,
    CodexModelCatalogSnapshot,
    CodexNoToolsAttestation,
    PrototypeImageIdentity,
)
from bongard.transport import (
    CodexReceipt,
    NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
    run_codex_named_images_structured,
    validate_codex_named_image_receipt,
    validate_codex_strict_output_schema,
)


PANEL_SOFT_PROPOSER_ARTIFACT_SCHEMA = "gkm.bongard-panel-soft-proposer-artifact.v1"
PANEL_SOFT_PROPOSER_DROP_SCHEMA = "gkm.bongard-panel-soft-proposer-drop.v1"
PANEL_SOFT_PROPOSER_EVIDENCE_SCHEMA = "gkm.bongard-panel-soft-proposer-evidence.v1"
PANEL_SOFT_PROPOSER_PROTOCOL_ID = "bongard.panel-soft-proposer/raw-support-one-call-v1"
PANEL_SOFT_PROPOSER_PRESENTATION_NAMES = tuple(
    f"panel_{index:03d}.png" for index in range(12)
)
PANEL_SOFT_PROPOSER_ROWS_PER_ORIENTATION = 4
PANEL_SOFT_PROPOSER_WITNESSES_PER_ROW = 2

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_SUPPORT_PANEL_ID = re.compile(
    r"(?P<family>[A-Za-z0-9_.-]+)/(?P<task>[A-Za-z0-9_.-]+)/"
    r"(?P<side>[01])/(?P<index>[0-6])\.png\Z"
)


class PanelSoftProposerError(ValueError):
    """A proposer input, artifact, or cold replay is invalid."""


class PanelSoftProposerStatus(str, Enum):
    SUCCESS = "success"
    TRANSPORT_ERROR = "transport_error"
    PARSER_ERROR = "parser_error"


class PanelSoftProposerDropCode(str, Enum):
    SEMANTIC_ROW_REJECTED = "semantic_row_rejected"
    DUPLICATE_SEMANTIC_ROW = "duplicate_semantic_row"


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "prose_is_observed_not_executed": True,
        "open_prose_semantic_positivity_proved": False,
        "arbitrary_code_allowed": False,
        "negation_allowed": False,
        "polarity_flip_allowed": False,
        "whole_panel_only": True,
        "segmentation_allowed": False,
        "crop_allowed": False,
        "atlas_allowed": False,
        "standalone_query_exclusion_verified": False,
        "task_runner_exact_plan_binding_required": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_checker_optional": True,
        "lean_affects_identity_or_decision": False,
    }


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != expected
    ):
        raise PanelSoftProposerError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise PanelSoftProposerError(f"{label} must be a raw SHA-256")
    return value


def _address_or_absent(value: object, label: str) -> str:
    if value != "absent" and (
        not isinstance(value, str) or _ADDRESS.fullmatch(value) is None
    ):
        raise PanelSoftProposerError(f"{label} differs")
    return value  # type: ignore[return-value]


def _canonical_payload(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise PanelSoftProposerError("proposer payload must be an object")
    try:
        result = json.loads(canonical_json(dict(value)).decode("utf-8"))
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise PanelSoftProposerError("proposer payload is not canonical JSON") from exc
    if not isinstance(result, dict):
        raise PanelSoftProposerError("proposer payload must be an object")
    return result


def _receipt_data(value: CodexReceipt | None) -> object:
    return None if value is None else value.to_dict()


def _receipt_from_data(value: object) -> CodexReceipt | None:
    if value is None:
        return None
    result = _scene_runtime._receipt_from_data(value)
    if not isinstance(result, CodexReceipt):
        raise PanelSoftProposerError("proposer receipt has the wrong type")
    return result


def panel_soft_proposer_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _field_name(orientation_index: int, rank: int, field: str) -> str:
    return f"side{orientation_index}_atom{rank}_{field}"


def panel_soft_proposer_output_schema() -> dict[str, object]:
    """Strict fixed-field schema: no variable arrays or size keywords."""

    properties: dict[str, object] = {}
    for orientation_index in range(2):
        for rank in range(PANEL_SOFT_PROPOSER_ROWS_PER_ORIENTATION):
            for field in ("phrase", "witness_a", "witness_b"):
                properties[_field_name(orientation_index, rank, field)] = {
                    "type": "string"
                }
    schema: dict[str, object] = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    validate_codex_strict_output_schema(schema)
    return schema


def panel_soft_proposer_prompt() -> str:
    return (
        "Inspect the twelve complete drawings exactly as supplied. "
        "panel_000.png through panel_005.png form group 0. "
        "panel_006.png through panel_011.png form group 1. Return exactly four "
        "affirmative complete-panel visual atoms for each native orientation. "
        "Fields beginning side0_ describe recurring visible qualities of group 0. "
        "Fields beginning side1_ describe recurring visible qualities of group 1. "
        "Every atom has one phrase plus exactly two positive witness clauses. "
        "A phrase states one concise visible quality of a complete drawing. Each "
        "witness states one visible mark or global form supporting that phrase. "
        "Bird-like objects, oblique angles, smooth-bend contours, gestalt shape, "
        "stroke character, enclosure, contact, symmetry, and spatial arrangement "
        "are permitted when genuinely visible. Keep every field atomic, affirmative, "
        "single-line, printable ASCII prose. Do not place the words 'and' or 'or' "
        "in any field. Do not use digits or operator symbols. Do not use negation, "
        "absence, contrast, comparison, reversed orientation, code, numeric thresholds, "
        "experimental roles, segmentation, crops, atlases, or hidden construction. "
        "Forbidden control or negative forms include no, not, never, without, absent, "
        "missing, only, instead, less, free, non-connected, unfilled, disconnected, "
        "ignore, override, follow, "
        "return, output, answer, choose, classify, score, instruction, prompt, system, "
        "developer, assistant, user, model, tool, Python, Lean, schema, JSON, criterion, "
        "verdict, present, mismatch, indeterminate, and error. "
        "Fill every fixed field."
    )


def panel_soft_proposer_protocol_digest() -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-panel-soft-proposer-protocol.v1",
            "protocol_id": PANEL_SOFT_PROPOSER_PROTOCOL_ID,
            "source_digest": panel_soft_proposer_source_digest(),
            "soft_cue_grammar_digest": object_bongard_soft_cue_grammar_digest(),
            "panel_atom_text_grammar_digest": panel_soft_atom_text_grammar_digest(),
            "transport_source_digest": _scene_runtime.prototype_scene_transport_source_digest(),
            "prompt": panel_soft_proposer_prompt(),
            "output_schema": panel_soft_proposer_output_schema(),
            "presentation_names": list(PANEL_SOFT_PROPOSER_PRESENTATION_NAMES),
            "group_partition": [list(range(0, 6)), list(range(6, 12))],
            "logical_proposer_attempts": 1,
            "transport_invocations_on_model_output": 1,
            "receipted_calls_on_model_output": 1,
            "raw_rows_per_orientation": PANEL_SOFT_PROPOSER_ROWS_PER_ORIENTATION,
            "positive_witnesses_per_row": PANEL_SOFT_PROPOSER_WITNESSES_PER_ROW,
            "minimum_valid_rows_per_orientation": 1,
            "duplicate_semantic_rows": "preserve_first_global_raw_order",
            **_authority_data(),
        }
    )


def _model_digest(model: str, reasoning_effort: str) -> str:
    if not isinstance(model, str) or not model or not isinstance(reasoning_effort, str) or not reasoning_effort:
        raise PanelSoftProposerError("model request differs")
    return canonical_digest(
        {
            "schema": "gkm.bongard-panel-soft-proposer-model.v1",
            "model": model,
            "reasoning_effort": reasoning_effort,
        }
    )


def _runtime_identity_digest(
    *,
    model: str,
    reasoning_effort: str,
    expected_launcher_digest: str,
    cloud_policy_cache_binding: str,
    model_catalog_digest: str,
    no_tools_attestation_digest: str,
) -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-panel-soft-proposer-runtime.v1",
            "model_digest": _model_digest(model, reasoning_effort),
            "expected_launcher_digest": expected_launcher_digest,
            "cloud_policy_cache_binding": cloud_policy_cache_binding,
            "model_catalog_digest": model_catalog_digest,
            "no_tools_attestation_digest": no_tools_attestation_digest,
            "runtime_helper_source_digest": _scene_runtime.prototype_scene_observer_source_digest(),
            "transport_source_digest": _scene_runtime.prototype_scene_transport_source_digest(),
        }
    )


@dataclass(frozen=True, order=True, slots=True)
class PanelSoftProposerDrop:
    orientation: str
    raw_rank: int
    code: PanelSoftProposerDropCode

    def __post_init__(self) -> None:
        if self.orientation not in PANEL_SOFT_ORIENTATIONS:
            raise PanelSoftProposerError("drop orientation differs")
        if type(self.raw_rank) is not int or self.raw_rank not in range(4):
            raise PanelSoftProposerError("drop raw rank differs")
        if not isinstance(self.code, PanelSoftProposerDropCode):
            raise TypeError("drop code has the wrong type")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": PANEL_SOFT_PROPOSER_DROP_SCHEMA,
            "orientation": self.orientation,
            "raw_rank": self.raw_rank,
            "code": self.code.value,
        }

    @classmethod
    def from_data(cls, value: object) -> "PanelSoftProposerDrop":
        raw = _fields(value, {"schema", "orientation", "raw_rank", "code"}, "proposer drop")
        if raw["schema"] != PANEL_SOFT_PROPOSER_DROP_SCHEMA:
            raise PanelSoftProposerError("drop schema differs")
        try:
            result = cls(raw["orientation"], raw["raw_rank"], PanelSoftProposerDropCode(raw["code"]))
        except (TypeError, ValueError) as exc:
            raise PanelSoftProposerError("drop differs") from exc
        if result.to_data() != dict(raw):
            raise PanelSoftProposerError("drop is not canonical")
        return result


def _raw_evidence_content(
    *,
    source_digest: str,
    protocol_digest: str,
    transport_source_digest: str,
    prompt_digest: str,
    output_schema_digest: str,
    model: str,
    reasoning_effort: str,
    model_digest: str,
    expected_launcher_digest: str,
    cloud_policy_cache_binding: str,
    model_catalog_digest: str,
    no_tools_attestation_digest: str,
    runtime_identity_digest: str,
    support_panel_ids: Sequence[str],
    presentation: Sequence[PrototypeImageIdentity],
    payload: Mapping[str, Any],
    receipt: CodexReceipt,
) -> dict[str, object]:
    return {
        "schema": PANEL_SOFT_PROPOSER_EVIDENCE_SCHEMA,
        "source_digest": source_digest,
        "protocol_digest": protocol_digest,
        "transport_source_digest": transport_source_digest,
        "prompt_digest": prompt_digest,
        "output_schema_digest": output_schema_digest,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "model_digest": model_digest,
        "expected_launcher_digest": expected_launcher_digest,
        "cloud_policy_cache_binding": cloud_policy_cache_binding,
        "model_catalog_digest": model_catalog_digest,
        "no_tools_attestation_digest": no_tools_attestation_digest,
        "runtime_identity_digest": runtime_identity_digest,
        "support_panel_ids": list(support_panel_ids),
        "presentation": [item.to_data() for item in presentation],
        "logical_proposer_attempt_count": 1,
        "transport_invocation_count": 1,
        "receipted_call_count": 1,
        "model_payload": dict(payload),
        "receipt": receipt.to_dict(),
        "excludes_parsed_drops": True,
        "excludes_vocabulary": True,
        "excludes_final_artifact_digest": True,
        **_authority_data(),
    }


def _raw_evidence_digest(**kwargs: Any) -> str:
    return canonical_digest(_raw_evidence_content(**kwargs))


def _payload_fields() -> set[str]:
    return {
        _field_name(orientation_index, rank, field)
        for orientation_index in range(2)
        for rank in range(4)
        for field in ("phrase", "witness_a", "witness_b")
    }


def _parse_payload(
    value: object, raw_evidence_digest: str
) -> tuple[PanelSoftVocabulary | None, tuple[PanelSoftProposerDrop, ...]]:
    raw = _fields(value, _payload_fields(), "panel soft proposer payload")
    accepted: list[PanelSoftAtom] = []
    drops: list[PanelSoftProposerDrop] = []
    semantic_keys: set[tuple[str, tuple[str, str]]] = set()
    raw_global_rank = 0
    for orientation_index, orientation in enumerate(PANEL_SOFT_ORIENTATIONS):
        for rank in range(4):
            try:
                candidate = PanelSoftAtom.create(
                    atom_id=f"atom_{raw_global_rank:04d}",
                    orientation=orientation,
                    phrase=raw[_field_name(orientation_index, rank, "phrase")],
                    witnesses=(
                        raw[_field_name(orientation_index, rank, "witness_a")],
                        raw[_field_name(orientation_index, rank, "witness_b")],
                    ),
                    proposer_artifact_digest=raw_evidence_digest,
                )
            except (PanelSoftPredicateError, TypeError, ValueError):
                drops.append(PanelSoftProposerDrop(orientation, rank, PanelSoftProposerDropCode.SEMANTIC_ROW_REJECTED))
                raw_global_rank += 1
                continue
            semantic_key = (
                candidate.phrase.cue_digest,
                tuple(sorted(item.cue_digest for item in candidate.witnesses)),
            )
            if semantic_key in semantic_keys:
                drops.append(PanelSoftProposerDrop(orientation, rank, PanelSoftProposerDropCode.DUPLICATE_SEMANTIC_ROW))
                raw_global_rank += 1
                continue
            semantic_keys.add(semantic_key)
            accepted.append(candidate)
            raw_global_rank += 1
    if {item.orientation for item in accepted} != set(PANEL_SOFT_ORIENTATIONS):
        return None, tuple(drops)
    atoms = tuple(
        PanelSoftAtom.create(
            atom_id=f"atom_{index:04d}",
            orientation=item.orientation,
            phrase=item.phrase,
            witnesses=item.witnesses,
            proposer_artifact_digest=raw_evidence_digest,
        )
        for index, item in enumerate(accepted)
    )
    return PanelSoftVocabulary.create(atoms), tuple(drops)


def _artifact_content(value: "PanelSoftProposerArtifact") -> dict[str, object]:
    return {
        "schema": PANEL_SOFT_PROPOSER_ARTIFACT_SCHEMA,
        "source_digest": value.source_digest,
        "soft_cue_grammar_digest": value.soft_cue_grammar_digest,
        "panel_atom_text_grammar_digest": value.panel_atom_text_grammar_digest,
        "protocol_digest": value.protocol_digest,
        "transport_source_digest": value.transport_source_digest,
        "prompt_digest": value.prompt_digest,
        "output_schema_digest": value.output_schema_digest,
        "model": value.model,
        "reasoning_effort": value.reasoning_effort,
        "model_digest": value.model_digest,
        "expected_launcher_digest": value.expected_launcher_digest,
        "cloud_policy_cache_binding": value.cloud_policy_cache_binding,
        "model_catalog_digest": value.model_catalog_digest,
        "no_tools_attestation_digest": value.no_tools_attestation_digest,
        "runtime_identity_digest": value.runtime_identity_digest,
        "support_panel_ids": list(value.support_panel_ids),
        "presentation": [item.to_data() for item in value.presentation],
        "logical_proposer_attempt_count": value.logical_proposer_attempt_count,
        "transport_invocation_count": value.transport_invocation_count,
        "receipted_call_count": value.receipted_call_count,
        "status": value.status.value,
        "model_payload": value.model_payload,
        "receipt": _receipt_data(value.receipt),
        "raw_proposer_evidence_digest": value.raw_proposer_evidence_digest,
        "atom_binding_field": "proposer_artifact_digest",
        "atom_binding_semantics": "raw_proposer_evidence_digest_not_final_artifact_digest",
        "drops": [item.to_data() for item in value.drops],
        "drop_count": len(value.drops),
        "vocabulary": None if value.vocabulary is None else value.vocabulary.to_data(),
        "failure_code": value.failure_code,
        "failure_type": value.failure_type,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PanelSoftProposerArtifact:
    source_digest: str
    soft_cue_grammar_digest: str
    panel_atom_text_grammar_digest: str
    protocol_digest: str
    transport_source_digest: str
    prompt_digest: str
    output_schema_digest: str
    model: str
    reasoning_effort: str
    model_digest: str
    expected_launcher_digest: str
    cloud_policy_cache_binding: str
    model_catalog_digest: str
    no_tools_attestation_digest: str
    runtime_identity_digest: str
    support_panel_ids: tuple[str, ...]
    presentation: tuple[PrototypeImageIdentity, ...]
    logical_proposer_attempt_count: int
    transport_invocation_count: int
    receipted_call_count: int
    status: PanelSoftProposerStatus
    model_payload: Mapping[str, Any] | None
    receipt: CodexReceipt | None
    raw_proposer_evidence_digest: str | None
    drops: tuple[PanelSoftProposerDrop, ...]
    vocabulary: PanelSoftVocabulary | None
    failure_code: str | None
    failure_type: str | None
    artifact_digest: str

    def __post_init__(self) -> None:
        for label, value in (
            ("source digest", self.source_digest),
            ("soft cue grammar digest", self.soft_cue_grammar_digest),
            ("panel atom text grammar digest", self.panel_atom_text_grammar_digest),
            ("protocol digest", self.protocol_digest),
            ("transport source digest", self.transport_source_digest),
            ("prompt digest", self.prompt_digest),
            ("output schema digest", self.output_schema_digest),
            ("model digest", self.model_digest),
            ("expected launcher digest", self.expected_launcher_digest),
            ("model catalog digest", self.model_catalog_digest),
            ("no tools attestation digest", self.no_tools_attestation_digest),
            ("runtime identity digest", self.runtime_identity_digest),
            ("artifact digest", self.artifact_digest),
        ):
            _digest(value, label)
        _address_or_absent(self.cloud_policy_cache_binding, "cloud policy cache binding")
        prompt = panel_soft_proposer_prompt()
        schema = panel_soft_proposer_output_schema()
        if (
            self.source_digest != panel_soft_proposer_source_digest()
            or self.soft_cue_grammar_digest != object_bongard_soft_cue_grammar_digest()
            or self.panel_atom_text_grammar_digest != panel_soft_atom_text_grammar_digest()
            or self.protocol_digest != panel_soft_proposer_protocol_digest()
            or self.transport_source_digest != _scene_runtime.prototype_scene_transport_source_digest()
            or self.prompt_digest != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            or self.output_schema_digest != canonical_digest(schema)
            or self.model_digest != _model_digest(self.model, self.reasoning_effort)
            or self.runtime_identity_digest != _runtime_identity_digest(
                model=self.model,
                reasoning_effort=self.reasoning_effort,
                expected_launcher_digest=self.expected_launcher_digest,
                cloud_policy_cache_binding=self.cloud_policy_cache_binding,
                model_catalog_digest=self.model_catalog_digest,
                no_tools_attestation_digest=self.no_tools_attestation_digest,
            )
        ):
            raise PanelSoftProposerError("artifact source, protocol, or runtime differs")
        if (
            _validated_support_panel_ids(self.support_panel_ids) != self.support_panel_ids
            or type(self.presentation) is not tuple
            or len(self.presentation) != 12
            or tuple(item.name for item in self.presentation) != PANEL_SOFT_PROPOSER_PRESENTATION_NAMES
            or type(self.logical_proposer_attempt_count) is not int
            or self.logical_proposer_attempt_count != 1
            or type(self.transport_invocation_count) is not int
            or self.transport_invocation_count not in (0, 1)
            or type(self.receipted_call_count) is not int
            or self.receipted_call_count not in (0, 1)
            or not isinstance(self.status, PanelSoftProposerStatus)
            or type(self.drops) is not tuple
            or any(not isinstance(item, PanelSoftProposerDrop) for item in self.drops)
            or tuple(sorted(self.drops)) != self.drops
            or len(set(self.drops)) != len(self.drops)
        ):
            raise PanelSoftProposerError("artifact presentation, status, or drops differ")
        if self.model_payload is not None:
            object.__setattr__(self, "model_payload", _canonical_payload(self.model_payload))
        if self.status is PanelSoftProposerStatus.TRANSPORT_ERROR:
            if any(item is not None for item in (self.model_payload, self.receipt, self.raw_proposer_evidence_digest, self.vocabulary)) or self.drops:
                raise PanelSoftProposerError("transport error fabricates proposer evidence")
        else:
            if self.model_payload is None or self.receipt is None or self.raw_proposer_evidence_digest is None:
                raise PanelSoftProposerError("receipted proposer artifact is incomplete")
            _digest(self.raw_proposer_evidence_digest, "raw proposer evidence digest")
            expected_evidence = _raw_evidence_digest(
                source_digest=self.source_digest,
                protocol_digest=self.protocol_digest,
                transport_source_digest=self.transport_source_digest,
                prompt_digest=self.prompt_digest,
                output_schema_digest=self.output_schema_digest,
                model=self.model,
                reasoning_effort=self.reasoning_effort,
                model_digest=self.model_digest,
                expected_launcher_digest=self.expected_launcher_digest,
                cloud_policy_cache_binding=self.cloud_policy_cache_binding,
                model_catalog_digest=self.model_catalog_digest,
                no_tools_attestation_digest=self.no_tools_attestation_digest,
                runtime_identity_digest=self.runtime_identity_digest,
                support_panel_ids=self.support_panel_ids,
                presentation=self.presentation,
                payload=self.model_payload,
                receipt=self.receipt,
            )
            if self.raw_proposer_evidence_digest != expected_evidence:
                raise PanelSoftProposerError("raw proposer evidence digest differs")
            try:
                parsed_vocabulary, parsed_drops = _parse_payload(
                    self.model_payload, self.raw_proposer_evidence_digest
                )
            except PanelSoftProposerError:
                parsed_vocabulary, parsed_drops = None, ()
            if parsed_drops != self.drops or parsed_vocabulary != self.vocabulary:
                raise PanelSoftProposerError("artifact parse projection differs")
        if self.status is PanelSoftProposerStatus.SUCCESS:
            if self.vocabulary is None or self.failure_code is not None or self.failure_type is not None:
                raise PanelSoftProposerError("successful proposer artifact differs")
        else:
            if self.vocabulary is not None or not isinstance(self.failure_code, str) or _CODE.fullmatch(self.failure_code) is None or not isinstance(self.failure_type, str) or _CODE.fullmatch(self.failure_type) is None:
                raise PanelSoftProposerError("failed proposer artifact differs")
        if (
            self.receipted_call_count != (1 if self.receipt is not None else 0)
            or self.receipted_call_count > self.transport_invocation_count
            or (
                self.status is not PanelSoftProposerStatus.TRANSPORT_ERROR
                and self.transport_invocation_count != 1
            )
        ):
            raise PanelSoftProposerError("transport or receipt accounting differs")
        if self.receipt is not None:
            assert self.model_payload is not None
            view = [item.to_data() for item in self.presentation]
            expected_set = "sha256:" + canonical_digest(
                {"schema": NAMED_IMAGE_INPUT_DIGEST_SCHEMA, "images": view}
            )
            receipt = self.receipt
            if (
                receipt.prompt_digest != self.prompt_digest
                or receipt.output_schema_digest != self.output_schema_digest
                or receipt.structured_output_digest != canonical_digest(dict(self.model_payload))
                or receipt.panel_view_digest != canonical_digest(view)
                or receipt.panel_set_digest != expected_set
                or receipt.requested_model != self.model
                or receipt.requested_reasoning_effort != self.reasoning_effort
                or receipt.codex_launcher_digest != self.expected_launcher_digest
                or receipt.cloud_config_bundle_cache_binding != self.cloud_policy_cache_binding
                or receipt.model_catalog_digest != self.model_catalog_digest
                or receipt.tool_surface_attestation_digest != self.no_tools_attestation_digest
            ):
                raise PanelSoftProposerError("artifact receipt binding differs")
        if self.artifact_digest != canonical_digest(_artifact_content(self)):
            raise PanelSoftProposerError("artifact digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_artifact_content(self), "artifact_digest": self.artifact_digest}

    @classmethod
    def from_data(cls, value: object) -> "PanelSoftProposerArtifact":
        expected = set(_artifact_content_fields()) | {"artifact_digest"}
        raw = _fields(value, expected, "panel soft proposer artifact")
        if (
            raw["schema"] != PANEL_SOFT_PROPOSER_ARTIFACT_SCHEMA
            or raw["atom_binding_field"] != "proposer_artifact_digest"
            or raw["atom_binding_semantics"] != "raw_proposer_evidence_digest_not_final_artifact_digest"
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["support_panel_ids"], list)
            or not isinstance(raw["presentation"], list)
            or not isinstance(raw["drops"], list)
            or raw["drop_count"] != len(raw["drops"])
        ):
            raise PanelSoftProposerError("artifact policy differs")
        try:
            result = cls(
                raw["source_digest"], raw["soft_cue_grammar_digest"],
                raw["panel_atom_text_grammar_digest"], raw["protocol_digest"],
                raw["transport_source_digest"], raw["prompt_digest"],
                raw["output_schema_digest"], raw["model"], raw["reasoning_effort"],
                raw["model_digest"], raw["expected_launcher_digest"],
                raw["cloud_policy_cache_binding"], raw["model_catalog_digest"],
                raw["no_tools_attestation_digest"], raw["runtime_identity_digest"],
                tuple(raw["support_panel_ids"]),
                tuple(PrototypeImageIdentity.from_data(item) for item in raw["presentation"]),
                raw["logical_proposer_attempt_count"], raw["transport_invocation_count"],
                raw["receipted_call_count"],
                PanelSoftProposerStatus(raw["status"]),
                raw["model_payload"], _receipt_from_data(raw["receipt"]),
                raw["raw_proposer_evidence_digest"],
                tuple(PanelSoftProposerDrop.from_data(item) for item in raw["drops"]),
                None if raw["vocabulary"] is None else PanelSoftVocabulary.from_data(raw["vocabulary"]),
                raw["failure_code"], raw["failure_type"], raw["artifact_digest"],
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, PanelSoftProposerError):
                raise
            raise PanelSoftProposerError("artifact fields differ") from exc
        if result.to_data() != dict(raw):
            raise PanelSoftProposerError("artifact is not canonical")
        return result


def _artifact_content_fields() -> tuple[str, ...]:
    return (
        "schema", "source_digest", "soft_cue_grammar_digest",
        "panel_atom_text_grammar_digest", "protocol_digest", "transport_source_digest",
        "prompt_digest", "output_schema_digest", "model", "reasoning_effort",
        "model_digest", "expected_launcher_digest", "cloud_policy_cache_binding",
        "model_catalog_digest", "no_tools_attestation_digest", "runtime_identity_digest",
        "support_panel_ids", "presentation", "logical_proposer_attempt_count",
        "transport_invocation_count", "receipted_call_count", "status",
        "model_payload", "receipt",
        "raw_proposer_evidence_digest", "atom_binding_field", "atom_binding_semantics",
        "drops", "drop_count", "vocabulary", "failure_code", "failure_type",
        *_authority_data(),
    )


def _seal_artifact(
    *,
    model: str,
    reasoning_effort: str,
    expected_launcher_digest: str,
    cloud_policy_cache_binding: str,
    model_catalog_digest: str,
    no_tools_attestation_digest: str,
    support_panel_ids: tuple[str, ...],
    presentation: tuple[PrototypeImageIdentity, ...],
    transport_invocation_count: int,
    status: PanelSoftProposerStatus,
    payload: Mapping[str, Any] | None,
    receipt: CodexReceipt | None,
    drops: tuple[PanelSoftProposerDrop, ...],
    vocabulary: PanelSoftVocabulary | None,
    failure_code: str | None,
    failure_type: str | None,
) -> PanelSoftProposerArtifact:
    prompt = panel_soft_proposer_prompt()
    schema = panel_soft_proposer_output_schema()
    source_digest = panel_soft_proposer_source_digest()
    protocol_digest = panel_soft_proposer_protocol_digest()
    transport_digest = _scene_runtime.prototype_scene_transport_source_digest()
    prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    schema_digest = canonical_digest(schema)
    model_digest = _model_digest(model, reasoning_effort)
    runtime_digest = _runtime_identity_digest(
        model=model,
        reasoning_effort=reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_binding=cloud_policy_cache_binding,
        model_catalog_digest=model_catalog_digest,
        no_tools_attestation_digest=no_tools_attestation_digest,
    )
    frozen_payload = None if payload is None else _canonical_payload(payload)
    evidence_digest = None
    if frozen_payload is not None and receipt is not None:
        evidence_digest = _raw_evidence_digest(
            source_digest=source_digest,
            protocol_digest=protocol_digest,
            transport_source_digest=transport_digest,
            prompt_digest=prompt_digest,
            output_schema_digest=schema_digest,
            model=model,
            reasoning_effort=reasoning_effort,
            model_digest=model_digest,
            expected_launcher_digest=expected_launcher_digest,
            cloud_policy_cache_binding=cloud_policy_cache_binding,
            model_catalog_digest=model_catalog_digest,
            no_tools_attestation_digest=no_tools_attestation_digest,
            runtime_identity_digest=runtime_digest,
            support_panel_ids=support_panel_ids,
            presentation=presentation,
            payload=frozen_payload,
            receipt=receipt,
        )
    values = {
        "source_digest": source_digest,
        "soft_cue_grammar_digest": object_bongard_soft_cue_grammar_digest(),
        "panel_atom_text_grammar_digest": panel_soft_atom_text_grammar_digest(),
        "protocol_digest": protocol_digest,
        "transport_source_digest": transport_digest,
        "prompt_digest": prompt_digest,
        "output_schema_digest": schema_digest,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "model_digest": model_digest,
        "expected_launcher_digest": expected_launcher_digest,
        "cloud_policy_cache_binding": cloud_policy_cache_binding,
        "model_catalog_digest": model_catalog_digest,
        "no_tools_attestation_digest": no_tools_attestation_digest,
        "runtime_identity_digest": runtime_digest,
        "support_panel_ids": support_panel_ids,
        "presentation": presentation,
        "logical_proposer_attempt_count": 1,
        "transport_invocation_count": transport_invocation_count,
        "receipted_call_count": 0 if receipt is None else 1,
        "status": status,
        "model_payload": frozen_payload,
        "receipt": receipt,
        "raw_proposer_evidence_digest": evidence_digest,
        "drops": drops,
        "vocabulary": vocabulary,
        "failure_code": failure_code,
        "failure_type": failure_type,
    }
    provisional = object.__new__(PanelSoftProposerArtifact)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return PanelSoftProposerArtifact(
        **values, artifact_digest=canonical_digest(_artifact_content(provisional))
    )


def _validated_support_panel_ids(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or len(value) != 12:
        raise PanelSoftProposerError("exactly twelve support panel IDs are required")
    row = tuple(value)
    matches = tuple(_SUPPORT_PANEL_ID.fullmatch(item) if isinstance(item, str) else None for item in row)
    if any(item is None for item in matches) or len(set(row)) != 12:
        raise PanelSoftProposerError("support panel IDs differ from the canonical layout")
    assert all(item is not None for item in matches)
    families = {item.group("family") for item in matches}
    tasks = {item.group("task") for item in matches}
    actual_slots = tuple((item.group("side"), item.group("index")) for item in matches)
    first_physical_side = actual_slots[0][0]
    second_physical_side = actual_slots[6][0]
    if (
        len(families) != 1
        or len(tasks) != 1
        or next(iter(families)) not in {"bd", "ff", "hd"}
        or not next(iter(tasks)).startswith(next(iter(families)) + "_")
        or first_physical_side == second_physical_side
        or tuple(side for side, _ in actual_slots[:6]) != (first_physical_side,) * 6
        or tuple(side for side, _ in actual_slots[6:]) != (second_physical_side,) * 6
        or tuple(int(index) for _, index in actual_slots[:6])
        != tuple(sorted(int(index) for _, index in actual_slots[:6]))
        or tuple(int(index) for _, index in actual_slots[6:])
        != tuple(sorted(int(index) for _, index in actual_slots[6:]))
    ):
        raise PanelSoftProposerError(
            "support IDs must be one task with six sorted unique IDs in each distinct physical-side block"
        )
    return row


def _validated_supports(
    support_pngs: Sequence[bytes], expected_support_sha256: Sequence[str]
) -> tuple[tuple[str, bytes], ...]:
    if (
        isinstance(support_pngs, (str, bytes))
        or isinstance(expected_support_sha256, (str, bytes))
        or len(support_pngs) != 12
        or len(expected_support_sha256) != 12
    ):
        raise PanelSoftProposerError("exactly twelve support PNGs are required")
    result: list[tuple[str, bytes]] = []
    for name, value, expected in zip(
        PANEL_SOFT_PROPOSER_PRESENTATION_NAMES,
        support_pngs,
        expected_support_sha256,
        strict=True,
    ):
        panel = _scene_runtime._validate_exact_png(value, name)
        if hashlib.sha256(panel).hexdigest() != _digest(expected, f"{name} digest"):
            raise PanelSoftProposerError("support bytes differ from commitment")
        result.append((name, panel))
    return tuple(result)


def propose_panel_soft_atoms(
    support_pngs: Sequence[bytes],
    *,
    support_panel_ids: Sequence[str],
    expected_support_sha256: Sequence[str],
    model: str,
    reasoning_effort: str = "medium",
    minutes: int = 15,
    verbose: bool = False,
    executable: str = "codex",
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None = None,
    expected_launcher_digest: str,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
    transport=run_codex_named_images_structured,
) -> PanelSoftProposerArtifact:
    """Propose atoms from exactly twelve raw support panels in one call."""

    frozen_support_ids = _validated_support_panel_ids(support_panel_ids)
    presentation_bytes = _validated_supports(support_pngs, expected_support_sha256)
    if not callable(transport):
        raise TypeError("transport must be callable")
    policy = _scene_runtime._policy_cache_binding(cloud_policy_cache_snapshot)
    model_catalog_digest, no_tools_digest = _scene_runtime._validate_no_tools_runtime(
        model_catalog_snapshot=model_catalog_snapshot,
        no_tools_attestation=no_tools_attestation,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_binding=policy,
    )
    prompt = panel_soft_proposer_prompt()
    schema = panel_soft_proposer_output_schema()
    presentation = _scene_runtime._image_identities(presentation_bytes)
    _scene_runtime._assert_model_visible_boundary(
        prompt,
        schema,
        PANEL_SOFT_PROPOSER_PRESENTATION_NAMES,
        hidden_values=(*frozen_support_ids, *(item.content_digest for item in presentation)),
    )
    transport_invocation_count = 0

    def counted_transport(*args: Any, **kwargs: Any):
        nonlocal transport_invocation_count
        transport_invocation_count += 1
        if transport_invocation_count != 1:
            raise PanelSoftProposerError("named-image transport was invoked more than once")
        return transport(*args, **kwargs)

    try:
        payload, receipt = _scene_runtime._stage_and_call(
            presentation_bytes,
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
            transport=counted_transport,
        )
    except Exception as exc:
        return _seal_artifact(
            model=model, reasoning_effort=reasoning_effort,
            expected_launcher_digest=expected_launcher_digest,
            cloud_policy_cache_binding=policy,
            model_catalog_digest=model_catalog_digest,
            no_tools_attestation_digest=no_tools_digest,
            support_panel_ids=frozen_support_ids,
            presentation=presentation,
            transport_invocation_count=transport_invocation_count,
            status=PanelSoftProposerStatus.TRANSPORT_ERROR,
            payload=None, receipt=None, drops=(), vocabulary=None,
            failure_code="proposer_transport_failed",
            failure_type=_scene_runtime._exception_type(exc),
        )
    evidence_digest = _raw_evidence_digest(
        source_digest=panel_soft_proposer_source_digest(),
        protocol_digest=panel_soft_proposer_protocol_digest(),
        transport_source_digest=_scene_runtime.prototype_scene_transport_source_digest(),
        prompt_digest=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        output_schema_digest=canonical_digest(schema),
        model=model,
        reasoning_effort=reasoning_effort,
        model_digest=_model_digest(model, reasoning_effort),
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_binding=policy,
        model_catalog_digest=model_catalog_digest,
        no_tools_attestation_digest=no_tools_digest,
        runtime_identity_digest=_runtime_identity_digest(
            model=model, reasoning_effort=reasoning_effort,
            expected_launcher_digest=expected_launcher_digest,
            cloud_policy_cache_binding=policy,
            model_catalog_digest=model_catalog_digest,
            no_tools_attestation_digest=no_tools_digest,
        ),
        support_panel_ids=frozen_support_ids,
        presentation=presentation,
        payload=payload,
        receipt=receipt,
    )
    try:
        vocabulary, drops = _parse_payload(payload, evidence_digest)
    except PanelSoftProposerError as exc:
        return _seal_artifact(
            model=model, reasoning_effort=reasoning_effort,
            expected_launcher_digest=expected_launcher_digest,
            cloud_policy_cache_binding=policy,
            model_catalog_digest=model_catalog_digest,
            no_tools_attestation_digest=no_tools_digest,
            support_panel_ids=frozen_support_ids,
            presentation=presentation,
            transport_invocation_count=transport_invocation_count,
            status=PanelSoftProposerStatus.PARSER_ERROR,
            payload=payload, receipt=receipt, drops=(), vocabulary=None,
            failure_code="proposer_payload_rejected",
            failure_type=_scene_runtime._exception_type(exc),
        )
    if vocabulary is None:
        return _seal_artifact(
            model=model, reasoning_effort=reasoning_effort,
            expected_launcher_digest=expected_launcher_digest,
            cloud_policy_cache_binding=policy,
            model_catalog_digest=model_catalog_digest,
            no_tools_attestation_digest=no_tools_digest,
            support_panel_ids=frozen_support_ids,
            presentation=presentation,
            transport_invocation_count=transport_invocation_count,
            status=PanelSoftProposerStatus.PARSER_ERROR,
            payload=payload, receipt=receipt, drops=drops, vocabulary=None,
            failure_code="proposer_orientation_empty",
            failure_type="PanelSoftProposerSemanticError",
        )
    return _seal_artifact(
        model=model, reasoning_effort=reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_binding=policy,
        model_catalog_digest=model_catalog_digest,
        no_tools_attestation_digest=no_tools_digest,
        support_panel_ids=frozen_support_ids,
        presentation=presentation,
        transport_invocation_count=transport_invocation_count,
        status=PanelSoftProposerStatus.SUCCESS,
        payload=payload, receipt=receipt, drops=drops, vocabulary=vocabulary,
        failure_code=None, failure_type=None,
    )


def verify_panel_soft_proposer_artifact(
    artifact: PanelSoftProposerArtifact,
    support_pngs: Sequence[bytes],
    *,
    support_panel_ids: Sequence[str],
    expected_artifact_digest: str,
    expected_runtime_identity_digest: str | None = None,
) -> PanelSoftProposerArtifact:
    """Cold-verify exact support pixels, payload, receipt, and parse projection."""

    if not isinstance(artifact, PanelSoftProposerArtifact):
        raise TypeError("artifact must be PanelSoftProposerArtifact")
    restored = PanelSoftProposerArtifact.from_data(artifact.to_data())
    if restored.artifact_digest != _digest(expected_artifact_digest, "expected artifact digest"):
        raise PanelSoftProposerError("artifact differs from commitment")
    if expected_runtime_identity_digest is not None and restored.runtime_identity_digest != _digest(expected_runtime_identity_digest, "expected runtime digest"):
        raise PanelSoftProposerError("runtime differs from commitment")
    expected = tuple(item.content_digest for item in restored.presentation)
    if _validated_support_panel_ids(support_panel_ids) != restored.support_panel_ids:
        raise PanelSoftProposerError("cold replay support panel IDs differ")
    presentation = _validated_supports(support_pngs, expected)
    identities = _scene_runtime._image_identities(presentation)
    if identities != restored.presentation:
        raise PanelSoftProposerError("cold replay support inputs differ")
    if restored.receipt is not None:
        assert restored.model_payload is not None
        with tempfile.TemporaryDirectory(prefix="bongard-panel-soft-proposer-replay-") as raw:
            directory = Path(raw)
            paths: list[str] = []
            for name, data in presentation:
                target = directory / name
                target.write_bytes(data)
                paths.append(str(target.resolve()))
            validate_codex_named_image_receipt(
                restored.receipt,
                panel_soft_proposer_prompt(),
                tuple(paths),
                PANEL_SOFT_PROPOSER_PRESENTATION_NAMES,
                panel_soft_proposer_output_schema(),
                dict(restored.model_payload),
            )
            if any(Path(path).read_bytes() != data for path, (_, data) in zip(paths, presentation, strict=True)):
                raise PanelSoftProposerError("cold replay support panel changed")
    return restored


__all__ = (
    "PANEL_SOFT_PROPOSER_PRESENTATION_NAMES",
    "PANEL_SOFT_PROPOSER_PROTOCOL_ID",
    "PanelSoftProposerArtifact",
    "PanelSoftProposerDrop",
    "PanelSoftProposerDropCode",
    "PanelSoftProposerError",
    "PanelSoftProposerStatus",
    "panel_soft_proposer_output_schema",
    "panel_soft_proposer_prompt",
    "panel_soft_proposer_protocol_digest",
    "panel_soft_proposer_source_digest",
    "propose_panel_soft_atoms",
    "verify_panel_soft_proposer_artifact",
)
