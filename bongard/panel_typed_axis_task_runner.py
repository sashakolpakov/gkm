"""Minimal positive-only task runner for the typed-axis v5 inventory.

The runner accepts an already-frozen support inventory and one mandatory
headless proposer attempt.  Proposer nominations and prose are archived but
never enter candidate derivation or ranking.  Zero survivors produce a typed
gap, one survivor freezes without a rank call, and multiple survivors make one
receipted text-only rank call over opaque aliases and equality formula wires.

The selected Python equality singleton/pair is persisted in the existing
fsynced write-once store before query evaluation.  Query cells are evaluated
with the same four dispositions as support cells: match predicts positive,
certified nonmatch predicts negative, indeterminate abstains, and error remains
error.  There is no negative formula, Not, OR, polarity operation, or Lean.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from typing import Any, Callable, Mapping, Sequence, TypeAlias

from bongard.canonical import canonical_digest, canonical_json
from bongard.evidence import Disposition
from bongard.object_bongard_release_gate import (
    ObjectBongardReleaseStore,
    ObjectBongardWriteOnceReceipt,
)
from bongard.object_bongard_turn_journal import ObjectBongardTurnRuntime
from bongard.panel_typed_axis_headless_proposer import (
    HeadlessTypedAxisAttemptErrorArtifact,
    HeadlessTypedAxisProposerArtifact,
    HeadlessTypedAxisProposerResult,
    headless_typed_axis_attempt_binding,
)
from bongard.panel_typed_axis_slate_v2 import (
    AXES,
    AXIS_DOMAINS,
    MAX_FORMULA_COUNT,
    Axis,
    EqualityAtom,
    EvidenceWitness,
    FormulaEvaluation,
    TypedAxisCell,
    TypedAxisInventory,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.transport import (
    CodexReceipt,
    CodexStructuredResult,
    run_codex_text_structured,
    validate_codex_receipt,
    validate_codex_strict_output_schema,
    validate_codex_text_receipt,
)


RUNNER_ID = "bongard.typed-axis/positive-formula-task-runner-python-v1"
RANK_INPUT_SCHEMA = "gkm.bongard-typed-axis-rank-input.v1"
RANK_ARTIFACT_SCHEMA = "gkm.bongard-typed-axis-rank-artifact.v1"
TASK_GAP_SCHEMA = "gkm.bongard-typed-axis-task-gap.v1"
FORMULA_FREEZE_SCHEMA = "gkm.bongard-typed-axis-formula-freeze.v1"
FORMULA_COMMIT_SCHEMA = "gkm.bongard-typed-axis-formula-commit.v1"
QUERY_EVIDENCE_SCHEMA = "gkm.bongard-typed-axis-query-evidence.v1"
QUERY_DECISION_SCHEMA = "gkm.bongard-typed-axis-query-decision.v1"
MAX_RANK_CANDIDATES = MAX_FORMULA_COUNT
MAX_RANK_PROMPT_BYTES = 512_000

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RAW_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:-]{0,255}\Z")


class TypedAxisTaskRunnerError(RuntimeError):
    """A mandatory attempt, rank, freeze, commit, or query differs."""


class TypedAxisQueryDisposition(str, Enum):
    MATCH = "match"
    NONMATCH = "nonmatch"
    INDETERMINATE = "indeterminate"
    ERROR = "error"


class TypedAxisQueryOutcome(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    ABSTAIN = "abstain"
    ERROR = "error"


TextTransport = Callable[..., CodexStructuredResult]


def panel_typed_axis_task_runner_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected
    ):
        raise TypedAxisTaskRunnerError(f"{label} fields differ")
    return value


def _address(value: object, label: str) -> str:
    if type(value) is not str or _ADDRESS.fullmatch(value) is None:
        raise TypedAxisTaskRunnerError(f"{label} must be a sha256: address")
    return value


def _raw_digest(value: object, label: str) -> str:
    if type(value) is not str or _RAW_DIGEST.fullmatch(value) is None:
        raise TypedAxisTaskRunnerError(f"{label} must be a raw SHA-256")
    return value


def _key(value: object, label: str) -> str:
    if type(value) is not str or _KEY.fullmatch(value) is None:
        raise TypedAxisTaskRunnerError(f"{label} is invalid")
    return value


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "implementation_language": "python",
        "python_is_canonical_authority": True,
        "positive_formula_count": 1,
        "negative_formula_present": False,
        "not_operator_present": False,
        "or_operator_present": False,
        "polarity_operation_present": False,
        "lean_present": False,
        "lean_required": False,
        "query_truth_label_present": False,
    }


def _canonical_inventory(value: object) -> TypedAxisInventory:
    if type(value) is not TypedAxisInventory:
        raise TypeError("runner needs exact TypedAxisInventory")
    # The exact frozen class constructor already recomputes all 1,366 formulas
    # from its matrix and accepts no nomination field.  Repeating its 30 MB JSON
    # round-trip at every runner boundary adds no invariant and is prohibitive
    # for a headless benchmark loop.  A single equality replay keeps this
    # boundary nomination-free while retaining the constructor proof.
    independently_derived = TypedAxisInventory.derive(value.matrix)
    if independently_derived != value:
        raise TypedAxisTaskRunnerError(
            "inventory differs from nomination-free deterministic derivation"
        )
    return value


def _formula_by_id(
    inventory: TypedAxisInventory, formula_id: str
) -> FormulaEvaluation:
    matches = tuple(item for item in inventory.formulas if item.formula_id == formula_id)
    if len(matches) != 1:
        raise TypedAxisTaskRunnerError("formula id is absent from closed inventory")
    return matches[0]


def _wire_for_formula(formula: FormulaEvaluation) -> dict[str, object]:
    if type(formula) is not FormulaEvaluation:
        raise TypeError("formula wire needs exact FormulaEvaluation")
    return {
        "operator": "all_of",
        "atoms": [
            {"axis": atom.axis.value, "equals": atom.value}
            for atom in formula.atoms
        ],
    }


def _canonical_wire(value: object) -> dict[str, object]:
    raw = _fields(value, {"operator", "atoms"}, "typed equality formula wire")
    atoms = raw["atoms"]
    if raw["operator"] != "all_of" or type(atoms) is not list or not 1 <= len(atoms) <= 2:
        raise TypedAxisTaskRunnerError("formula wire operator or arity differs")
    restored: list[EqualityAtom] = []
    for item in atoms:
        atom = _fields(item, {"axis", "equals"}, "typed equality atom wire")
        try:
            axis = Axis(atom["axis"])
            equality = EqualityAtom(axis, atom["equals"])
        except (TypeError, ValueError) as exc:
            raise TypedAxisTaskRunnerError("typed equality atom differs") from exc
        restored.append(equality)
    if (
        len({item.axis for item in restored}) != len(restored)
        or tuple(sorted(restored, key=lambda item: AXES.index(item.axis)))
        != tuple(restored)
    ):
        raise TypedAxisTaskRunnerError("typed equality atom order differs")
    result = {
        "operator": "all_of",
        "atoms": [
            {"axis": item.axis.value, "equals": item.value} for item in restored
        ],
    }
    if canonical_json(result) != canonical_json(dict(raw)):
        raise TypedAxisTaskRunnerError("formula wire is not canonical")
    return result


def _version_space_content(inventory: TypedAxisInventory) -> dict[str, object]:
    return {
        "schema": "gkm.bongard-typed-axis-positive-version-space.v1",
        "inventory_address": inventory.inventory_address,
        "survivors": [
            {
                "formula_id": formula_id,
                "formula_wire": _wire_for_formula(_formula_by_id(inventory, formula_id)),
            }
            for formula_id in inventory.admitted_formula_ids
        ],
        "inventory_derived_without_nominations": True,
    }


def _version_space_digest(inventory: TypedAxisInventory) -> str:
    return "sha256:" + canonical_digest(_version_space_content(inventory))


def _canonical_attempt(
    value: object,
    *,
    inventory: TypedAxisInventory,
    expected_attempt_digest: str,
) -> HeadlessTypedAxisProposerResult:
    expected = _raw_digest(expected_attempt_digest, "expected headless attempt digest")
    if type(value) is HeadlessTypedAxisProposerArtifact:
        restored: HeadlessTypedAxisProposerResult = (
            HeadlessTypedAxisProposerArtifact.from_data(value.to_data())
        )
    elif type(value) is HeadlessTypedAxisAttemptErrorArtifact:
        restored = HeadlessTypedAxisAttemptErrorArtifact.from_data(value.to_data())
    else:
        raise TypeError("runner requires one exact headless success-or-error attempt")
    binding = headless_typed_axis_attempt_binding(restored)
    if (
        restored.attempt_digest != expected
        or binding["attempt_digest"] != expected
        or binding["support_matrix_address"] != inventory.matrix.matrix_address
        or binding["runner_must_bind_attempt"] is not True
        or binding["omission_or_reroll_allowed"] is not False
        or binding["error_is_axis_gap_or_negative_evidence"] is not False
    ):
        raise TypedAxisTaskRunnerError("mandatory headless attempt binding differs")
    return restored


def _attempt_kind(value: HeadlessTypedAxisProposerResult) -> str:
    if type(value) is HeadlessTypedAxisProposerArtifact:
        return "success"
    if type(value) is HeadlessTypedAxisAttemptErrorArtifact:
        return "error"
    raise TypeError("headless attempt union differs")


def _attempt_from_data(kind: object, value: object) -> HeadlessTypedAxisProposerResult:
    if kind == "success":
        return HeadlessTypedAxisProposerArtifact.from_data(value)
    if kind == "error":
        return HeadlessTypedAxisAttemptErrorArtifact.from_data(value)
    raise TypedAxisTaskRunnerError("headless attempt kind differs")


def _receipt_from_data(value: object) -> CodexReceipt:
    if not isinstance(value, Mapping):
        raise TypedAxisTaskRunnerError("rank receipt must be an object")
    raw = dict(value)
    try:
        validate_codex_receipt(raw)
        if type(raw["event_types"]) is not list or type(raw["item_types"]) is not list:
            raise TypedAxisTaskRunnerError("rank receipt sequence fields differ")
        receipt = CodexReceipt(
            **{
                **raw,
                "event_types": tuple(raw["event_types"]),
                "item_types": tuple(raw["item_types"]),
            }
        )
    except Exception as exc:
        if isinstance(exc, TypedAxisTaskRunnerError):
            raise
        raise TypedAxisTaskRunnerError("rank receipt is invalid") from exc
    if receipt.to_dict() != raw:
        raise TypedAxisTaskRunnerError("rank receipt is not canonical")
    return receipt


@dataclass(frozen=True, slots=True)
class TypedAxisRankInput:
    inventory_address: str
    version_space_digest: str
    formula_ids: tuple[str, ...]
    formula_wires: tuple[Mapping[str, Any], ...]
    aliases: tuple[str, ...]
    record_digest: str

    def __post_init__(self) -> None:
        _address(self.inventory_address, "rank inventory address")
        _address(self.version_space_digest, "rank version-space digest")
        if (
            type(self.formula_ids) is not tuple
            or not 2 <= len(self.formula_ids) <= MAX_RANK_CANDIDATES
            or len(set(self.formula_ids)) != len(self.formula_ids)
            or any(_KEY.fullmatch(item) is None for item in self.formula_ids)
            or type(self.formula_wires) is not tuple
            or len(self.formula_wires) != len(self.formula_ids)
            or type(self.aliases) is not tuple
            or self.aliases
            != tuple(f"candidate_{index:04d}" for index in range(len(self.formula_ids)))
        ):
            raise TypedAxisTaskRunnerError("typed rank input shape differs")
        wires = tuple(_canonical_wire(item) for item in self.formula_wires)
        if tuple(dict(item) for item in self.formula_wires) != wires:
            raise TypedAxisTaskRunnerError("typed rank input wires differ")
        _address(self.record_digest, "rank input digest")
        if self.record_digest != "sha256:" + canonical_digest(self.content_data()):
            raise TypedAxisTaskRunnerError("typed rank input digest differs")

    @classmethod
    def from_inventory(cls, inventory: TypedAxisInventory) -> "TypedAxisRankInput":
        frozen = _canonical_inventory(inventory)
        ids = frozen.admitted_formula_ids
        wires = tuple(_wire_for_formula(_formula_by_id(frozen, item)) for item in ids)
        values = {
            "inventory_address": frozen.inventory_address,
            "version_space_digest": _version_space_digest(frozen),
            "formula_ids": ids,
            "formula_wires": wires,
            "aliases": tuple(f"candidate_{index:04d}" for index in range(len(ids))),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            record_digest="sha256:" + canonical_digest(provisional.content_data()),
        )

    def content_data(self) -> dict[str, object]:
        return {
            "schema": RANK_INPUT_SCHEMA,
            "runner_id": RUNNER_ID,
            "inventory_address": self.inventory_address,
            "version_space_digest": self.version_space_digest,
            "formula_ids": list(self.formula_ids),
            "formula_wires": [dict(item) for item in self.formula_wires],
            "aliases": list(self.aliases),
            "candidate_order": "inventory_admitted_formula_order",
            "model_visible_fields": ["opaque_alias", "formula_wire"],
            "proposer_material_model_visible": False,
            "panel_role_side_query_material_model_visible": False,
            **_authority_data(),
        }

    def visible_data(self) -> list[dict[str, object]]:
        return [
            {"opaque_alias": alias, "formula_wire": dict(wire)}
            for alias, wire in zip(self.aliases, self.formula_wires, strict=True)
        ]

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "TypedAxisRankInput":
        expected = {
            "schema", "runner_id", "inventory_address", "version_space_digest",
            "formula_ids", "formula_wires", "aliases", "candidate_order",
            "model_visible_fields", "proposer_material_model_visible",
            "panel_role_side_query_material_model_visible", *_authority_data(),
            "record_digest",
        }
        raw = _fields(value, expected, "typed rank input")
        if (
            raw["schema"] != RANK_INPUT_SCHEMA
            or raw["runner_id"] != RUNNER_ID
            or raw["candidate_order"] != "inventory_admitted_formula_order"
            or raw["model_visible_fields"] != ["opaque_alias", "formula_wire"]
            or raw["proposer_material_model_visible"] is not False
            or raw["panel_role_side_query_material_model_visible"] is not False
            or any(raw[name] != item for name, item in _authority_data().items())
            or any(type(raw[name]) is not list for name in ("formula_ids", "formula_wires", "aliases"))
        ):
            raise TypedAxisTaskRunnerError("typed rank input policy differs")
        result = cls(
            raw["inventory_address"], raw["version_space_digest"],
            tuple(raw["formula_ids"]), tuple(_canonical_wire(item) for item in raw["formula_wires"]),
            tuple(raw["aliases"]), raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise TypedAxisTaskRunnerError("typed rank input is not canonical")
        return result


def typed_axis_rank_prompt(rank_input: TypedAxisRankInput) -> str:
    frozen = TypedAxisRankInput.from_data(rank_input.to_data())
    visible = canonical_json(frozen.visible_data()).decode("utf-8")
    prompt = (
        "Rank every verified positive equality-conjunction candidate from most "
        "coherent, salient, reusable, and concise to least. Return exactly one "
        "complete permutation of the opaque aliases and invent nothing.\n"
        + visible
    )
    if len(prompt.encode("utf-8")) > MAX_RANK_PROMPT_BYTES:
        raise TypedAxisTaskRunnerError("typed rank prompt exceeds byte guard")
    return prompt


def typed_axis_rank_output_schema(rank_input: TypedAxisRankInput) -> dict[str, object]:
    frozen = TypedAxisRankInput.from_data(rank_input.to_data())
    schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "ordered_aliases": {
                "type": "array",
                "items": {"type": "string", "enum": list(frozen.aliases)},
            }
        },
        "required": ["ordered_aliases"],
        "additionalProperties": False,
    }
    validate_codex_strict_output_schema(schema)
    return schema


def _ordered_aliases(value: object, rank_input: TypedAxisRankInput) -> tuple[str, ...]:
    raw = _fields(value, {"ordered_aliases"}, "typed rank payload")
    aliases = raw["ordered_aliases"]
    if type(aliases) is not list or any(type(item) is not str for item in aliases):
        raise TypedAxisTaskRunnerError("rank aliases must be a string list")
    result = tuple(aliases)
    if len(result) != len(rank_input.aliases) or set(result) != set(rank_input.aliases) or len(set(result)) != len(result):
        raise TypedAxisTaskRunnerError("rank payload must be the exact full permutation")
    return result


def _rank_artifact_content(value: "TypedAxisRankArtifact") -> dict[str, object]:
    return {
        "schema": RANK_ARTIFACT_SCHEMA,
        "runner_id": RUNNER_ID,
        "rank_input": value.rank_input.to_data(),
        "rank_input_digest": value.rank_input.record_digest,
        "model_payload": dict(value.model_payload),
        "ordered_formula_ids": list(value.ordered_formula_ids),
        "selected_formula_id": value.selected_formula_id,
        "receipt": value.receipt.to_dict(),
        "receipt_digest": value.receipt.receipt_digest,
        "runtime_binding": dict(value.runtime_binding),
        "transport_kind": value.transport_kind,
        "rank_transport_invocations": 1,
        "rank_receipt_authenticated": True,
        "benchmark_sealable": False,
        "durable_exactly_once_journal_embedded": False,
        "full_permutation_required": True,
        "model_visible_fields": ["opaque_alias", "formula_wire"],
        "proposer_material_model_visible": False,
        "panel_role_side_query_material_model_visible": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class TypedAxisRankArtifact:
    rank_input: TypedAxisRankInput
    model_payload: Mapping[str, Any]
    ordered_formula_ids: tuple[str, ...]
    selected_formula_id: str
    receipt: CodexReceipt
    runtime_binding: Mapping[str, Any]
    transport_kind: str
    record_digest: str

    def __post_init__(self) -> None:
        if type(self.rank_input) is not TypedAxisRankInput:
            raise TypeError("rank artifact needs exact TypedAxisRankInput")
        rank_input = TypedAxisRankInput.from_data(self.rank_input.to_data())
        if not isinstance(self.model_payload, Mapping):
            raise TypeError("rank payload must be a mapping")
        payload = json.loads(canonical_json(dict(self.model_payload)).decode("utf-8"))
        aliases = _ordered_aliases(payload, rank_input)
        by_alias = dict(zip(rank_input.aliases, rank_input.formula_ids, strict=True))
        ordered = tuple(by_alias[item] for item in aliases)
        if (
            ordered != self.ordered_formula_ids
            or self.selected_formula_id != ordered[0]
            or type(self.receipt) is not CodexReceipt
            or not isinstance(self.runtime_binding, Mapping)
            or self.transport_kind not in {"production_direct", "injected_unverified"}
        ):
            raise TypedAxisTaskRunnerError("rank artifact selection fields differ")
        prompt = typed_axis_rank_prompt(rank_input)
        schema = typed_axis_rank_output_schema(rank_input)
        try:
            validate_codex_text_receipt(self.receipt.to_dict(), prompt, schema)
        except Exception as exc:
            raise TypedAxisTaskRunnerError("rank receipt input binding differs") from exc
        runtime = dict(self.runtime_binding)
        if (
            self.receipt.structured_output_digest != canonical_digest(payload)
            or runtime.get("model") != self.receipt.requested_model
            or runtime.get("reasoning_effort") != self.receipt.requested_reasoning_effort
            or runtime.get("expected_launcher_digest") != self.receipt.codex_launcher_digest
            or runtime.get("cloud_policy_cache_binding")
            != self.receipt.cloud_config_bundle_cache_binding
            or runtime.get("model_catalog_raw_digest") != self.receipt.model_catalog_digest
            or runtime.get("no_tools_attestation_digest")
            != self.receipt.tool_surface_attestation_digest
        ):
            raise TypedAxisTaskRunnerError("rank receipt runtime or payload differs")
        _address(self.record_digest, "rank artifact digest")
        if self.record_digest != "sha256:" + canonical_digest(_rank_artifact_content(self)):
            raise TypedAxisTaskRunnerError("rank artifact digest differs")

    @classmethod
    def create(
        cls,
        rank_input: TypedAxisRankInput,
        result: CodexStructuredResult,
        runtime: ObjectBongardTurnRuntime,
        *,
        transport_kind: str,
    ) -> "TypedAxisRankArtifact":
        if type(result) is not CodexStructuredResult or type(result.receipt) is not CodexReceipt:
            raise TypedAxisTaskRunnerError("rank transport returned no full receipt")
        runtime.validate_receipt(result.receipt)
        payload = json.loads(canonical_json(dict(result.payload)).decode("utf-8"))
        aliases = _ordered_aliases(payload, rank_input)
        by_alias = dict(zip(rank_input.aliases, rank_input.formula_ids, strict=True))
        ordered = tuple(by_alias[item] for item in aliases)
        values = {
            "rank_input": rank_input,
            "model_payload": payload,
            "ordered_formula_ids": ordered,
            "selected_formula_id": ordered[0],
            "receipt": result.receipt,
            "runtime_binding": dict(runtime.binding),
            "transport_kind": transport_kind,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            record_digest="sha256:" + canonical_digest(_rank_artifact_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_rank_artifact_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "TypedAxisRankArtifact":
        expected = {
            "schema", "runner_id", "rank_input", "rank_input_digest",
            "model_payload", "ordered_formula_ids", "selected_formula_id",
            "receipt", "receipt_digest", "runtime_binding", "transport_kind",
            "rank_transport_invocations", "rank_receipt_authenticated",
            "benchmark_sealable", "durable_exactly_once_journal_embedded",
            "full_permutation_required", "model_visible_fields",
            "proposer_material_model_visible",
            "panel_role_side_query_material_model_visible", *_authority_data(),
            "record_digest",
        }
        raw = _fields(value, expected, "typed rank artifact")
        if (
            raw["schema"] != RANK_ARTIFACT_SCHEMA
            or raw["runner_id"] != RUNNER_ID
            or raw["rank_transport_invocations"] != 1
            or raw["rank_receipt_authenticated"] is not True
            or raw["benchmark_sealable"] is not False
            or raw["durable_exactly_once_journal_embedded"] is not False
            or raw["full_permutation_required"] is not True
            or raw["model_visible_fields"] != ["opaque_alias", "formula_wire"]
            or raw["proposer_material_model_visible"] is not False
            or raw["panel_role_side_query_material_model_visible"] is not False
            or any(raw[name] != item for name, item in _authority_data().items())
            or type(raw["ordered_formula_ids"]) is not list
        ):
            raise TypedAxisTaskRunnerError("typed rank artifact policy differs")
        rank_input = TypedAxisRankInput.from_data(raw["rank_input"])
        receipt = _receipt_from_data(raw["receipt"])
        result = cls(
            rank_input, dict(raw["model_payload"]), tuple(raw["ordered_formula_ids"]),
            raw["selected_formula_id"], receipt, dict(raw["runtime_binding"]),
            raw["transport_kind"], raw["record_digest"],
        )
        if (
            raw["rank_input_digest"] != rank_input.record_digest
            or raw["receipt_digest"] != receipt.receipt_digest
            or result.to_data() != dict(raw)
        ):
            raise TypedAxisTaskRunnerError("typed rank artifact is not canonical")
        return result


def _gap_content(value: "TypedAxisTaskGap") -> dict[str, object]:
    return {
        "schema": TASK_GAP_SCHEMA,
        "runner_id": RUNNER_ID,
        "kind": "no_positive_formula_survivor",
        "inventory_address": value.inventory_address,
        "support_matrix_address": value.support_matrix_address,
        "version_space_digest": value.version_space_digest,
        "survivor_count": 0,
        "headless_attempt_kind": value.headless_attempt_kind,
        "headless_attempt": value.headless_attempt.to_data(),
        "headless_attempt_digest": value.headless_attempt.attempt_digest,
        "headless_attempt_mandatory": True,
        "headless_attempt_omission_or_reroll_allowed": False,
        "inventory_derived_without_nominations": True,
        "rank_transport_invocations": 0,
        "query_release_authorized": False,
        "typed_gap_not_exception": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class TypedAxisTaskGap:
    inventory_address: str
    support_matrix_address: str
    version_space_digest: str
    headless_attempt_kind: str
    headless_attempt: HeadlessTypedAxisProposerResult
    record_digest: str

    def __post_init__(self) -> None:
        _address(self.inventory_address, "gap inventory address")
        _address(self.support_matrix_address, "gap matrix address")
        _address(self.version_space_digest, "gap version-space digest")
        attempt = _attempt_from_data(self.headless_attempt_kind, self.headless_attempt.to_data())
        if attempt.request.support_matrix_address != self.support_matrix_address:
            raise TypedAxisTaskRunnerError("gap attempt matrix differs")
        _address(self.record_digest, "gap digest")
        if self.record_digest != "sha256:" + canonical_digest(_gap_content(self)):
            raise TypedAxisTaskRunnerError("typed task gap digest differs")

    @classmethod
    def create(
        cls, inventory: TypedAxisInventory, attempt: HeadlessTypedAxisProposerResult
    ) -> "TypedAxisTaskGap":
        values = {
            "inventory_address": inventory.inventory_address,
            "support_matrix_address": inventory.matrix.matrix_address,
            "version_space_digest": _version_space_digest(inventory),
            "headless_attempt_kind": _attempt_kind(attempt),
            "headless_attempt": attempt,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            record_digest="sha256:" + canonical_digest(_gap_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_gap_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "TypedAxisTaskGap":
        expected = {
            "schema", "runner_id", "kind", "inventory_address",
            "support_matrix_address", "version_space_digest", "survivor_count",
            "headless_attempt_kind", "headless_attempt", "headless_attempt_digest",
            "headless_attempt_mandatory", "headless_attempt_omission_or_reroll_allowed",
            "inventory_derived_without_nominations", "rank_transport_invocations",
            "query_release_authorized", "typed_gap_not_exception", *_authority_data(),
            "record_digest",
        }
        raw = _fields(value, expected, "typed task gap")
        if (
            raw["schema"] != TASK_GAP_SCHEMA or raw["runner_id"] != RUNNER_ID
            or raw["kind"] != "no_positive_formula_survivor" or raw["survivor_count"] != 0
            or raw["headless_attempt_mandatory"] is not True
            or raw["headless_attempt_omission_or_reroll_allowed"] is not False
            or raw["inventory_derived_without_nominations"] is not True
            or raw["rank_transport_invocations"] != 0
            or raw["query_release_authorized"] is not False
            or raw["typed_gap_not_exception"] is not True
            or any(raw[name] != item for name, item in _authority_data().items())
        ):
            raise TypedAxisTaskRunnerError("typed task gap policy differs")
        attempt = _attempt_from_data(raw["headless_attempt_kind"], raw["headless_attempt"])
        result = cls(
            raw["inventory_address"], raw["support_matrix_address"],
            raw["version_space_digest"], raw["headless_attempt_kind"], attempt,
            raw["record_digest"],
        )
        if raw["headless_attempt_digest"] != attempt.attempt_digest or result.to_data() != dict(raw):
            raise TypedAxisTaskRunnerError("typed task gap is not canonical")
        return result


def _freeze_content(value: "TypedAxisFormulaFreeze") -> dict[str, object]:
    return {
        "schema": FORMULA_FREEZE_SCHEMA,
        "runner_id": RUNNER_ID,
        "runner_source_digest": panel_typed_axis_task_runner_source_digest(),
        "inventory_address": value.inventory_address,
        "support_matrix_address": value.support_matrix_address,
        "version_space_digest": value.version_space_digest,
        "survivor_formula_ids": list(value.survivor_formula_ids),
        "survivor_count": len(value.survivor_formula_ids),
        "selected_formula_id": value.selected_formula_id,
        "selected_formula_wire": dict(value.selected_formula_wire),
        "selected_predicate_digest": value.selected_predicate_digest,
        "selection_mode": value.selection_mode,
        "rank_artifact": None if value.rank_artifact is None else value.rank_artifact.to_data(),
        "rank_artifact_digest": None if value.rank_artifact is None else value.rank_artifact.record_digest,
        "rank_transport_invocations": 0 if value.rank_artifact is None else 1,
        "headless_attempt_kind": value.headless_attempt_kind,
        "headless_attempt": value.headless_attempt.to_data(),
        "headless_attempt_digest": value.headless_attempt.attempt_digest,
        "headless_attempt_mandatory": True,
        "headless_attempt_omission_or_reroll_allowed": False,
        "headless_nominations_or_prose_enter_selection": False,
        "inventory_derived_without_nominations": True,
        "query_material_seen": False,
        "frozen_before_query": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class TypedAxisFormulaFreeze:
    inventory_address: str
    support_matrix_address: str
    version_space_digest: str
    survivor_formula_ids: tuple[str, ...]
    selected_formula_id: str
    selected_formula_wire: Mapping[str, Any]
    selected_predicate_digest: str
    selection_mode: str
    rank_artifact: TypedAxisRankArtifact | None
    headless_attempt_kind: str
    headless_attempt: HeadlessTypedAxisProposerResult
    record_digest: str

    def __post_init__(self) -> None:
        _address(self.inventory_address, "freeze inventory address")
        _address(self.support_matrix_address, "freeze matrix address")
        _address(self.version_space_digest, "freeze version-space digest")
        if (
            type(self.survivor_formula_ids) is not tuple
            or not self.survivor_formula_ids
            or len(set(self.survivor_formula_ids)) != len(self.survivor_formula_ids)
            or self.selected_formula_id not in self.survivor_formula_ids
        ):
            raise TypedAxisTaskRunnerError("freeze survivor ids differ")
        wire = _canonical_wire(self.selected_formula_wire)
        if dict(self.selected_formula_wire) != wire:
            raise TypedAxisTaskRunnerError("freeze selected wire differs")
        expected_predicate = "sha256:" + canonical_digest(
            {"schema": "gkm.bongard-frozen-python-equality-predicate.v1", "wire": wire}
        )
        if self.selected_predicate_digest != expected_predicate:
            raise TypedAxisTaskRunnerError("freeze predicate digest differs")
        attempt = _attempt_from_data(self.headless_attempt_kind, self.headless_attempt.to_data())
        if attempt.request.support_matrix_address != self.support_matrix_address:
            raise TypedAxisTaskRunnerError("freeze attempt matrix differs")
        if len(self.survivor_formula_ids) == 1:
            if (
                self.selection_mode != "unique_survivor_zero_rank_call"
                or self.rank_artifact is not None
                or self.selected_formula_id != self.survivor_formula_ids[0]
            ):
                raise TypedAxisTaskRunnerError("unique freeze selection differs")
        else:
            if type(self.rank_artifact) is not TypedAxisRankArtifact:
                raise TypedAxisTaskRunnerError("multi-survivor freeze needs rank artifact")
            rank = TypedAxisRankArtifact.from_data(self.rank_artifact.to_data())
            if (
                self.selection_mode != "one_receipted_headless_rank"
                or rank.rank_input.inventory_address != self.inventory_address
                or rank.rank_input.version_space_digest != self.version_space_digest
                or rank.rank_input.formula_ids != self.survivor_formula_ids
                or rank.selected_formula_id != self.selected_formula_id
                or dict(rank.rank_input.formula_wires[
                    rank.rank_input.formula_ids.index(self.selected_formula_id)
                ]) != wire
            ):
                raise TypedAxisTaskRunnerError("multi-survivor freeze rank differs")
        _address(self.record_digest, "formula freeze digest")
        if self.record_digest != "sha256:" + canonical_digest(_freeze_content(self)):
            raise TypedAxisTaskRunnerError("formula freeze digest differs")

    @classmethod
    def create(
        cls,
        inventory: TypedAxisInventory,
        attempt: HeadlessTypedAxisProposerResult,
        *,
        rank_artifact: TypedAxisRankArtifact | None,
    ) -> "TypedAxisFormulaFreeze":
        ids = inventory.admitted_formula_ids
        if not ids:
            raise TypedAxisTaskRunnerError("cannot freeze an empty version space")
        if len(ids) == 1:
            selected = ids[0]
            mode = "unique_survivor_zero_rank_call"
        else:
            if type(rank_artifact) is not TypedAxisRankArtifact:
                raise TypedAxisTaskRunnerError("multiple survivors require exact rank artifact")
            selected = rank_artifact.selected_formula_id
            mode = "one_receipted_headless_rank"
        wire = _wire_for_formula(_formula_by_id(inventory, selected))
        values = {
            "inventory_address": inventory.inventory_address,
            "support_matrix_address": inventory.matrix.matrix_address,
            "version_space_digest": _version_space_digest(inventory),
            "survivor_formula_ids": ids,
            "selected_formula_id": selected,
            "selected_formula_wire": wire,
            "selected_predicate_digest": "sha256:" + canonical_digest(
                {"schema": "gkm.bongard-frozen-python-equality-predicate.v1", "wire": wire}
            ),
            "selection_mode": mode,
            "rank_artifact": rank_artifact,
            "headless_attempt_kind": _attempt_kind(attempt),
            "headless_attempt": attempt,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            record_digest="sha256:" + canonical_digest(_freeze_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_freeze_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "TypedAxisFormulaFreeze":
        expected = {
            "schema", "runner_id", "runner_source_digest", "inventory_address",
            "support_matrix_address", "version_space_digest", "survivor_formula_ids",
            "survivor_count", "selected_formula_id", "selected_formula_wire",
            "selected_predicate_digest", "selection_mode", "rank_artifact",
            "rank_artifact_digest", "rank_transport_invocations",
            "headless_attempt_kind", "headless_attempt", "headless_attempt_digest",
            "headless_attempt_mandatory", "headless_attempt_omission_or_reroll_allowed",
            "headless_nominations_or_prose_enter_selection",
            "inventory_derived_without_nominations", "query_material_seen",
            "frozen_before_query", *_authority_data(), "record_digest",
        }
        raw = _fields(value, expected, "typed formula freeze")
        if (
            raw["schema"] != FORMULA_FREEZE_SCHEMA or raw["runner_id"] != RUNNER_ID
            or raw["runner_source_digest"] != panel_typed_axis_task_runner_source_digest()
            or raw["survivor_count"] != len(raw["survivor_formula_ids"])
            or raw["headless_attempt_mandatory"] is not True
            or raw["headless_attempt_omission_or_reroll_allowed"] is not False
            or raw["headless_nominations_or_prose_enter_selection"] is not False
            or raw["inventory_derived_without_nominations"] is not True
            or raw["query_material_seen"] is not False
            or raw["frozen_before_query"] is not True
            or any(raw[name] != item for name, item in _authority_data().items())
            or type(raw["survivor_formula_ids"]) is not list
        ):
            raise TypedAxisTaskRunnerError("typed formula freeze policy differs")
        attempt = _attempt_from_data(raw["headless_attempt_kind"], raw["headless_attempt"])
        rank = None if raw["rank_artifact"] is None else TypedAxisRankArtifact.from_data(raw["rank_artifact"])
        result = cls(
            raw["inventory_address"], raw["support_matrix_address"], raw["version_space_digest"],
            tuple(raw["survivor_formula_ids"]), raw["selected_formula_id"],
            _canonical_wire(raw["selected_formula_wire"]), raw["selected_predicate_digest"],
            raw["selection_mode"], rank, raw["headless_attempt_kind"], attempt,
            raw["record_digest"],
        )
        if (
            raw["rank_artifact_digest"] != (None if rank is None else rank.record_digest)
            or raw["rank_transport_invocations"] != (0 if rank is None else 1)
            or raw["headless_attempt_digest"] != attempt.attempt_digest
            or result.to_data() != dict(raw)
        ):
            raise TypedAxisTaskRunnerError("typed formula freeze is not canonical")
        return result


TypedAxisTaskResult: TypeAlias = TypedAxisTaskGap | TypedAxisFormulaFreeze


def run_typed_axis_formula_task(
    inventory: TypedAxisInventory,
    headless_attempt: HeadlessTypedAxisProposerResult,
    *,
    expected_headless_attempt_digest: str,
    rank_runtime: ObjectBongardTurnRuntime | None = None,
    rank_transport: TextTransport | None = None,
) -> TypedAxisTaskResult:
    """Derive, optionally rank once, and freeze without seeing query material."""

    frozen = _canonical_inventory(inventory)
    attempt = _canonical_attempt(
        headless_attempt,
        inventory=frozen,
        expected_attempt_digest=expected_headless_attempt_digest,
    )
    if not frozen.admitted_formula_ids:
        return TypedAxisTaskGap.create(frozen, attempt)
    rank_artifact: TypedAxisRankArtifact | None = None
    if len(frozen.admitted_formula_ids) > 1:
        if type(rank_runtime) is not ObjectBongardTurnRuntime or not callable(rank_transport):
            raise TypedAxisTaskRunnerError(
                "multiple survivors require one exact runtime and rank transport"
            )
        rank_input = TypedAxisRankInput.from_inventory(frozen)
        prompt = typed_axis_rank_prompt(rank_input)
        schema = typed_axis_rank_output_schema(rank_input)
        transport_kind = (
            "production_direct"
            if rank_transport is run_codex_text_structured
            else "injected_unverified"
        )
        try:
            result = rank_transport(
                prompt,
                schema,
                model=rank_runtime.model,
                reasoning_effort=rank_runtime.reasoning_effort,
                minutes=rank_runtime.minutes,
                verbose=rank_runtime.verbose,
                executable=rank_runtime.executable,
                cloud_policy_cache_snapshot=rank_runtime.cloud_policy_cache_snapshot,
                model_catalog_snapshot=rank_runtime.model_catalog_snapshot,
                tool_surface_attestation=rank_runtime.no_tools_attestation,
                expected_launcher_digest=rank_runtime.expected_launcher_digest,
                expected_tool_surface_attestation_digest=(
                    rank_runtime.no_tools_attestation.attestation_digest
                ),
            )
        except Exception as exc:
            raise TypedAxisTaskRunnerError(
                "the single rank transport failed; reroll is forbidden"
            ) from exc
        rank_artifact = TypedAxisRankArtifact.create(
            rank_input, result, rank_runtime, transport_kind=transport_kind
        )
    return TypedAxisFormulaFreeze.create(
        frozen, attempt, rank_artifact=rank_artifact
    )


def cold_replay_typed_axis_task_result(
    result: TypedAxisTaskResult,
    *,
    inventory: TypedAxisInventory,
    headless_attempt: HeadlessTypedAxisProposerResult,
    expected_artifact_address: str,
) -> TypedAxisTaskResult:
    """Rebuild the gap/freeze from archived evidence with zero model calls."""

    expected = _address(expected_artifact_address, "expected task artifact address")
    frozen = _canonical_inventory(inventory)
    attempt = _canonical_attempt(
        headless_attempt,
        inventory=frozen,
        expected_attempt_digest=headless_attempt.attempt_digest,
    )
    if type(result) is TypedAxisTaskGap:
        restored: TypedAxisTaskResult = TypedAxisTaskGap.from_data(result.to_data())
        rebuilt = TypedAxisTaskGap.create(frozen, attempt)
    elif type(result) is TypedAxisFormulaFreeze:
        restored = TypedAxisFormulaFreeze.from_data(result.to_data())
        rank = restored.rank_artifact
        if rank is not None:
            expected_input = TypedAxisRankInput.from_inventory(frozen)
            if rank.rank_input != expected_input:
                raise TypedAxisTaskRunnerError("cold rank input differs from inventory")
        rebuilt = TypedAxisFormulaFreeze.create(frozen, attempt, rank_artifact=rank)
    else:
        raise TypeError("cold replay needs exact typed task result union")
    if restored != result or rebuilt != restored or restored.record_digest != expected:
        raise TypedAxisTaskRunnerError("typed task result cold replay differs")
    return restored


def _commit_content(value: "TypedAxisFormulaCommit") -> dict[str, object]:
    return {
        "schema": FORMULA_COMMIT_SCHEMA,
        "runner_id": RUNNER_ID,
        "formula_freeze": value.formula_freeze.to_data(),
        "formula_freeze_digest": value.formula_freeze.record_digest,
        "formula_freeze_store_receipt": value.formula_freeze_store_receipt.to_data(),
        "formula_freeze_store_receipt_digest": value.formula_freeze_store_receipt.record_digest,
        "selected_predicate_digest": value.formula_freeze.selected_predicate_digest,
        "durably_persisted_and_reloaded_before_query": True,
        "exact_canonical_freeze_bytes_bound": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class TypedAxisFormulaCommit:
    formula_freeze: TypedAxisFormulaFreeze
    formula_freeze_store_receipt: ObjectBongardWriteOnceReceipt
    record_digest: str

    def __post_init__(self) -> None:
        if type(self.formula_freeze) is not TypedAxisFormulaFreeze:
            raise TypeError("commit needs exact TypedAxisFormulaFreeze")
        if type(self.formula_freeze_store_receipt) is not ObjectBongardWriteOnceReceipt:
            raise TypeError("commit needs exact write-once receipt")
        freeze = TypedAxisFormulaFreeze.from_data(self.formula_freeze.to_data())
        receipt = ObjectBongardWriteOnceReceipt.from_data(
            self.formula_freeze_store_receipt.to_data()
        )
        payload = canonical_json(freeze.to_data()) + b"\n"
        if (
            receipt.object_kind != "typed-axis-formula-freeze"
            or receipt.object_digest != freeze.record_digest
            or receipt.payload_digest != "sha256:" + hashlib.sha256(payload).hexdigest()
            or receipt.size_bytes != len(payload)
        ):
            raise TypedAxisTaskRunnerError("freeze receipt does not bind exact bytes")
        _address(self.record_digest, "formula commit digest")
        if self.record_digest != "sha256:" + canonical_digest(_commit_content(self)):
            raise TypedAxisTaskRunnerError("formula commit digest differs")

    @classmethod
    def seal(
        cls,
        freeze: TypedAxisFormulaFreeze,
        receipt: ObjectBongardWriteOnceReceipt,
    ) -> "TypedAxisFormulaCommit":
        values = {"formula_freeze": freeze, "formula_freeze_store_receipt": receipt}
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            record_digest="sha256:" + canonical_digest(_commit_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_commit_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "TypedAxisFormulaCommit":
        expected = {
            "schema", "runner_id", "formula_freeze", "formula_freeze_digest",
            "formula_freeze_store_receipt", "formula_freeze_store_receipt_digest",
            "selected_predicate_digest", "durably_persisted_and_reloaded_before_query",
            "exact_canonical_freeze_bytes_bound", *_authority_data(), "record_digest",
        }
        raw = _fields(value, expected, "typed formula commit")
        if (
            raw["schema"] != FORMULA_COMMIT_SCHEMA or raw["runner_id"] != RUNNER_ID
            or raw["durably_persisted_and_reloaded_before_query"] is not True
            or raw["exact_canonical_freeze_bytes_bound"] is not True
            or any(raw[name] != item for name, item in _authority_data().items())
        ):
            raise TypedAxisTaskRunnerError("typed formula commit policy differs")
        freeze = TypedAxisFormulaFreeze.from_data(raw["formula_freeze"])
        receipt = ObjectBongardWriteOnceReceipt.from_data(raw["formula_freeze_store_receipt"])
        result = cls(freeze, receipt, raw["record_digest"])
        if (
            raw["formula_freeze_digest"] != freeze.record_digest
            or raw["formula_freeze_store_receipt_digest"] != receipt.record_digest
            or raw["selected_predicate_digest"] != freeze.selected_predicate_digest
            or result.to_data() != dict(raw)
        ):
            raise TypedAxisTaskRunnerError("typed formula commit is not canonical")
        return result


def persist_typed_axis_formula_commit(
    store: ObjectBongardReleaseStore,
    freeze: TypedAxisFormulaFreeze,
) -> tuple[TypedAxisFormulaCommit, ObjectBongardWriteOnceReceipt]:
    """Persist/reload freeze and commit records before returning query authority."""

    if type(store) is not ObjectBongardReleaseStore:
        raise TypeError("formula persistence needs exact ObjectBongardReleaseStore")
    frozen = TypedAxisFormulaFreeze.from_data(freeze.to_data())
    freeze_receipt = store.persist(
        object_kind="typed-axis-formula-freeze",
        object_digest=frozen.record_digest,
        data=frozen.to_data(),
    )
    commit = TypedAxisFormulaCommit.seal(frozen, freeze_receipt)
    commit_receipt = store.persist(
        object_kind="typed-axis-formula-commit",
        object_digest=commit.record_digest,
        data=commit.to_data(),
    )
    return commit, commit_receipt


def verify_typed_axis_formula_commit(
    commit: TypedAxisFormulaCommit,
    commit_receipt: ObjectBongardWriteOnceReceipt,
    *,
    store: ObjectBongardReleaseStore,
    inventory: TypedAxisInventory,
    headless_attempt: HeadlessTypedAxisProposerResult,
) -> TypedAxisFormulaCommit:
    """Verify both durable objects and replay the freeze without model calls."""

    if type(commit) is not TypedAxisFormulaCommit:
        raise TypeError("commit verification needs exact TypedAxisFormulaCommit")
    if type(commit_receipt) is not ObjectBongardWriteOnceReceipt:
        raise TypeError("commit verification needs exact write-once receipt")
    restored = TypedAxisFormulaCommit.from_data(commit.to_data())
    store.verify(
        restored.formula_freeze_store_receipt,
        expected_data=restored.formula_freeze.to_data(),
    )
    store.verify(commit_receipt, expected_data=restored.to_data())
    if (
        commit_receipt.object_kind != "typed-axis-formula-commit"
        or commit_receipt.object_digest != restored.record_digest
    ):
        raise TypedAxisTaskRunnerError("commit receipt does not bind exact commit")
    cold_replay_typed_axis_task_result(
        restored.formula_freeze,
        inventory=inventory,
        headless_attempt=headless_attempt,
        expected_artifact_address=restored.formula_freeze.record_digest,
    )
    return restored


def _query_evidence_content(value: "TypedAxisQueryEvidence") -> dict[str, object]:
    return {
        "schema": QUERY_EVIDENCE_SCHEMA,
        "query_id": value.query_id,
        "query_panel_sha256": value.query_panel_sha256,
        "query_release_custody_address": value.query_release_custody_address,
        "observer_artifact_address": value.observer_artifact_address,
        "observer_protocol_digest": value.observer_protocol_digest,
        "formula_commit_address": value.formula_commit_address,
        "support_inventory_address": value.support_inventory_address,
        "cells": [item.to_data() for item in value.cells],
        "axis_order": [axis.value for axis in AXES],
        "query_truth_label_present": False,
        "panel_bytes_verified_inside_runner": False,
        "external_query_release_and_observer_custody_required": True,
    }


@dataclass(frozen=True, slots=True)
class TypedAxisQueryEvidence:
    query_id: str
    query_panel_sha256: str
    query_release_custody_address: str
    observer_artifact_address: str
    observer_protocol_digest: str
    formula_commit_address: str
    support_inventory_address: str
    cells: tuple[TypedAxisCell, ...]
    record_digest: str

    def __post_init__(self) -> None:
        _key(self.query_id, "query id")
        _raw_digest(self.query_panel_sha256, "query panel digest")
        for label, value in (
            ("query release custody", self.query_release_custody_address),
            ("observer artifact", self.observer_artifact_address),
            ("observer protocol", self.observer_protocol_digest),
            ("formula commit", self.formula_commit_address),
            ("support inventory", self.support_inventory_address),
        ):
            _address(value, label)
        if (
            type(self.cells) is not tuple
            or len(self.cells) != len(AXES)
            or any(type(item) is not TypedAxisCell for item in self.cells)
            or tuple(item.axis for item in self.cells) != AXES
            or any(item.observer_protocol_digest != self.observer_protocol_digest for item in self.cells)
        ):
            raise TypedAxisTaskRunnerError("query cells or protocol differ")
        _address(self.record_digest, "query evidence digest")
        if self.record_digest != "sha256:" + canonical_digest(_query_evidence_content(self)):
            raise TypedAxisTaskRunnerError("query evidence digest differs")

    @classmethod
    def create(
        cls,
        *,
        query_id: str,
        query_panel_sha256: str,
        query_release_custody_address: str,
        observer_artifact_address: str,
        formula_commit: TypedAxisFormulaCommit,
        cells: Sequence[TypedAxisCell],
    ) -> "TypedAxisQueryEvidence":
        frozen_cells = tuple(cells)
        if not frozen_cells:
            raise TypedAxisTaskRunnerError("query cells are empty")
        values = {
            "query_id": query_id,
            "query_panel_sha256": query_panel_sha256,
            "query_release_custody_address": query_release_custody_address,
            "observer_artifact_address": observer_artifact_address,
            "observer_protocol_digest": frozen_cells[0].observer_protocol_digest,
            "formula_commit_address": formula_commit.record_digest,
            "support_inventory_address": formula_commit.formula_freeze.inventory_address,
            "cells": frozen_cells,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            record_digest="sha256:" + canonical_digest(_query_evidence_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_query_evidence_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "TypedAxisQueryEvidence":
        expected = {
            "schema", "query_id", "query_panel_sha256", "query_release_custody_address",
            "observer_artifact_address", "observer_protocol_digest", "formula_commit_address",
            "support_inventory_address", "cells", "axis_order", "query_truth_label_present",
            "panel_bytes_verified_inside_runner",
            "external_query_release_and_observer_custody_required", "record_digest",
        }
        raw = _fields(value, expected, "typed query evidence")
        if (
            raw["schema"] != QUERY_EVIDENCE_SCHEMA
            or raw["axis_order"] != [axis.value for axis in AXES]
            or raw["query_truth_label_present"] is not False
            or raw["panel_bytes_verified_inside_runner"] is not False
            or raw["external_query_release_and_observer_custody_required"] is not True
            or type(raw["cells"]) is not list
        ):
            raise TypedAxisTaskRunnerError("typed query evidence policy differs")
        result = cls(
            raw["query_id"], raw["query_panel_sha256"], raw["query_release_custody_address"],
            raw["observer_artifact_address"], raw["observer_protocol_digest"],
            raw["formula_commit_address"], raw["support_inventory_address"],
            tuple(TypedAxisCell.from_data(item) for item in raw["cells"]),
            raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise TypedAxisTaskRunnerError("typed query evidence is not canonical")
        return result


def _query_evaluation(
    wire: Mapping[str, Any], evidence: TypedAxisQueryEvidence
) -> tuple[tuple[EvidenceWitness, ...], TypedAxisQueryDisposition, TypedAxisQueryOutcome]:
    canonical = _canonical_wire(wire)
    atoms = tuple(
        EqualityAtom(Axis(item["axis"]), item["equals"])
        for item in canonical["atoms"]
    )
    by_axis = {item.axis: item for item in evidence.cells}
    witnesses = tuple(
        EvidenceWitness.evaluate(by_axis[atom.axis], atom.value) for atom in atoms
    )
    states = tuple(item.disposition for item in witnesses)
    if Disposition.ERROR in states:
        disposition = TypedAxisQueryDisposition.ERROR
        outcome = TypedAxisQueryOutcome.ERROR
    elif Disposition.CERTIFIED_ABSENT in states:
        disposition = TypedAxisQueryDisposition.NONMATCH
        outcome = TypedAxisQueryOutcome.NEGATIVE
    elif all(item is Disposition.PRESENT for item in states):
        disposition = TypedAxisQueryDisposition.MATCH
        outcome = TypedAxisQueryOutcome.POSITIVE
    else:
        disposition = TypedAxisQueryDisposition.INDETERMINATE
        outcome = TypedAxisQueryOutcome.ABSTAIN
    return witnesses, disposition, outcome


def _query_decision_content(value: "TypedAxisQueryDecision") -> dict[str, object]:
    witnesses, disposition, outcome = _query_evaluation(
        value.selected_formula_wire, value.query_evidence
    )
    return {
        "schema": QUERY_DECISION_SCHEMA,
        "runner_id": RUNNER_ID,
        "formula_commit_address": value.formula_commit_address,
        "formula_commit_store_receipt_digest": value.formula_commit_store_receipt_digest,
        "inventory_address": value.inventory_address,
        "selected_formula_id": value.selected_formula_id,
        "selected_formula_wire": dict(value.selected_formula_wire),
        "selected_predicate_digest": value.selected_predicate_digest,
        "query_evidence": value.query_evidence.to_data(),
        "query_evidence_digest": value.query_evidence.record_digest,
        "atom_witnesses": [item.to_data() for item in witnesses],
        "formula_disposition": disposition.value,
        "outcome": outcome.value,
        "decision_rule": "match-positive_nonmatch-negative_indeterminate-abstain_error-error",
        "negative_formula_evaluated": False,
        "model_calls_during_query_evaluation": 0,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class TypedAxisQueryDecision:
    formula_commit_address: str
    formula_commit_store_receipt_digest: str
    inventory_address: str
    selected_formula_id: str
    selected_formula_wire: Mapping[str, Any]
    selected_predicate_digest: str
    query_evidence: TypedAxisQueryEvidence
    formula_disposition: TypedAxisQueryDisposition
    outcome: TypedAxisQueryOutcome
    record_digest: str

    def __post_init__(self) -> None:
        _address(self.formula_commit_address, "query formula commit")
        _address(self.formula_commit_store_receipt_digest, "query commit receipt")
        _address(self.inventory_address, "query inventory")
        _key(self.selected_formula_id, "selected formula id")
        wire = _canonical_wire(self.selected_formula_wire)
        if dict(self.selected_formula_wire) != wire:
            raise TypedAxisTaskRunnerError("query selected formula wire differs")
        _address(self.selected_predicate_digest, "query selected predicate")
        if type(self.query_evidence) is not TypedAxisQueryEvidence:
            raise TypeError("decision needs exact TypedAxisQueryEvidence")
        evidence = TypedAxisQueryEvidence.from_data(self.query_evidence.to_data())
        _witnesses, disposition, outcome = _query_evaluation(wire, evidence)
        if (
            evidence.formula_commit_address != self.formula_commit_address
            or evidence.support_inventory_address != self.inventory_address
            or self.formula_disposition is not disposition
            or self.outcome is not outcome
        ):
            raise TypedAxisTaskRunnerError("query decision binding or mapping differs")
        _address(self.record_digest, "query decision digest")
        if self.record_digest != "sha256:" + canonical_digest(_query_decision_content(self)):
            raise TypedAxisTaskRunnerError("query decision digest differs")

    @classmethod
    def create(
        cls,
        commit: TypedAxisFormulaCommit,
        commit_receipt: ObjectBongardWriteOnceReceipt,
        evidence: TypedAxisQueryEvidence,
    ) -> "TypedAxisQueryDecision":
        freeze = commit.formula_freeze
        witnesses, disposition, outcome = _query_evaluation(
            freeze.selected_formula_wire, evidence
        )
        del witnesses
        values = {
            "formula_commit_address": commit.record_digest,
            "formula_commit_store_receipt_digest": commit_receipt.record_digest,
            "inventory_address": freeze.inventory_address,
            "selected_formula_id": freeze.selected_formula_id,
            "selected_formula_wire": dict(freeze.selected_formula_wire),
            "selected_predicate_digest": freeze.selected_predicate_digest,
            "query_evidence": evidence,
            "formula_disposition": disposition,
            "outcome": outcome,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            record_digest="sha256:" + canonical_digest(_query_decision_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_query_decision_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "TypedAxisQueryDecision":
        expected = {
            "schema", "runner_id", "formula_commit_address",
            "formula_commit_store_receipt_digest", "inventory_address",
            "selected_formula_id", "selected_formula_wire", "selected_predicate_digest",
            "query_evidence", "query_evidence_digest", "atom_witnesses",
            "formula_disposition", "outcome", "decision_rule",
            "negative_formula_evaluated", "model_calls_during_query_evaluation",
            *_authority_data(), "record_digest",
        }
        raw = _fields(value, expected, "typed query decision")
        if (
            raw["schema"] != QUERY_DECISION_SCHEMA or raw["runner_id"] != RUNNER_ID
            or raw["decision_rule"]
            != "match-positive_nonmatch-negative_indeterminate-abstain_error-error"
            or raw["negative_formula_evaluated"] is not False
            or raw["model_calls_during_query_evaluation"] != 0
            or any(raw[name] != item for name, item in _authority_data().items())
            or type(raw["atom_witnesses"]) is not list
        ):
            raise TypedAxisTaskRunnerError("typed query decision policy differs")
        evidence = TypedAxisQueryEvidence.from_data(raw["query_evidence"])
        result = cls(
            raw["formula_commit_address"], raw["formula_commit_store_receipt_digest"],
            raw["inventory_address"], raw["selected_formula_id"],
            _canonical_wire(raw["selected_formula_wire"]), raw["selected_predicate_digest"],
            evidence, TypedAxisQueryDisposition(raw["formula_disposition"]),
            TypedAxisQueryOutcome(raw["outcome"]), raw["record_digest"],
        )
        if (
            raw["query_evidence_digest"] != evidence.record_digest
            or raw["atom_witnesses"]
            != _query_decision_content(result)["atom_witnesses"]
            or result.to_data() != dict(raw)
        ):
            raise TypedAxisTaskRunnerError("typed query decision is not canonical")
        return result


def evaluate_typed_axis_query(
    commit: TypedAxisFormulaCommit,
    commit_receipt: ObjectBongardWriteOnceReceipt,
    *,
    store: ObjectBongardReleaseStore,
    inventory: TypedAxisInventory,
    headless_attempt: HeadlessTypedAxisProposerResult,
    query_evidence: TypedAxisQueryEvidence,
) -> TypedAxisQueryDecision:
    """Verify durable freeze/commit custody, then execute one Python predicate."""

    verified = verify_typed_axis_formula_commit(
        commit,
        commit_receipt,
        store=store,
        inventory=inventory,
        headless_attempt=headless_attempt,
    )
    evidence = TypedAxisQueryEvidence.from_data(query_evidence.to_data())
    if (
        evidence.formula_commit_address != verified.record_digest
        or evidence.support_inventory_address
        != verified.formula_freeze.inventory_address
        or evidence.observer_protocol_digest
        != inventory.matrix.observer_protocol_digest
    ):
        raise TypedAxisTaskRunnerError("query evidence commit, inventory, or protocol differs")
    return TypedAxisQueryDecision.create(verified, commit_receipt, evidence)


def cold_replay_typed_axis_query_decision(
    decision: TypedAxisQueryDecision,
    *,
    commit: TypedAxisFormulaCommit,
    commit_receipt: ObjectBongardWriteOnceReceipt,
    store: ObjectBongardReleaseStore,
    inventory: TypedAxisInventory,
    headless_attempt: HeadlessTypedAxisProposerResult,
    expected_artifact_address: str,
) -> TypedAxisQueryDecision:
    """Re-evaluate archived query cells with zero model calls."""

    if type(decision) is not TypedAxisQueryDecision:
        raise TypeError("query replay needs exact TypedAxisQueryDecision")
    expected = _address(expected_artifact_address, "expected query decision address")
    restored = TypedAxisQueryDecision.from_data(decision.to_data())
    replayed = evaluate_typed_axis_query(
        commit,
        commit_receipt,
        store=store,
        inventory=inventory,
        headless_attempt=headless_attempt,
        query_evidence=restored.query_evidence,
    )
    if replayed != restored or restored.record_digest != expected:
        raise TypedAxisTaskRunnerError("typed query decision cold replay differs")
    return restored


__all__ = (
    "MAX_RANK_CANDIDATES",
    "RUNNER_ID",
    "TextTransport",
    "TypedAxisFormulaCommit",
    "TypedAxisFormulaFreeze",
    "TypedAxisQueryDecision",
    "TypedAxisQueryDisposition",
    "TypedAxisQueryEvidence",
    "TypedAxisQueryOutcome",
    "TypedAxisRankArtifact",
    "TypedAxisRankInput",
    "TypedAxisTaskGap",
    "TypedAxisTaskResult",
    "TypedAxisTaskRunnerError",
    "cold_replay_typed_axis_query_decision",
    "cold_replay_typed_axis_task_result",
    "evaluate_typed_axis_query",
    "panel_typed_axis_task_runner_source_digest",
    "persist_typed_axis_formula_commit",
    "run_typed_axis_formula_task",
    "typed_axis_rank_output_schema",
    "typed_axis_rank_prompt",
    "verify_typed_axis_formula_commit",
)
