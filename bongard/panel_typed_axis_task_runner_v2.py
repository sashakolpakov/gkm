"""No-narrator symbolic runner for task-bound typed-axis support.

Ordering is exact: derive the fixed version space first; zero survivors return
a typed gap with zero model calls; one survivor freezes with zero model calls;
and more than one survivor requires one exactly-once text journal turn.  This
engineering successor is deliberately unsealable until its support custody is
backed by an independently authenticated inference artifact.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from bongard.canonical import canonical_digest, canonical_json
from bongard.object_bongard_turn_journal import (
    ObjectBongardTextTurnJournalTransport,
    ObjectBongardTurnJournalSummary,
    ObjectBongardTurnRuntime,
)
from bongard.panel_typed_axis_custody_v2 import TaskBoundTypedAxisSupportArtifact
from bongard.panel_typed_axis_slate_v2 import (
    AXES,
    Axis,
    EqualityAtom,
    FormulaEvaluation,
    TypedAxisInventory,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.transport import (
    CodexReceipt,
    CodexStructuredResult,
    run_codex_text_structured,
    validate_codex_strict_output_schema,
    validate_codex_text_receipt,
)


RUNNER_ID = "bongard.typed-axis/task-bound-positive-formula-python-v2"
TASK_GAP_SCHEMA = "gkm.bongard-typed-axis-task-gap.v2"
RANK_INPUT_SCHEMA = "gkm.bongard-typed-axis-rank-input.v2"
RANK_ARTIFACT_SCHEMA = "gkm.bongard-typed-axis-rank-artifact.v2"
FORMULA_FREEZE_SCHEMA = "gkm.bongard-typed-axis-formula-freeze.v2"
RANK_TURN_KIND = "typed_axis_formula_rank_v2"

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:-]{0,255}\Z")


class TypedAxisTaskRunnerV2Error(RuntimeError):
    """The fixed version space, rank journal, or selected formula differs."""


def panel_typed_axis_task_runner_v2_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _address(value: object, label: str) -> str:
    if type(value) is not str or _ADDRESS.fullmatch(value) is None:
        raise TypedAxisTaskRunnerV2Error(f"{label} must be a sha256: address")
    return value


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise TypedAxisTaskRunnerV2Error(f"{label} fields differ")
    return value


def _authority() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "implementation_language": "python",
        "python_is_canonical_authority": True,
        "one_positive_formula_only": True,
        "negative_formula_present": False,
        "not_operator_present": False,
        "or_operator_present": False,
        "polarity_operation_present": False,
        "lean_present": False,
        "lean_required": False,
    }


def _inventory(support: TaskBoundTypedAxisSupportArtifact) -> TypedAxisInventory:
    if type(support) is not TaskBoundTypedAxisSupportArtifact:
        raise TypeError("v2 runner needs exact task-bound support custody")
    restored = TaskBoundTypedAxisSupportArtifact.from_data(support.to_data())
    if restored != support:
        raise TypedAxisTaskRunnerV2Error("support custody canonical replay differs")
    return TypedAxisInventory.derive(restored.matrix)


def _formula(inventory: TypedAxisInventory, formula_id: str) -> FormulaEvaluation:
    found = tuple(item for item in inventory.formulas if item.formula_id == formula_id)
    if len(found) != 1:
        raise TypedAxisTaskRunnerV2Error("formula is outside the fixed inventory")
    return found[0]


def _wire(formula: FormulaEvaluation) -> dict[str, object]:
    return {
        "operator": "all_of",
        "atoms": [
            {"axis": item.axis.value, "equals": item.value} for item in formula.atoms
        ],
    }


def _canonical_wire(value: object) -> dict[str, object]:
    raw = _fields(value, {"operator", "atoms"}, "formula wire")
    if raw["operator"] != "all_of" or type(raw["atoms"]) is not list:
        raise TypedAxisTaskRunnerV2Error("formula wire operator differs")
    atoms: list[EqualityAtom] = []
    for item in raw["atoms"]:
        atom = _fields(item, {"axis", "equals"}, "formula atom")
        atoms.append(EqualityAtom(Axis(atom["axis"]), atom["equals"]))
    if (
        not 1 <= len(atoms) <= 2
        or len({item.axis for item in atoms}) != len(atoms)
        or tuple(sorted(atoms, key=lambda item: AXES.index(item.axis))) != tuple(atoms)
    ):
        raise TypedAxisTaskRunnerV2Error("formula wire arity or order differs")
    result = {
        "operator": "all_of",
        "atoms": [{"axis": item.axis.value, "equals": item.value} for item in atoms],
    }
    if canonical_json(result) != canonical_json(dict(raw)):
        raise TypedAxisTaskRunnerV2Error("formula wire is not canonical")
    return result


def _version_space_digest(inventory: TypedAxisInventory) -> str:
    return "sha256:" + canonical_digest(
        {
            "schema": "gkm.bongard-typed-axis-positive-version-space.v2",
            "inventory_address": inventory.inventory_address,
            "survivors": [
                {"formula_id": item, "wire": _wire(_formula(inventory, item))}
                for item in inventory.admitted_formula_ids
            ],
            "derived_before_any_rank_call": True,
        }
    )


@dataclass(frozen=True, slots=True)
class TypedAxisRankInputV2:
    inventory_address: str
    version_space_digest: str
    formula_ids: tuple[str, ...]
    formula_wires: tuple[Mapping[str, Any], ...]
    aliases: tuple[str, ...]
    record_digest: str

    def __post_init__(self) -> None:
        _address(self.inventory_address, "rank inventory")
        _address(self.version_space_digest, "rank version space")
        if (
            type(self.formula_ids) is not tuple
            or len(self.formula_ids) <= 1
            or len(set(self.formula_ids)) != len(self.formula_ids)
            or any(_KEY.fullmatch(item) is None for item in self.formula_ids)
            or type(self.formula_wires) is not tuple
            or len(self.formula_wires) != len(self.formula_ids)
            or self.aliases
            != tuple(f"candidate_{index:04d}" for index in range(len(self.formula_ids)))
        ):
            raise TypedAxisTaskRunnerV2Error("rank input shape differs")
        if tuple(_canonical_wire(item) for item in self.formula_wires) != tuple(
            dict(item) for item in self.formula_wires
        ):
            raise TypedAxisTaskRunnerV2Error("rank wires differ")
        _address(self.record_digest, "rank input digest")
        if self.record_digest != "sha256:" + canonical_digest(self.content_data()):
            raise TypedAxisTaskRunnerV2Error("rank input digest differs")

    @classmethod
    def create(cls, inventory: TypedAxisInventory) -> "TypedAxisRankInputV2":
        ids = inventory.admitted_formula_ids
        values = {
            "inventory_address": inventory.inventory_address,
            "version_space_digest": _version_space_digest(inventory),
            "formula_ids": ids,
            "formula_wires": tuple(_wire(_formula(inventory, item)) for item in ids),
            "aliases": tuple(f"candidate_{index:04d}" for index in range(len(ids))),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, record_digest="sha256:" + canonical_digest(provisional.content_data()))

    def content_data(self) -> dict[str, object]:
        return {
            "schema": RANK_INPUT_SCHEMA,
            "inventory_address": self.inventory_address,
            "version_space_digest": self.version_space_digest,
            "formula_ids": list(self.formula_ids),
            "formula_wires": [dict(item) for item in self.formula_wires],
            "aliases": list(self.aliases),
            "model_visible_fields": ["opaque_alias", "formula_wire"],
            "support_pixels_or_roles_model_visible": False,
            "narrator_or_proposer_material_model_visible": False,
            **_authority(),
        }

    def visible_data(self) -> list[dict[str, object]]:
        return [
            {"opaque_alias": alias, "formula_wire": dict(wire)}
            for alias, wire in zip(self.aliases, self.formula_wires, strict=True)
        ]


def typed_axis_rank_prompt_v2(value: TypedAxisRankInputV2) -> str:
    return (
        "Rank every verified positive equality-conjunction candidate from most "
        "coherent, salient, reusable, and concise to least. Return exactly one "
        "complete permutation of the opaque aliases and invent nothing.\n"
        + canonical_json(value.visible_data()).decode("utf-8")
    )


def typed_axis_rank_schema_v2(value: TypedAxisRankInputV2) -> dict[str, object]:
    schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "ordered_aliases": {
                "type": "array",
                "items": {"type": "string", "enum": list(value.aliases)},
            }
        },
        "required": ["ordered_aliases"],
        "additionalProperties": False,
    }
    validate_codex_strict_output_schema(schema)
    return schema


def _summary_data(summary: ObjectBongardTurnJournalSummary) -> dict[str, object]:
    data = summary.to_data()
    content = {key: item for key, item in data.items() if key != "record_digest"}
    if data["record_digest"] != "sha256:" + canonical_digest(content):
        raise TypedAxisTaskRunnerV2Error("rank journal summary digest differs")
    return data


@dataclass(frozen=True, slots=True)
class TypedAxisRankArtifactV2:
    rank_input: TypedAxisRankInputV2
    model_payload: Mapping[str, Any]
    receipt: CodexReceipt
    ordered_formula_ids: tuple[str, ...]
    selected_formula_id: str
    journal_summary: ObjectBongardTurnJournalSummary
    record_digest: str

    def __post_init__(self) -> None:
        if type(self.rank_input) is not TypedAxisRankInputV2 or type(self.receipt) is not CodexReceipt:
            raise TypeError("rank artifact members differ")
        payload = json.loads(canonical_json(dict(self.model_payload)))
        aliases = payload.get("ordered_aliases")
        if (
            type(aliases) is not list
            or tuple(aliases) == ()
            or len(aliases) != len(self.rank_input.aliases)
            or set(aliases) != set(self.rank_input.aliases)
            or len(set(aliases)) != len(aliases)
        ):
            raise TypedAxisTaskRunnerV2Error("rank response is not a full permutation")
        by_alias = dict(zip(self.rank_input.aliases, self.rank_input.formula_ids, strict=True))
        ordered = tuple(by_alias[item] for item in aliases)
        try:
            validate_codex_text_receipt(
                self.receipt.to_dict(),
                typed_axis_rank_prompt_v2(self.rank_input),
                typed_axis_rank_schema_v2(self.rank_input),
            )
        except Exception as exc:
            raise TypedAxisTaskRunnerV2Error("rank receipt binding differs") from exc
        if (
            ordered != self.ordered_formula_ids
            or self.selected_formula_id != ordered[0]
            or self.receipt.structured_output_digest != canonical_digest(payload)
            or self.journal_summary.terminal_status != "success"
        ):
            raise TypedAxisTaskRunnerV2Error("rank selection or terminal differs")
        _summary_data(self.journal_summary)
        _address(self.record_digest, "rank artifact digest")
        if self.record_digest != "sha256:" + canonical_digest(self.content_data()):
            raise TypedAxisTaskRunnerV2Error("rank artifact digest differs")

    def content_data(self) -> dict[str, object]:
        return {
            "schema": RANK_ARTIFACT_SCHEMA,
            "rank_input": self.rank_input.content_data() | {"record_digest": self.rank_input.record_digest},
            "model_payload": dict(self.model_payload),
            "receipt": self.receipt.to_dict(),
            "ordered_formula_ids": list(self.ordered_formula_ids),
            "selected_formula_id": self.selected_formula_id,
            "journal_summary": _summary_data(self.journal_summary),
            "rank_transport_invocations": 1,
            "exactly_once_text_journal_required": True,
            "narrator_model_calls": 0,
            "benchmark_sealable": False,
            **_authority(),
        }


def build_typed_axis_rank_journal_v2(
    directory: str | Path,
    *,
    support: TaskBoundTypedAxisSupportArtifact,
    inventory: TypedAxisInventory,
    runtime: ObjectBongardTurnRuntime,
    underlying_transport=run_codex_text_structured,
) -> ObjectBongardTextTurnJournalTransport:
    rank_input = TypedAxisRankInputV2.create(inventory)
    return ObjectBongardTextTurnJournalTransport(
        directory,
        authorization_digest=support.release_authorization.record_digest,
        execution_precommit_digest=support.execution_precommit.record_digest,
        task_id=support.task_id,
        turn_kind=RANK_TURN_KIND,
        expected_prompt=typed_axis_rank_prompt_v2(rank_input),
        expected_output_schema=typed_axis_rank_schema_v2(rank_input),
        runtime=runtime,
        underlying_transport=underlying_transport,
    )


def _gap_content(value: "TypedAxisTaskGapV2") -> dict[str, object]:
    return {
        "schema": TASK_GAP_SCHEMA,
        "runner_id": RUNNER_ID,
        "runner_source_digest": panel_typed_axis_task_runner_v2_source_digest(),
        "support_custody_address": value.support.record_digest,
        "inventory_address": value.inventory_address,
        "version_space_digest": value.version_space_digest,
        "empty_gap": value.empty_gap,
        "survivor_count": 0,
        "rank_model_calls": 0,
        "narrator_model_calls": 0,
        "query_release_authorized": False,
        "benchmark_sealable": False,
        **_authority(),
    }


@dataclass(frozen=True, slots=True)
class TypedAxisTaskGapV2:
    support: TaskBoundTypedAxisSupportArtifact
    inventory_address: str
    version_space_digest: str
    empty_gap: Mapping[str, Any]
    record_digest: str

    def __post_init__(self) -> None:
        inventory = _inventory(self.support)
        if inventory.empty_gap is None or self.empty_gap != inventory.empty_gap.to_data():
            raise TypedAxisTaskRunnerV2Error("typed gap witness differs")
        if self.inventory_address != inventory.inventory_address or self.version_space_digest != _version_space_digest(inventory):
            raise TypedAxisTaskRunnerV2Error("typed gap inventory differs")
        if self.record_digest != "sha256:" + canonical_digest(_gap_content(self)):
            raise TypedAxisTaskRunnerV2Error("typed gap digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_gap_content(self), "record_digest": self.record_digest}


def _freeze_content(value: "TypedAxisFormulaFreezeV2") -> dict[str, object]:
    return {
        "schema": FORMULA_FREEZE_SCHEMA,
        "runner_id": RUNNER_ID,
        "runner_source_digest": panel_typed_axis_task_runner_v2_source_digest(),
        "support_custody": value.support.to_data(),
        "support_custody_address": value.support.record_digest,
        "task_id": value.support.task_id,
        "task_plan_digest": value.support.task_plan.record_digest,
        "execution_precommit_digest": value.support.execution_precommit.record_digest,
        "release_authorization_digest": value.support.release_authorization.record_digest,
        "inventory_address": value.inventory_address,
        "version_space_digest": value.version_space_digest,
        "survivor_formula_ids": list(value.survivor_formula_ids),
        "selected_formula_id": value.selected_formula_id,
        "selected_formula_wire": dict(value.selected_formula_wire),
        "selected_predicate_digest": value.selected_predicate_digest,
        "selection_mode": value.selection_mode,
        "rank_artifact": None if value.rank_artifact is None else value.rank_artifact.content_data() | {"record_digest": value.rank_artifact.record_digest},
        "rank_model_calls": 0 if value.rank_artifact is None else 1,
        "narrator_model_calls": 0,
        "version_space_derived_before_rank": True,
        "frozen_before_query": True,
        "observer_inference_externally_authenticated": False,
        "benchmark_sealable": False,
        "query_release_authorized": False,
        **_authority(),
    }


@dataclass(frozen=True, slots=True)
class TypedAxisFormulaFreezeV2:
    support: TaskBoundTypedAxisSupportArtifact
    inventory_address: str
    version_space_digest: str
    survivor_formula_ids: tuple[str, ...]
    selected_formula_id: str
    selected_formula_wire: Mapping[str, Any]
    selected_predicate_digest: str
    selection_mode: str
    rank_artifact: TypedAxisRankArtifactV2 | None
    record_digest: str

    def __post_init__(self) -> None:
        inventory = _inventory(self.support)
        wire = _canonical_wire(self.selected_formula_wire)
        expected_predicate = "sha256:" + canonical_digest(
            {"schema": "gkm.bongard-frozen-python-equality-predicate.v2", "wire": wire}
        )
        if (
            inventory.empty_gap is not None
            or self.inventory_address != inventory.inventory_address
            or self.version_space_digest != _version_space_digest(inventory)
            or self.survivor_formula_ids != inventory.admitted_formula_ids
            or self.selected_formula_id not in self.survivor_formula_ids
            or wire != _wire(_formula(inventory, self.selected_formula_id))
            or self.selected_predicate_digest != expected_predicate
        ):
            raise TypedAxisTaskRunnerV2Error("formula freeze differs from fixed version space")
        if len(self.survivor_formula_ids) == 1:
            if self.rank_artifact is not None or self.selection_mode != "unique_survivor_zero_model_calls":
                raise TypedAxisTaskRunnerV2Error("unique survivor used rank authority")
        elif (
            type(self.rank_artifact) is not TypedAxisRankArtifactV2
            or self.selection_mode != "journaled_full_permutation_rank"
            or self.rank_artifact.selected_formula_id != self.selected_formula_id
            or self.rank_artifact.rank_input.formula_ids != self.survivor_formula_ids
        ):
            raise TypedAxisTaskRunnerV2Error("multi-survivor journaled rank differs")
        if self.record_digest != "sha256:" + canonical_digest(_freeze_content(self)):
            raise TypedAxisTaskRunnerV2Error("formula freeze digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_freeze_content(self), "record_digest": self.record_digest}


TypedAxisTaskResultV2 = TypedAxisTaskGapV2 | TypedAxisFormulaFreezeV2


def run_typed_axis_formula_task_v2(
    support: TaskBoundTypedAxisSupportArtifact,
    *,
    rank_runtime: ObjectBongardTurnRuntime | None = None,
    rank_journal: ObjectBongardTextTurnJournalTransport | None = None,
) -> TypedAxisTaskResultV2:
    inventory = _inventory(support)
    version = _version_space_digest(inventory)
    if not inventory.admitted_formula_ids:
        if rank_runtime is not None or rank_journal is not None:
            raise TypedAxisTaskRunnerV2Error("empty version space cannot receive rank authority")
        values = {
            "support": support,
            "inventory_address": inventory.inventory_address,
            "version_space_digest": version,
            "empty_gap": inventory.empty_gap.to_data(),  # type: ignore[union-attr]
        }
        provisional = object.__new__(TypedAxisTaskGapV2)
        for name, item in values.items(): object.__setattr__(provisional, name, item)
        return TypedAxisTaskGapV2(**values, record_digest="sha256:" + canonical_digest(_gap_content(provisional)))
    rank: TypedAxisRankArtifactV2 | None = None
    if len(inventory.admitted_formula_ids) == 1:
        if rank_runtime is not None or rank_journal is not None:
            raise TypedAxisTaskRunnerV2Error("unique survivor must use zero model calls")
        selected = inventory.admitted_formula_ids[0]
        mode = "unique_survivor_zero_model_calls"
    else:
        if type(rank_runtime) is not ObjectBongardTurnRuntime or type(rank_journal) is not ObjectBongardTextTurnJournalTransport:
            raise TypedAxisTaskRunnerV2Error("multiple survivors require exact runtime and text journal")
        rank_input = TypedAxisRankInputV2.create(inventory)
        if (
            rank_journal.authorization_digest != support.release_authorization.record_digest
            or rank_journal.execution_precommit_digest != support.execution_precommit.record_digest
            or rank_journal.task_id != support.task_id
            or rank_journal.turn_kind != RANK_TURN_KIND
            or rank_journal.runtime != rank_runtime
            or rank_journal.expected_prompt != typed_axis_rank_prompt_v2(rank_input)
            or rank_journal.expected_output_schema != typed_axis_rank_schema_v2(rank_input)
        ):
            raise TypedAxisTaskRunnerV2Error("rank journal belongs to another task or version space")
        result = rank_journal(
            rank_journal.expected_prompt,
            rank_journal.expected_output_schema,
            model=rank_runtime.model,
            reasoning_effort=rank_runtime.reasoning_effort,
            minutes=rank_runtime.minutes,
            verbose=rank_runtime.verbose,
            executable=rank_runtime.executable,
            cloud_policy_cache_snapshot=rank_runtime.cloud_policy_cache_snapshot,
            model_catalog_snapshot=rank_runtime.model_catalog_snapshot,
            tool_surface_attestation=rank_runtime.no_tools_attestation,
            expected_launcher_digest=rank_runtime.expected_launcher_digest,
            expected_tool_surface_attestation_digest=rank_runtime.no_tools_attestation.attestation_digest,
        )
        if type(result) is not CodexStructuredResult:
            raise TypedAxisTaskRunnerV2Error("rank journal returned the wrong envelope")
        payload = json.loads(canonical_json(dict(result.payload)))
        aliases = payload.get("ordered_aliases")
        if type(aliases) is not list or len(aliases) != len(rank_input.aliases) or set(aliases) != set(rank_input.aliases) or len(set(aliases)) != len(aliases):
            raise TypedAxisTaskRunnerV2Error("rank payload must be the exact full permutation")
        by_alias = dict(zip(rank_input.aliases, rank_input.formula_ids, strict=True))
        ordered = tuple(by_alias[item] for item in aliases)
        summary = rank_journal.verify()
        values_rank = {
            "rank_input": rank_input,
            "model_payload": payload,
            "receipt": result.receipt,
            "ordered_formula_ids": ordered,
            "selected_formula_id": ordered[0],
            "journal_summary": summary,
        }
        provisional_rank = object.__new__(TypedAxisRankArtifactV2)
        for name, item in values_rank.items(): object.__setattr__(provisional_rank, name, item)
        rank = TypedAxisRankArtifactV2(**values_rank, record_digest="sha256:" + canonical_digest(provisional_rank.content_data()))
        selected = rank.selected_formula_id
        mode = "journaled_full_permutation_rank"
    wire = _wire(_formula(inventory, selected))
    values_freeze = {
        "support": support,
        "inventory_address": inventory.inventory_address,
        "version_space_digest": version,
        "survivor_formula_ids": inventory.admitted_formula_ids,
        "selected_formula_id": selected,
        "selected_formula_wire": wire,
        "selected_predicate_digest": "sha256:" + canonical_digest(
            {"schema": "gkm.bongard-frozen-python-equality-predicate.v2", "wire": wire}
        ),
        "selection_mode": mode,
        "rank_artifact": rank,
    }
    provisional_freeze = object.__new__(TypedAxisFormulaFreezeV2)
    for name, item in values_freeze.items(): object.__setattr__(provisional_freeze, name, item)
    return TypedAxisFormulaFreezeV2(**values_freeze, record_digest="sha256:" + canonical_digest(_freeze_content(provisional_freeze)))


def cold_replay_typed_axis_task_result_v2(
    result: TypedAxisTaskResultV2,
    *,
    expected_artifact_address: str,
    rank_journal: ObjectBongardTextTurnJournalTransport | None = None,
) -> TypedAxisTaskResultV2:
    expected = _address(expected_artifact_address, "expected task result")
    if type(result) is TypedAxisTaskGapV2:
        rebuilt = run_typed_axis_formula_task_v2(result.support)
    elif type(result) is TypedAxisFormulaFreezeV2:
        if result.rank_artifact is None:
            rebuilt = run_typed_axis_formula_task_v2(result.support)
        else:
            if type(rank_journal) is not ObjectBongardTextTurnJournalTransport:
                raise TypedAxisTaskRunnerV2Error("ranked replay needs its exact external journal")
            if rank_journal.verify().to_data() != result.rank_artifact.journal_summary.to_data():
                raise TypedAxisTaskRunnerV2Error("rank journal terminal differs on cold replay")
            # Rebuild selection from the archived, receipted full permutation;
            # never invoke the journal transport during cold replay.
            inventory = _inventory(result.support)
            expected_input = TypedAxisRankInputV2.create(inventory)
            if result.rank_artifact.rank_input != expected_input:
                raise TypedAxisTaskRunnerV2Error("rank input differs on cold replay")
            rebuilt = result
            TypedAxisFormulaFreezeV2(**{name: getattr(result, name) for name in result.__dataclass_fields__})
    else:
        raise TypeError("task replay needs exact v2 gap or freeze")
    if rebuilt != result or result.record_digest != expected:
        raise TypedAxisTaskRunnerV2Error("typed-axis task cold replay differs")
    return result


__all__ = (
    "TypedAxisFormulaFreezeV2",
    "TypedAxisRankArtifactV2",
    "TypedAxisRankInputV2",
    "TypedAxisTaskGapV2",
    "TypedAxisTaskResultV2",
    "TypedAxisTaskRunnerV2Error",
    "build_typed_axis_rank_journal_v2",
    "cold_replay_typed_axis_task_result_v2",
    "panel_typed_axis_task_runner_v2_source_digest",
    "run_typed_axis_formula_task_v2",
    "typed_axis_rank_prompt_v2",
    "typed_axis_rank_schema_v2",
)
