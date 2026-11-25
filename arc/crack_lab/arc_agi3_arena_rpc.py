#!/usr/bin/env python3
"""Narrow host-owned Arena RPC boundary for the contiguous ARC-AGI-3 run.

The trusted host replays the admitted parent path into one immutable
``gkm_arena.Arena`` seed before any model observation.  An untrusted solver
container receives one replaceable exploration proxy exposing only observe,
reset-by-reclone, step, available actions, completed-level reward, and terminal
state.  Engine objects, game source, private attributes, paths, and host
tracebacks never cross the boundary.

This module is transport and schema code.  Container construction, campaign
scheduling, promotion, and independent replay remain separate trusted gates.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import multiprocessing
import numbers
import os
import secrets
import socket
import stat
import threading
import time
import enum
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import arc_agi3_contiguous_supervisor as ProbeContract


RPC_SCHEMA = "arc-agi3-arena-rpc/v1"
PUBLIC_OBSERVATION_RECEIPT_SCHEMA = 1
PUBLIC_OBSERVATION_RECEIPT_KIND = (
    "arc_agi3_public_observation_content"
)
PUBLIC_ACTION_BASIS_KIND = "arc_agi3_public_action_basis"
PUBLIC_RESPONSE_SIGNATURE_KIND = (
    "arc_agi3_public_response_signature"
)
PUBLIC_ACTION_BASIS_GENESIS_SHA256 = hashlib.sha256(
    b"arc_agi3_public_action_basis_genesis_v1"
).hexdigest()
REWARD_BOUNDARY_POLICY = (
    "first_exact_reward_absorbing_fresh_replay_v1"
)
ACTION7_ROLLBACK_POLICY = (
    "context_specific_exact_frame_level_terminal_or_reconstruct_v1"
)
MAX_MESSAGE_BYTES = 256 * 1024
MAX_FRAME_SIDE = 64
MAX_TOTAL_CELLS = MAX_FRAME_SIDE * MAX_FRAME_SIDE
DEFAULT_REAL_STEP_CAP = 600
DEFAULT_TOTAL_STEP_CAP = 240_000
DEFAULT_RESET_CAP = 8_192
DEFAULT_SOCKET_TIMEOUT_SECONDS = 30.0
FRESH_PROCESS_START_TIMEOUT_SECONDS = 15.0
FRESH_PROCESS_STOP_TIMEOUT_SECONDS = 5.0
_SHA256_RE = __import__("re").compile(r"[0-9a-f]{64}")
_IDENTIFIER_RE = __import__("re").compile(r"[A-Za-z0-9_.:-]{1,256}")
_PUBLIC_OPERATIONS = frozenset({
    "open",
    "observe",
    "reset",
    "step",
    "close",
})
_EXPLORATION_MODES = frozenset({
    "continue_parent",
    "fresh_prefix",
})


class ArenaRpcError(RuntimeError):
    """A local transport or remote public-contract failure."""


class ArenaRpcContractError(ArenaRpcError):
    """A malformed or unauthorized RPC request."""


class ArenaLike(Protocol):
    """The trusted subset required from ``gkm_arena.Arena``."""

    @property
    def actions(self) -> Sequence[int]: ...

    @property
    def levels_completed(self) -> int: ...

    @property
    def path(self) -> Sequence[Any]: ...

    def terminal(self) -> bool: ...

    def frame(self) -> Any: ...

    def reset(self) -> Any: ...

    def step(self, action: Any, x: int | None = None, y: int | None = None) -> Any: ...

    def clone(self) -> "ArenaLike": ...


@dataclass(frozen=True)
class ArenaSessionBinding:
    """Exact immutable lineage/frontier identity for one Arena attempt."""

    campaign_id: str
    generation_id: str
    attempt_id: str
    game: str
    parent_level: int
    target_level: int
    parent_checkpoint_sha256: str
    frontier_sha256: str
    exploration_mode: str


@dataclass(frozen=True)
class ArenaHostResult:
    """Trusted result retained on the host after the client disconnects."""

    binding_sha256: str
    game: str
    exploration_mode: str
    parent_level: int
    levels_completed: int
    parent_path: tuple[int | tuple[int, int, int], ...]
    path: tuple[int | tuple[int, int, int], ...]
    parent_replay_steps: int
    exploration_steps: int
    resets: int
    total_steps: int
    parent_terminal: bool
    parent_snapshot_sha256: str


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _strict_object(
    value: object,
    *,
    required: set[str],
    optional: set[str] | None = None,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ArenaRpcContractError(f"{label} must be a JSON object")
    optional = optional or set()
    keys = set(value)
    if keys != required and not (
        required <= keys and keys <= required | optional
    ):
        # Field names are untrusted.  Do not reflect them into the response or
        # the immutable host transcript: a malicious key can contain a token,
        # host path, or other data the solver wants to smuggle into evidence.
        raise ArenaRpcContractError(f"{label} fields mismatch")
    return value


def _loads_json(raw: bytes, *, label: str) -> object:
    """Decode strict JSON, rejecting duplicate keys and non-finite aliases."""

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ArenaRpcContractError(
                    f"{label} contains duplicate object fields"
                )
            result[key] = value
        return result

    def reject_constant(_value: str) -> None:
        raise ArenaRpcContractError(
            f"{label} contains a non-finite number"
        )

    try:
        return json.loads(
            raw,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except ArenaRpcContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArenaRpcContractError(f"{label} is not valid JSON") from exc


def _safe_operation_label(value: object) -> str | None:
    return value if isinstance(value, str) and value in _PUBLIC_OPERATIONS else None


def _normalize_action(value: object) -> int | tuple[int, int, int]:
    if _is_plain_int(value):
        action = int(value)
        if 1 <= action <= 7 and action != 6:
            return action
        raise ArenaRpcContractError(
            "simple action must be an integer in 1..5 or 7"
        )
    if (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(_is_plain_int(item) for item in value)
        and value[0] == 6
    ):
        x, y = int(value[1]), int(value[2])
        if not (0 <= x < MAX_FRAME_SIDE and 0 <= y < MAX_FRAME_SIDE):
            raise ArenaRpcContractError(
                "coordinate action must be [6, x, y] with x,y in 0..63"
            )
        return (6, x, y)
    raise ArenaRpcContractError(
        "action must be a simple integer 1..5 or 7, or [6, x, y]"
    )


def _normalize_path(
    value: Sequence[Any],
) -> tuple[int | tuple[int, int, int], ...]:
    out: list[int | tuple[int, int, int]] = []
    for action in value:
        normalized = _normalize_action(action)
        out.append(normalized)
    return tuple(out)


def _normalize_frame(frame: Any) -> list[list[int]]:
    """Convert a frame without exposing NumPy-specific objects over JSON."""

    if hasattr(frame, "tolist"):
        frame = frame.tolist()
    if not isinstance(frame, (list, tuple)) or not frame:
        raise ArenaRpcContractError("Arena frame must be a nonempty row sequence")
    if len(frame) > MAX_FRAME_SIDE:
        raise ArenaRpcContractError("Arena frame exceeds 64 rows")
    rows: list[list[int]] = []
    width: int | None = None
    cells = 0
    for raw_row in frame:
        if not isinstance(raw_row, (list, tuple)) or not raw_row:
            raise ArenaRpcContractError("Arena frame rows must be nonempty sequences")
        if width is None:
            width = len(raw_row)
            if width > MAX_FRAME_SIDE:
                raise ArenaRpcContractError("Arena frame exceeds 64 columns")
        elif len(raw_row) != width:
            raise ArenaRpcContractError("Arena frame must be rectangular")
        row: list[int] = []
        for raw_cell in raw_row:
            if (
                isinstance(raw_cell, bool)
                or not isinstance(raw_cell, numbers.Integral)
            ):
                raise ArenaRpcContractError(
                    "Arena frame cells must be integer colour indices"
                )
            cell = int(raw_cell)
            if not (0 <= cell <= 15):
                raise ArenaRpcContractError(
                    "Arena frame cells must be colour indices in 0..15"
                )
            row.append(cell)
        rows.append(row)
        cells += len(row)
    if cells > MAX_TOTAL_CELLS:
        raise ArenaRpcContractError("Arena frame exceeds the 64x64 cell bound")
    return rows


def _normalize_actions(actions: object) -> list[int]:
    if not isinstance(actions, (list, tuple)):
        raise ArenaRpcContractError("Arena actions must be a sequence")
    out: list[int] = []
    for raw in actions:
        if not _is_plain_int(raw) or not 1 <= int(raw) <= 7:
            raise ArenaRpcContractError("Arena action IDs must be integers in 1..7")
        action = int(raw)
        if action in out:
            raise ArenaRpcContractError("Arena action IDs must be unique")
        out.append(action)
    if not out:
        raise ArenaRpcContractError("Arena must expose at least one action")
    return out


def _snapshot_payload(arena: ArenaLike) -> dict[str, Any]:
    levels = arena.levels_completed
    if not _is_plain_int(levels) or int(levels) < 0:
        raise ArenaRpcContractError(
            "Arena levels_completed must be a nonnegative integer"
        )
    terminal = arena.terminal()
    if not isinstance(terminal, bool):
        raise ArenaRpcContractError("Arena terminal() must return a boolean")
    return {
        "frame": _normalize_frame(arena.frame()),
        "actions": _normalize_actions(arena.actions),
        "levels_completed": int(levels),
        "terminal": terminal,
    }


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def public_observation_receipt_bytes(
    value: Mapping[str, Any],
) -> bytes:
    """Canonical immutable bytes; their SHA-256 is the semantic identity."""

    validate_public_observation_receipt(value)
    return _canonical_json(value)


def validate_public_observation_receipt(
    value: object,
    *,
    game: str | None = None,
    frontier_sha256: str | None = None,
    parent_checkpoint_sha256: str | None = None,
) -> str:
    """Validate one metadata-free public action/response content receipt."""

    receipt = _strict_object(
        value,
        required={
            "schema",
            "kind",
            "game",
            "frontier_sha256",
            "parent_checkpoint_sha256",
            "public_action_basis",
            "public_action_basis_sha256",
            "public_response_signature",
            "public_response_signature_sha256",
        },
        label="public observation receipt",
    )
    basis = _strict_object(
        receipt["public_action_basis"],
        required={
            "schema",
            "kind",
            "operation_index",
            "previous_public_action_basis_sha256",
            "operation",
        },
        label="public action basis",
    )
    signature = _strict_object(
        receipt["public_response_signature"],
        required={"schema", "kind", "operation_index", "result"},
        label="public response signature",
    )
    if (
        receipt["schema"] != PUBLIC_OBSERVATION_RECEIPT_SCHEMA
        or receipt["kind"] != PUBLIC_OBSERVATION_RECEIPT_KIND
        or not isinstance(receipt["game"], str)
        or _IDENTIFIER_RE.fullmatch(receipt["game"]) is None
        or _SHA256_RE.fullmatch(
            str(receipt["frontier_sha256"])
        )
        is None
        or _SHA256_RE.fullmatch(
            str(receipt["parent_checkpoint_sha256"])
        )
        is None
        or (
            game is not None
            and receipt["game"] != game
        )
        or (
            frontier_sha256 is not None
            and receipt["frontier_sha256"] != frontier_sha256
        )
        or (
            parent_checkpoint_sha256 is not None
            and receipt["parent_checkpoint_sha256"]
            != parent_checkpoint_sha256
        )
        or basis["schema"] != PUBLIC_OBSERVATION_RECEIPT_SCHEMA
        or basis["kind"] != PUBLIC_ACTION_BASIS_KIND
        or not _is_plain_int(basis["operation_index"])
        or basis["operation_index"] < 0
        or _SHA256_RE.fullmatch(
            str(basis["previous_public_action_basis_sha256"])
        )
        is None
        or signature["schema"]
        != PUBLIC_OBSERVATION_RECEIPT_SCHEMA
        or signature["kind"] != PUBLIC_RESPONSE_SIGNATURE_KIND
        or not _is_plain_int(signature["operation_index"])
        or signature["operation_index"] != basis["operation_index"]
        or not isinstance(signature["result"], dict)
    ):
        raise ArenaRpcContractError(
            "public observation receipt identity is malformed"
        )
    operation = _strict_object(
        basis["operation"],
        required={"op"},
        optional={"action"},
        label="public action-basis operation",
    )
    op = operation["op"]
    if (
        not isinstance(op, str)
        or op not in {"open", "observe", "reset", "step"}
        or (basis["operation_index"] == 0) != (op == "open")
        or (
            basis["operation_index"] == 0
            and basis["previous_public_action_basis_sha256"]
            != PUBLIC_ACTION_BASIS_GENESIS_SHA256
        )
    ):
        raise ArenaRpcContractError(
            "public action basis has an invalid operation order"
        )
    if op == "step":
        if set(operation) != {"op", "action"}:
            raise ArenaRpcContractError(
                "public step basis lacks its exact action"
            )
        normalized = _normalize_action(operation["action"])
        expected_action = (
            list(normalized)
            if isinstance(normalized, tuple)
            else normalized
        )
        if operation["action"] != expected_action:
            raise ArenaRpcContractError(
                "public step action is not canonically encoded"
            )
    elif set(operation) != {"op"}:
        raise ArenaRpcContractError(
            "non-step public operation carries an action"
        )
    result = signature["result"]
    snapshot = result.get("snapshot")
    if (
        set(result)
        not in ({"snapshot"}, {"binding_sha256", "snapshot"})
        or (basis["operation_index"] == 0)
        != ("binding_sha256" in result)
        or (
            "binding_sha256" in result
            and _SHA256_RE.fullmatch(str(result["binding_sha256"]))
            is None
        )
        or not isinstance(snapshot, dict)
        or set(snapshot)
        != {"frame", "actions", "levels_completed", "terminal"}
        or _normalize_frame(snapshot["frame"]) != snapshot["frame"]
        or _normalize_actions(snapshot["actions"])
        != snapshot["actions"]
        or not _is_plain_int(snapshot["levels_completed"])
        or snapshot["levels_completed"] < 0
        or not isinstance(snapshot["terminal"], bool)
    ):
        raise ArenaRpcContractError(
            "public response signature is not a canonical Arena snapshot"
        )
    if (
        receipt["public_action_basis_sha256"]
        != _json_sha256(dict(basis))
        or receipt["public_response_signature_sha256"]
        != _json_sha256(dict(signature))
    ):
        raise ArenaRpcContractError(
            "public observation commitments do not match their content"
        )
    return _json_sha256(dict(receipt))


def _public_state_payload(arena: ArenaLike) -> dict[str, Any]:
    return {
        "path": [
            list(action) if isinstance(action, tuple) else action
            for action in _normalize_path(tuple(arena.path))
        ],
        "snapshot": _snapshot_payload(arena),
    }


def _reward_boundary_payload(
    *,
    binding: ArenaSessionBinding,
    binding_sha256: str,
    path: tuple[int | tuple[int, int, int], ...],
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the first exact reward as an absorbing session boundary."""

    if (
        not path
        or snapshot.get("levels_completed") != binding.target_level
    ):
        raise ArenaRpcContractError(
            "reward boundary is not the exact target transition"
        )
    crossing_action = path[-1]
    return {
        "schema": RPC_SCHEMA,
        "kind": "arena_reward_boundary",
        "policy": REWARD_BOUNDARY_POLICY,
        "binding_sha256": binding_sha256,
        "game": binding.game,
        "parent_level": binding.parent_level,
        "target_level": binding.target_level,
        "levels_before": binding.parent_level,
        "levels_completed": binding.target_level,
        "path_sha256": _json_sha256([
            list(action) if isinstance(action, tuple) else action
            for action in path
        ]),
        "path_length": len(path),
        "crossing_action_sha256": _json_sha256(
            list(crossing_action)
            if isinstance(crossing_action, tuple)
            else crossing_action
        ),
        "snapshot_sha256": _json_sha256(dict(snapshot)),
    }


def _rollback_projection(
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the complete public state ACTION7 must restore exactly."""

    if (
        set(snapshot)
        != {"frame", "actions", "levels_completed", "terminal"}
    ):
        raise ArenaRpcContractError(
            "rollback snapshot does not have the public Arena schema"
        )
    return {
        "frame": json.loads(
            json.dumps(
                snapshot["frame"],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        ),
        "levels_completed": snapshot["levels_completed"],
        "terminal": snapshot["terminal"],
    }


def _apply_exact_action(
    arena: ArenaLike,
    action: int | tuple[int, int, int],
) -> None:
    before = _normalize_path(tuple(arena.path))
    if isinstance(action, tuple):
        arena.step(action[0], action[1], action[2])
    else:
        arena.step(action)
    after = _normalize_path(tuple(arena.path))
    if (
        len(after) != len(before) + 1
        or after[:-1] != before
        or after[-1] != action
    ):
        raise ArenaRpcContractError(
            "Arena step did not append exactly the requested action"
        )


def _fresh_arena_at_seed(
    arena_factory: Callable[[str], ArenaLike],
    game: str,
    seed_path: tuple[int | tuple[int, int, int], ...],
    seed_snapshot: Mapping[str, Any],
) -> ArenaLike:
    arena = arena_factory(game)
    if (
        _normalize_path(tuple(arena.path))
        or int(arena.levels_completed) != 0
    ):
        raise ArenaRpcContractError(
            "fresh-process Arena did not begin at public zero"
        )
    for action in seed_path:
        action_id = action[0] if isinstance(action, tuple) else action
        if action_id not in _normalize_actions(arena.actions):
            raise ArenaRpcContractError(
                "fresh-process seed replay requested an unavailable action"
            )
        _apply_exact_action(arena, action)
    if (
        _normalize_path(tuple(arena.path)) != seed_path
        or _snapshot_payload(arena) != dict(seed_snapshot)
    ):
        raise ArenaRpcContractError(
            "fresh-process seed replay differs from the authenticated seed"
        )
    return arena


def _deterministic_canary_action(
    arena: ArenaLike,
) -> int | tuple[int, int, int]:
    actions = _normalize_actions(arena.actions)
    simple = sorted(action for action in actions if action != 6)
    if simple:
        return simple[0]
    if 6 in actions:
        return (6, 0, 0)
    raise ArenaRpcContractError(
        "Arena exposes no deterministic canary action"
    )


def _reachable_mutable_identities(
    root: object,
    *,
    maximum_nodes: int = 200_000,
) -> tuple[set[int], int, int]:
    """Inventory reachable mutable Python state without retaining its values.

    Unknown opaque objects make the result inconclusive.  The host never emits
    object names, values, identities, or types; only bounded counts enter the
    hashed canary observation.
    """

    atomic = (
        type(None),
        bool,
        int,
        float,
        complex,
        str,
        bytes,
        range,
        types.CodeType,
        types.FunctionType,
        types.BuiltinFunctionType,
        types.MethodType,
        types.BuiltinMethodType,
        types.ModuleType,
        type,
        enum.Enum,
    )
    mutable: set[int] = set()
    visited: set[int] = set()
    opaque = 0
    stack = [root]
    while stack:
        value = stack.pop()
        if isinstance(value, atomic):
            continue
        identity = id(value)
        if identity in visited:
            continue
        visited.add(identity)
        if len(visited) > maximum_nodes:
            raise ArenaRpcContractError(
                "clone-isolation mutable graph exceeds its hard bound"
            )
        if isinstance(value, (tuple, frozenset)):
            stack.extend(value)
            continue
        if isinstance(value, dict):
            mutable.add(identity)
            stack.extend(value.keys())
            stack.extend(value.values())
            continue
        if isinstance(value, (list, set)):
            mutable.add(identity)
            stack.extend(value)
            continue
        if isinstance(value, (bytearray, memoryview)):
            mutable.add(identity)
            continue
        try:
            namespace = object.__getattribute__(value, "__dict__")
        except (AttributeError, TypeError):
            namespace = None
        if isinstance(namespace, dict):
            mutable.add(identity)
            # Two distinct wrappers can be made to share one assigned
            # ``__dict__``.  Inventory the namespace object itself, not only
            # the values reachable through it.
            mutable.add(id(namespace))
            stack.extend(namespace.values())
            continue
        slots: list[str] = []
        try:
            lineage = type(value).__mro__
        except (AttributeError, TypeError):
            lineage = ()
        for owner in lineage:
            declared = owner.__dict__.get("__slots__", ())
            if isinstance(declared, str):
                declared = (declared,)
            if isinstance(declared, (tuple, list)):
                slots.extend(
                    slot for slot in declared
                    if isinstance(slot, str)
                    and slot not in {"__dict__", "__weakref__"}
                )
        if slots:
            mutable.add(identity)
            for slot in slots:
                try:
                    child = object.__getattribute__(value, slot)
                except (AttributeError, TypeError):
                    continue
                stack.append(child)
            continue
        # A distinct wrapper can still share an uninspectable C allocation.
        # Treat every such object as opaque rather than inferring isolation.
        mutable.add(identity)
        opaque += 1
    return mutable, len(visited), opaque


def _mutable_graph_canary(
    seed: ArenaLike,
    left: ArenaLike,
    right: ArenaLike,
) -> tuple[str, int, str]:
    try:
        seed_ids, seed_nodes, seed_opaque = (
            _reachable_mutable_identities(seed)
        )
        left_ids, left_nodes, left_opaque = (
            _reachable_mutable_identities(left)
        )
        right_ids, right_nodes, right_opaque = (
            _reachable_mutable_identities(right)
        )
    except Exception:
        return (
            "INCONCLUSIVE",
            0,
            _json_sha256({
                "status": "INCONCLUSIVE",
                "bounded": False,
            }),
        )
    shared = (
        (seed_ids & left_ids)
        | (seed_ids & right_ids)
        | (left_ids & right_ids)
    )
    opaque_count = seed_opaque + left_opaque + right_opaque
    status = (
        "LEAK"
        if shared
        else "INCONCLUSIVE"
        if opaque_count
        else "PASS"
    )
    observation = {
        "status": status,
        "shared_mutable_identity_count": len(shared),
        "reachable_node_counts": [
            seed_nodes,
            left_nodes,
            right_nodes,
        ],
        "opaque_node_count": opaque_count,
        "bounded": True,
    }
    return status, len(shared), _json_sha256(observation)


def _clone_isolation_canary(
    *,
    arena_factory: Callable[[str], ArenaLike],
    game: str,
    seed_path: tuple[int | tuple[int, int, int], ...],
    seed_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Behaviorally compare sibling branches on a sacrificial exact seed."""

    seed_path_wire = [
        list(action) if isinstance(action, tuple) else action
        for action in seed_path
    ]
    fields: dict[str, Any] = {
        "schema": ProbeContract.PROBE_ISOLATION_SCHEMA,
        "kind": ProbeContract.PROBE_ISOLATION_KIND,
        "authority": ProbeContract.PROBE_ISOLATION_AUTHORITY,
        "algorithm": ProbeContract.PROBE_ISOLATION_CANARY,
        "mode": ProbeContract.FRESH_PROCESS_PER_CANDIDATE_MODE,
        "seed_snapshot_sha256": _json_sha256(dict(seed_snapshot)),
        "seed_path_sha256": _json_sha256(seed_path_wire),
        "canary_status": "INCONCLUSIVE",
        "failure_stage": "SEED",
        "canary_action": 1,
        "canary_action_sha256": _json_sha256(1),
        "mutable_graph_status": "INCONCLUSIVE",
        "shared_mutable_identity_count": 0,
        "mutable_graph_observation_sha256": _json_sha256({
            "status": "INCONCLUSIVE",
            "bounded": False,
        }),
        "seed_before_sha256": None,
        "left_before_sha256": None,
        "right_before_sha256": None,
        "left_after_sha256": None,
        "right_after_left_sha256": None,
        "seed_after_left_sha256": None,
        "right_after_sha256": None,
        "seed_after_right_sha256": None,
        "mutation_observed": False,
        "sibling_unchanged": False,
        "matching_trajectory": False,
        "fallback_process_ready": False,
        "fallback_process_identity_sha256": None,
    }
    try:
        seed = _fresh_arena_at_seed(
            arena_factory, game, seed_path, seed_snapshot
        )
        action = _deterministic_canary_action(seed)
        wire_action = (
            list(action) if isinstance(action, tuple) else action
        )
        fields["canary_action"] = wire_action
        fields["canary_action_sha256"] = _json_sha256(wire_action)
        seed_before = _json_sha256(_public_state_payload(seed))
        fields["seed_before_sha256"] = seed_before
    except Exception:
        return fields

    fields["failure_stage"] = "CLONE"
    try:
        left = seed.clone()
        right = seed.clone()
        if (
            left is seed
            or right is seed
            or left is right
        ):
            fields["canary_status"] = "LEAK"
            return fields
        left_before = _json_sha256(_public_state_payload(left))
        right_before = _json_sha256(_public_state_payload(right))
        fields["left_before_sha256"] = left_before
        fields["right_before_sha256"] = right_before
        if left_before != seed_before or right_before != seed_before:
            fields["canary_status"] = "LEAK"
            return fields
        graph_status, shared_count, graph_digest = (
            _mutable_graph_canary(seed, left, right)
        )
        fields["mutable_graph_status"] = graph_status
        fields["shared_mutable_identity_count"] = shared_count
        fields["mutable_graph_observation_sha256"] = graph_digest
        if graph_status != "PASS":
            fields["canary_status"] = graph_status
            return fields
    except Exception:
        return fields

    fields["failure_stage"] = "LEFT_STEP"
    try:
        _apply_exact_action(left, action)
    except Exception:
        return fields
    fields["failure_stage"] = "LEFT_OBSERVATION"
    try:
        left_after = _json_sha256(_public_state_payload(left))
        right_after_left = _json_sha256(_public_state_payload(right))
        seed_after_left = _json_sha256(_public_state_payload(seed))
        fields.update({
            "left_after_sha256": left_after,
            "right_after_left_sha256": right_after_left,
            "seed_after_left_sha256": seed_after_left,
            "mutation_observed": left_after != left_before,
            "sibling_unchanged": (
                right_after_left == right_before
                and seed_after_left == seed_before
            ),
        })
        if not fields["mutation_observed"] or not fields["sibling_unchanged"]:
            fields["canary_status"] = "LEAK"
            return fields
    except Exception:
        return fields

    fields["failure_stage"] = "RIGHT_STEP"
    try:
        _apply_exact_action(right, action)
    except Exception:
        return fields
    fields["failure_stage"] = "RIGHT_OBSERVATION"
    try:
        right_after = _json_sha256(_public_state_payload(right))
        seed_after_right = _json_sha256(_public_state_payload(seed))
        fields["right_after_sha256"] = right_after
        fields["seed_after_right_sha256"] = seed_after_right
        fields["matching_trajectory"] = (
            right_after == left_after
            and seed_after_right == seed_before
        )
    except Exception:
        return fields
    fields["failure_stage"] = "COMPARE"
    if not fields["matching_trajectory"]:
        fields["canary_status"] = "LEAK"
        return fields
    fields.update({
        "mode": ProbeContract.VERIFIED_ISOLATED_CLONE_MODE,
        "canary_status": "PASS",
        "failure_stage": "NONE",
    })
    return fields


def _fresh_process_mac(
    token: str, value: Mapping[str, Any]
) -> str:
    return hmac.new(
        token.encode("ascii"),
        _canonical_json(dict(value)),
        hashlib.sha256,
    ).hexdigest()


def _fresh_process_send(
    connection: Any,
    token: str,
    body: Mapping[str, Any],
) -> None:
    connection.send({
        **dict(body),
        "mac": _fresh_process_mac(token, body),
    })


def _fresh_process_worker(
    connection: Any,
    token: str,
    game: str,
    arena_factory: Callable[[str], ArenaLike],
    seed_path: tuple[int | tuple[int, int, int], ...],
    seed_snapshot: dict[str, Any],
    process_nonce: str,
) -> None:
    """Own exactly one candidate branch in a separate spawned process."""

    try:
        arena = _fresh_arena_at_seed(
            arena_factory, game, seed_path, seed_snapshot
        )
        ready = {
            "kind": "ready",
            "seq": -1,
            "pid": os.getpid(),
            "process_nonce": process_nonce,
            "state": _public_state_payload(arena),
        }
        _fresh_process_send(connection, token, ready)
        expected_seq = 0
        while True:
            value = connection.recv()
            if not isinstance(value, dict):
                raise ArenaRpcContractError(
                    "fresh-process request is not an object"
                )
            request = dict(value)
            observed_mac = request.pop("mac", None)
            if (
                not isinstance(observed_mac, str)
                or not hmac.compare_digest(
                    observed_mac,
                    _fresh_process_mac(token, request),
                )
                or request.get("kind") != "request"
                or request.get("seq") != expected_seq
                or set(request)
                not in (
                    {"kind", "seq", "op"},
                    {"kind", "seq", "op", "action"},
                )
            ):
                raise ArenaRpcContractError(
                    "fresh-process request authentication failed"
                )
            op = request["op"]
            if op == "step" and set(request) == {
                "kind",
                "seq",
                "op",
                "action",
            }:
                action = _normalize_action(request["action"])
                action_id = (
                    action[0] if isinstance(action, tuple) else action
                )
                if action_id not in _normalize_actions(arena.actions):
                    raise ArenaRpcContractError(
                        "fresh-process action is unavailable"
                    )
                _apply_exact_action(arena, action)
                _fresh_process_send(
                    connection,
                    token,
                    {
                        "kind": "response",
                        "seq": expected_seq,
                        "ok": True,
                        "state": _public_state_payload(arena),
                    },
                )
            elif op == "close" and set(request) == {
                "kind",
                "seq",
                "op",
            }:
                _fresh_process_send(
                    connection,
                    token,
                    {
                        "kind": "response",
                        "seq": expected_seq,
                        "ok": True,
                        "state": _public_state_payload(arena),
                    },
                )
                return
            else:
                raise ArenaRpcContractError(
                    "fresh-process operation is invalid"
                )
            expected_seq += 1
    except BaseException:
        try:
            _fresh_process_send(
                connection,
                token,
                {
                    "kind": "fatal",
                    "seq": -1,
                    "error": "engine_contract_failure",
                },
            )
        except BaseException:
            pass
    finally:
        connection.close()


class _FreshProcessArena:
    """ArenaLike proxy whose entire candidate branch lives in one process."""

    def __init__(
        self,
        *,
        game: str,
        arena_factory: Callable[[str], ArenaLike],
        seed_path: tuple[int | tuple[int, int, int], ...],
        seed_snapshot: Mapping[str, Any],
    ) -> None:
        try:
            context = multiprocessing.get_context("spawn")
            parent, child = context.Pipe(duplex=True)
        except (RuntimeError, ValueError, OSError) as exc:
            raise ArenaRpcContractError(
                "fresh-process probing substrate is unavailable"
            ) from exc
        self._connection = parent
        self._token = secrets.token_hex(32)
        self._seq = 0
        self._closed = False
        process_nonce = secrets.token_hex(32)
        self._process = context.Process(
            target=_fresh_process_worker,
            args=(
                child,
                self._token,
                game,
                arena_factory,
                seed_path,
                dict(seed_snapshot),
                process_nonce,
            ),
            name="arc-agi3-fresh-probe",
            daemon=False,
        )
        try:
            self._process.start()
        except BaseException as exc:
            parent.close()
            child.close()
            raise ArenaRpcContractError(
                "independent clone failed and fresh-process probing could "
                "not start"
            ) from exc
        child.close()
        try:
            ready = self._receive(
                timeout=FRESH_PROCESS_START_TIMEOUT_SECONDS
            )
            if (
                set(ready)
                != {
                    "kind",
                    "seq",
                    "pid",
                    "process_nonce",
                    "state",
                    "mac",
                }
                or ready.get("kind") != "ready"
                or ready.get("seq") != -1
                or ready.get("pid") != self._process.pid
                or ready.get("process_nonce") != process_nonce
            ):
                raise ArenaRpcContractError(
                    "fresh-process readiness evidence is malformed"
                )
            self._state = self._validate_state(ready["state"])
            expected = {
                "path": [
                    list(action) if isinstance(action, tuple) else action
                    for action in seed_path
                ],
                "snapshot": dict(seed_snapshot),
            }
            if self._state != expected:
                raise ArenaRpcContractError(
                    "fresh-process readiness targets another seed"
                )
            self.identity_sha256 = _json_sha256({
                "pid": ready["pid"],
                "process_nonce": process_nonce,
                "seed_path_sha256": _json_sha256(expected["path"]),
                "seed_snapshot_sha256": _json_sha256(
                    expected["snapshot"]
                ),
            })
        except BaseException:
            self.terminate()
            raise

    def _receive(self, *, timeout: float) -> dict[str, Any]:
        if not self._connection.poll(timeout):
            raise ArenaRpcContractError(
                "fresh-process probe did not answer within its hard bound"
            )
        try:
            value = self._connection.recv()
        except (EOFError, OSError) as exc:
            raise ArenaRpcContractError(
                "fresh-process probe exited without authenticated evidence"
            ) from exc
        if not isinstance(value, dict):
            raise ArenaRpcContractError(
                "fresh-process response is not an object"
            )
        unsigned = dict(value)
        observed_mac = unsigned.pop("mac", None)
        if (
            not isinstance(observed_mac, str)
            or not hmac.compare_digest(
                observed_mac,
                _fresh_process_mac(self._token, unsigned),
            )
        ):
            raise ArenaRpcContractError(
                "fresh-process response authentication failed"
            )
        if unsigned.get("kind") == "fatal":
            raise ArenaRpcContractError(
                "fresh-process engine contract failed"
            )
        return value

    @staticmethod
    def _validate_state(value: object) -> dict[str, Any]:
        if (
            not isinstance(value, dict)
            or set(value) != {"path", "snapshot"}
            or not isinstance(value["path"], list)
            or not isinstance(value["snapshot"], dict)
        ):
            raise ArenaRpcContractError(
                "fresh-process state evidence is malformed"
            )
        normalized_path = _normalize_path(tuple(value["path"]))
        snapshot = value["snapshot"]
        # Re-run all public normalization without constructing an engine.
        if (
            set(snapshot)
            != {"frame", "actions", "levels_completed", "terminal"}
            or _normalize_frame(snapshot["frame"]) != snapshot["frame"]
            or _normalize_actions(snapshot["actions"])
            != snapshot["actions"]
            or not _is_plain_int(snapshot["levels_completed"])
            or snapshot["levels_completed"] < 0
            or not isinstance(snapshot["terminal"], bool)
        ):
            raise ArenaRpcContractError(
                "fresh-process public snapshot is malformed"
            )
        return {
            "path": [
                list(action) if isinstance(action, tuple) else action
                for action in normalized_path
            ],
            "snapshot": json.loads(
                json.dumps(snapshot, sort_keys=True)
            ),
        }

    @property
    def actions(self) -> Sequence[int]:
        return tuple(self._state["snapshot"]["actions"])

    @property
    def levels_completed(self) -> int:
        return int(self._state["snapshot"]["levels_completed"])

    @property
    def path(self) -> Sequence[Any]:
        return tuple(
            tuple(action) if isinstance(action, list) else action
            for action in self._state["path"]
        )

    def terminal(self) -> bool:
        return bool(self._state["snapshot"]["terminal"])

    def frame(self) -> Any:
        return [
            list(row) for row in self._state["snapshot"]["frame"]
        ]

    def reset(self) -> Any:
        raise ArenaRpcContractError(
            "fresh-process branches are replaced, never reset"
        )

    def clone(self) -> ArenaLike:
        raise ArenaRpcContractError(
            "fresh-process branches cannot create engine clones"
        )

    def step(
        self,
        action: Any,
        x: int | None = None,
        y: int | None = None,
    ) -> Any:
        normalized = (
            _normalize_action((action, x, y))
            if action == 6
            else _normalize_action(action)
        )
        if action != 6 and (x is not None or y is not None):
            raise ArenaRpcContractError(
                "fresh-process simple action cannot carry coordinates"
            )
        wire = (
            list(normalized)
            if isinstance(normalized, tuple)
            else normalized
        )
        response = self._request("step", action=wire)
        self._state = self._validate_state(response["state"])
        return self.frame()

    def _request(
        self, op: str, *, action: object | None = None
    ) -> dict[str, Any]:
        if self._closed:
            raise ArenaRpcContractError(
                "fresh-process branch is already closed"
            )
        body: dict[str, Any] = {
            "kind": "request",
            "seq": self._seq,
            "op": op,
        }
        if action is not None:
            body["action"] = action
        try:
            self._connection.send({
                **body,
                "mac": _fresh_process_mac(self._token, body),
            })
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise ArenaRpcContractError(
                "fresh-process request transport failed"
            ) from exc
        response = self._receive(
            timeout=DEFAULT_SOCKET_TIMEOUT_SECONDS
        )
        if (
            set(response)
            != {"kind", "seq", "ok", "state", "mac"}
            or response.get("kind") != "response"
            or response.get("seq") != self._seq
            or response.get("ok") is not True
        ):
            raise ArenaRpcContractError(
                "fresh-process response schema is invalid"
            )
        self._seq += 1
        return response

    def close(self) -> None:
        if self._closed:
            return
        try:
            response = self._request("close")
            self._state = self._validate_state(response["state"])
        finally:
            self._closed = True
            self._connection.close()
            self._process.join(FRESH_PROCESS_STOP_TIMEOUT_SECONDS)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(FRESH_PROCESS_STOP_TIMEOUT_SECONDS)
            if self._process.is_alive() and hasattr(self._process, "kill"):
                self._process.kill()
                self._process.join(FRESH_PROCESS_STOP_TIMEOUT_SECONDS)
            if self._process.is_alive():
                raise ArenaRpcContractError(
                    "fresh-process probe survived exact teardown"
                )
            self._process.close()

    def terminate(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        try:
            self._connection.close()
        finally:
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(FRESH_PROCESS_STOP_TIMEOUT_SECONDS)
            if self._process.is_alive() and hasattr(self._process, "kill"):
                self._process.kill()
                self._process.join(FRESH_PROCESS_STOP_TIMEOUT_SECONDS)
            if self._process.is_alive():
                raise ArenaRpcContractError(
                    "fresh-process probe survived forced teardown"
                )
            self._process.close()


def _binding_payload(binding: ArenaSessionBinding) -> dict[str, Any]:
    return {
        "schema": RPC_SCHEMA,
        "kind": "arena_session_binding",
        **asdict(binding),
    }


def _validate_binding(
    binding: ArenaSessionBinding,
    *,
    game: str,
) -> str:
    if (
        not isinstance(binding, ArenaSessionBinding)
        or binding.game != game
        or any(
            not isinstance(value, str)
            or _IDENTIFIER_RE.fullmatch(value) is None
            for value in (
                binding.campaign_id,
                binding.generation_id,
                binding.attempt_id,
                binding.game,
            )
        )
        or not _is_plain_int(binding.parent_level)
        or binding.parent_level < 0
        or not _is_plain_int(binding.target_level)
        or binding.target_level != binding.parent_level + 1
        or not _SHA256_RE.fullmatch(
            binding.parent_checkpoint_sha256
        )
        or not _SHA256_RE.fullmatch(binding.frontier_sha256)
        or binding.exploration_mode not in _EXPLORATION_MODES
    ):
        raise ArenaRpcContractError(
            "Arena lineage/frontier binding is malformed"
        )
    return hashlib.sha256(
        _canonical_json(_binding_payload(binding))
    ).hexdigest()


def _wire_mac(token: str, value: Mapping[str, Any]) -> str:
    return hmac.new(
        token.encode("ascii"),
        _canonical_json(value),
        hashlib.sha256,
    ).hexdigest()


class ArenaHostSession:
    """One immutable seed and one controller-selected probe substrate."""

    def __init__(
        self,
        game: str,
        *,
        binding: ArenaSessionBinding,
        parent_path: Sequence[Any],
        arena_factory: Callable[[str], ArenaLike] | None = None,
        token: str | None = None,
        real_step_cap: int = DEFAULT_REAL_STEP_CAP,
        total_step_cap: int = DEFAULT_TOTAL_STEP_CAP,
        reset_cap: int = DEFAULT_RESET_CAP,
    ):
        if not isinstance(game, str) or not game:
            raise ArenaRpcContractError("game must be a nonempty string")
        for label, value in (
            ("real_step_cap", real_step_cap),
            ("total_step_cap", total_step_cap),
            ("reset_cap", reset_cap),
        ):
            if not _is_plain_int(value) or int(value) <= 0:
                raise ArenaRpcContractError(
                    f"{label} must be a positive integer"
                )
        if real_step_cap > total_step_cap:
            raise ArenaRpcContractError(
                "real_step_cap cannot exceed total_step_cap"
            )
        self.binding = binding
        self.binding_sha256 = _validate_binding(
            binding, game=game
        )
        self.game = game
        self.token = secrets.token_hex(32) if token is None else token
        if (
            not isinstance(self.token, str)
            or len(self.token) < 32
            or len(self.token) > 256
            or any(
                ord(character) < 33 or ord(character) > 126
                for character in self.token
            )
        ):
            raise ArenaRpcContractError(
                "session token must be 32..256 printable non-space "
                "ASCII characters"
            )
        self.session_id = _wire_mac(
            self.token,
            {
                "schema": RPC_SCHEMA,
                "kind": "session",
            },
        )
        self.real_step_cap = int(real_step_cap)
        self.total_step_cap = int(total_step_cap)
        self.reset_cap = int(reset_cap)
        normalized_parent = _normalize_path(tuple(parent_path))
        if (
            len(normalized_parent) > self.real_step_cap
            or (
                binding.parent_level == 0
                and normalized_parent
            )
            or (
                binding.parent_level > 0
                and not normalized_parent
            )
        ):
            raise ArenaRpcContractError(
                "parent path is incompatible with the bound frontier"
            )
        if (
            binding.exploration_mode == "fresh_prefix"
            and (
                binding.parent_level == 0
                or len(normalized_parent) != self.real_step_cap
            )
        ):
            raise ArenaRpcContractError(
                "fresh-prefix exploration requires a nonzero parent exactly "
                "at the candidate-path cap"
            )
        if (
            binding.exploration_mode == "fresh_prefix"
            and self.total_step_cap
            < len(normalized_parent) + self.real_step_cap
        ):
            raise ArenaRpcContractError(
                "fresh-prefix total step cap cannot fund one complete "
                "candidate path after parent admission"
            )
        if (
            binding.exploration_mode == "continue_parent"
            and len(normalized_parent) == self.real_step_cap
        ):
            raise ArenaRpcContractError(
                "exhausted parent requires fresh-prefix exploration"
            )
        if arena_factory is None:
            import gkm_arena

            arena_factory = gkm_arena.Arena
        root = arena_factory(game)
        root_path = _normalize_path(tuple(root.path))
        root_levels = root.levels_completed
        if (
            root_path
            or not _is_plain_int(root_levels)
            or int(root_levels) != 0
        ):
            raise ArenaRpcContractError(
                "fresh Arena did not begin at the public zero state"
            )
        zero_root: ArenaLike | None = None
        zero_snapshot: dict[str, Any] | None = None
        if binding.exploration_mode == "fresh_prefix":
            # A fresh-prefix seed must not inherit any clone implementation
            # defect from the separately replayed parent.  It is an
            # independently constructed public-zero Arena.
            zero_root = arena_factory(game)
            if (
                zero_root is root
                or _normalize_path(tuple(zero_root.path))
                or _snapshot_payload(zero_root)
                != _snapshot_payload(root)
            ):
                raise ArenaRpcContractError(
                    "fresh-prefix zero-state clone is not independent and "
                    "exact"
                )
            zero_snapshot = _snapshot_payload(zero_root)
        for index, action in enumerate(normalized_parent):
            if int(root.levels_completed) >= binding.parent_level:
                raise ArenaRpcContractError(
                    "parent checkpoint extends past its exact level boundary"
                )
            action_id = (
                action[0] if isinstance(action, tuple) else action
            )
            if action_id not in _normalize_actions(root.actions):
                raise ArenaRpcContractError(
                    "parent replay requested an unavailable action"
                )
            before = _normalize_path(tuple(root.path))
            if isinstance(action, tuple):
                root.step(action[0], action[1], action[2])
            else:
                root.step(action)
            after = _normalize_path(tuple(root.path))
            if (
                len(after) != len(before) + 1
                or after[:-1] != before
                or after[-1] != action
            ):
                raise ArenaRpcContractError(
                    "parent replay did not append the exact action"
                )
            if (
                index < len(normalized_parent) - 1
                and int(root.levels_completed)
                >= binding.parent_level
            ):
                raise ArenaRpcContractError(
                    "parent path is not an exact first-reach boundary"
                )
        if (
            _normalize_path(tuple(root.path)) != normalized_parent
            or not _is_plain_int(root.levels_completed)
            or int(root.levels_completed) != binding.parent_level
        ):
            raise ArenaRpcContractError(
                "trusted parent replay did not reach the bound parent level"
            )
        self._seeded_root = root
        self._parent_path = normalized_parent
        self._seed_snapshot = _snapshot_payload(root)
        if binding.exploration_mode == "fresh_prefix":
            assert zero_root is not None
            assert zero_snapshot is not None
            self._exploration_seeded_root = zero_root
            self._exploration_seed_path = ()
            self._exploration_seed_snapshot = zero_snapshot
        else:
            self._exploration_seeded_root = root
            self._exploration_seed_path = normalized_parent
            self._exploration_seed_snapshot = self._seed_snapshot
        self._arena_factory = arena_factory
        self._fresh_process_identity_sha256s: list[str] = []
        canary = _clone_isolation_canary(
            arena_factory=arena_factory,
            game=game,
            seed_path=self._exploration_seed_path,
            seed_snapshot=self._exploration_seed_snapshot,
        )
        if (
            canary["mode"]
            == ProbeContract.FRESH_PROCESS_PER_CANDIDATE_MODE
        ):
            try:
                exploration = self._fresh_process_seed()
            except Exception as exc:
                raise ArenaRpcContractError(
                    "independent clone failed and authenticated "
                    "fresh-process-per-candidate probing is unavailable"
                ) from exc
            canary["fallback_process_ready"] = True
            canary["fallback_process_identity_sha256"] = (
                exploration.identity_sha256
            )
        else:
            exploration = self._clone_seed()
        (
            self.probe_isolation_mode,
            self.probe_isolation_evidence_sha256,
        ) = ProbeContract.validate_probe_isolation_evidence(
            canary,
            expected_seed_snapshot_sha256=_json_sha256(
                self._exploration_seed_snapshot
            ),
            expected_seed_path_sha256=_json_sha256([
                list(action) if isinstance(action, tuple) else action
                for action in self._exploration_seed_path
            ]),
        )
        self._probe_isolation_evidence = json.loads(
            _canonical_json(canary)
        )
        self._exploration = exploration
        self._exploration_path = self._exploration_seed_path
        self._closed_exploration_state: dict[str, Any] | None = None
        self._last_seq = -1
        self._opened = False
        self._closed = False
        self._close_delivered = False
        self._parent_replay_steps = len(normalized_parent)
        self._exploration_steps = 0
        self._total_steps = len(normalized_parent)
        self._resets = 0
        self._reward_boundary: dict[str, Any] | None = None
        self._reward_boundary_sha256: str | None = None
        self._action7_checkpoint: dict[str, Any] | None = None
        self._branch_invalidated = False
        self._rollback_reconstructions = 0
        self._public_action_basis_sha256 = (
            PUBLIC_ACTION_BASIS_GENESIS_SHA256
        )
        self._public_action_count = 0
        self._public_observation_receipts: dict[
            str, dict[str, Any]
        ] = {}
        self._public_observation_commitments_by_seq: dict[
            int, dict[str, str]
        ] = {}
        self._lock = threading.RLock()

    def binding_event(self) -> dict[str, Any]:
        """Public retained session binding, excluding the secret token."""

        return {
            **_binding_payload(self.binding),
            "binding_sha256": self.binding_sha256,
            "session_id": self.session_id,
            "parent_path_sha256": hashlib.sha256(
                _canonical_json(list(self._parent_path))
            ).hexdigest(),
            "parent_replay_steps": self._parent_replay_steps,
            "seed_snapshot_sha256": hashlib.sha256(
                _canonical_json(self._seed_snapshot)
            ).hexdigest(),
            "exploration_seed_path_sha256": hashlib.sha256(
                _canonical_json(list(self._exploration_seed_path))
            ).hexdigest(),
            "exploration_seed_snapshot_sha256": hashlib.sha256(
                _canonical_json(self._exploration_seed_snapshot)
            ).hexdigest(),
            "probe_isolation_mode": self.probe_isolation_mode,
            "probe_isolation_evidence":
                self._probe_isolation_evidence,
            "probe_isolation_evidence_sha256":
                self.probe_isolation_evidence_sha256,
            "reward_boundary_policy": REWARD_BOUNDARY_POLICY,
            "action7_rollback_policy": ACTION7_ROLLBACK_POLICY,
            "real_step_cap": self.real_step_cap,
            "total_step_cap": self.total_step_cap,
            "reset_cap": self.reset_cap,
        }

    def _assert_seed_immutable(self) -> None:
        if (
            _normalize_path(tuple(self._seeded_root.path))
            != self._parent_path
            or _snapshot_payload(self._seeded_root)
            != self._seed_snapshot
        ):
            raise ArenaRpcContractError(
                "immutable seeded Arena root changed after admission"
            )
        if (
            _normalize_path(tuple(self._exploration_seeded_root.path))
            != self._exploration_seed_path
            or _snapshot_payload(self._exploration_seeded_root)
            != self._exploration_seed_snapshot
        ):
            raise ArenaRpcContractError(
                "immutable exploration seed changed after admission"
            )

    def _clone_seed(self) -> ArenaLike:
        if (
            getattr(self, "probe_isolation_mode", None)
            == ProbeContract.FRESH_PROCESS_PER_CANDIDATE_MODE
        ):
            raise ArenaRpcContractError(
                "fresh-process mode cannot use an engine clone"
            )
        cloned = self._exploration_seeded_root.clone()
        if cloned is self._exploration_seeded_root:
            raise ArenaRpcContractError(
                "Arena exploration clone is not independent"
            )
        if (
            _normalize_path(tuple(cloned.path))
            != self._exploration_seed_path
            or _snapshot_payload(cloned)
            != self._exploration_seed_snapshot
        ):
            raise ArenaRpcContractError(
                "Arena exploration clone differs from the seeded root"
            )
        self._assert_seed_immutable()
        return cloned

    def _fresh_process_seed(self) -> _FreshProcessArena:
        fresh = _FreshProcessArena(
            game=self.game,
            arena_factory=self._arena_factory,
            seed_path=self._exploration_seed_path,
            seed_snapshot=self._exploration_seed_snapshot,
        )
        self._fresh_process_identity_sha256s.append(
            fresh.identity_sha256
        )
        if len(set(self._fresh_process_identity_sha256s)) != len(
            self._fresh_process_identity_sha256s
        ):
            fresh.terminate()
            raise ArenaRpcContractError(
                "fresh-process probe identity was reused"
            )
        return fresh

    def _replace_exploration(self) -> ArenaLike:
        if (
            self.probe_isolation_mode
            == ProbeContract.VERIFIED_ISOLATED_CLONE_MODE
        ):
            return self._clone_seed()
        prior = self._exploration
        assert isinstance(prior, _FreshProcessArena)
        prior.close()
        return self._fresh_process_seed()

    def _reconstruct_action7_branch(
        self,
        checkpoint: Mapping[str, Any],
    ) -> None:
        """Rebuild the pre-branch state only from the authenticated seed."""

        path = checkpoint.get("path")
        projection = checkpoint.get("projection")
        if (
            not isinstance(path, tuple)
            or not isinstance(projection, dict)
            or path[: len(self._exploration_seed_path)]
            != self._exploration_seed_path
        ):
            raise ArenaRpcContractError(
                "ACTION7 rollback checkpoint is malformed"
            )
        self._exploration = self._replace_exploration()
        self._exploration_path = self._exploration_seed_path
        for action in path[len(self._exploration_seed_path):]:
            _apply_exact_action(self._exploration, action)
            self._exploration_path = (
                self._exploration_path + (action,)
            )
        reconstructed = _snapshot_payload(self._exploration)
        if (
            _normalize_path(tuple(self._exploration.path)) != path
            or self._exploration_path != path
            or _rollback_projection(reconstructed) != projection
        ):
            raise ArenaRpcContractError(
                "ACTION7 branch reconstruction differs from its "
                "authenticated checkpoint"
            )
        self._rollback_reconstructions += 1

    def _close_exploration(self) -> None:
        if (
            self.probe_isolation_mode
            != ProbeContract.FRESH_PROCESS_PER_CANDIDATE_MODE
            or not isinstance(self._exploration, _FreshProcessArena)
        ):
            return
        self._closed_exploration_state = _public_state_payload(
            self._exploration
        )
        self._exploration.close()

    def _discard_resources(self) -> None:
        """Best-effort exact child teardown on protocol-invalid transport."""

        with self._lock:
            if isinstance(
                getattr(self, "_exploration", None),
                _FreshProcessArena,
            ):
                self._exploration.terminate()

    def _authenticate(self, request: Mapping[str, Any]) -> int:
        if (
            request.get("schema") != RPC_SCHEMA
            or request.get("session") != self.session_id
            or not isinstance(request.get("mac"), str)
            or _SHA256_RE.fullmatch(request["mac"]) is None
        ):
            raise ArenaRpcContractError("RPC authentication failed")
        unsigned = dict(request)
        observed_mac = unsigned.pop("mac")
        expected_mac = _wire_mac(self.token, unsigned)
        if not hmac.compare_digest(observed_mac, expected_mac):
            raise ArenaRpcContractError("RPC authentication failed")
        seq = request.get("seq")
        if not _is_plain_int(seq) or int(seq) < 0:
            raise ArenaRpcContractError(
                "RPC sequence must be a nonnegative integer"
            )
        if int(seq) != self._last_seq + 1:
            raise ArenaRpcContractError(
                "RPC sequence is stale or non-contiguous"
            )
        self._last_seq = int(seq)
        return int(seq)

    def signed_response(
        self,
        *,
        seq: int,
        ok: bool,
        result: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        if ok:
            body: dict[str, Any] = {
                "schema": RPC_SCHEMA,
                "session": self.session_id,
                "seq": seq,
                "ok": True,
                "result": dict(result or {}),
            }
        else:
            body = {
                "schema": RPC_SCHEMA,
                "session": self.session_id,
                "seq": seq,
                "ok": False,
                "error": (
                    error
                    if isinstance(error, str)
                    else "Arena RPC request failed"
                ),
            }
        return {**body, "mac": _wire_mac(self.token, body)}

    def _record_public_observation(
        self,
        *,
        seq: int,
        operation: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> None:
        """Commit public semantics, excluding session/attempt metadata."""

        canonical_operation = json.loads(
            _canonical_json(dict(operation))
        )
        basis = {
            "schema": PUBLIC_OBSERVATION_RECEIPT_SCHEMA,
            "kind": PUBLIC_ACTION_BASIS_KIND,
            "operation_index": self._public_action_count,
            "previous_public_action_basis_sha256": (
                self._public_action_basis_sha256
            ),
            "operation": canonical_operation,
        }
        signature = {
            "schema": PUBLIC_OBSERVATION_RECEIPT_SCHEMA,
            "kind": PUBLIC_RESPONSE_SIGNATURE_KIND,
            "operation_index": self._public_action_count,
            "result": json.loads(_canonical_json(dict(result))),
        }
        receipt = {
            "schema": PUBLIC_OBSERVATION_RECEIPT_SCHEMA,
            "kind": PUBLIC_OBSERVATION_RECEIPT_KIND,
            "game": self.game,
            "frontier_sha256": self.binding.frontier_sha256,
            "parent_checkpoint_sha256": (
                self.binding.parent_checkpoint_sha256
            ),
            "public_action_basis": basis,
            "public_action_basis_sha256": _json_sha256(basis),
            "public_response_signature": signature,
            "public_response_signature_sha256": _json_sha256(signature),
        }
        receipt_sha256 = validate_public_observation_receipt(
            receipt,
            game=self.game,
            frontier_sha256=self.binding.frontier_sha256,
            parent_checkpoint_sha256=(
                self.binding.parent_checkpoint_sha256
            ),
        )
        self._public_action_basis_sha256 = (
            receipt["public_action_basis_sha256"]
        )
        self._public_action_count += 1
        self._public_observation_receipts.setdefault(
            receipt_sha256, receipt
        )
        self._public_observation_commitments_by_seq[seq] = {
            "public_observation_receipt_sha256": receipt_sha256,
            "public_action_basis_sha256": (
                receipt["public_action_basis_sha256"]
            ),
            "public_response_signature_sha256": (
                receipt["public_response_signature_sha256"]
            ),
        }

    def public_observation_commitment(
        self, seq: int
    ) -> dict[str, str] | None:
        """Return the host commitment logged for one successful response."""

        with self._lock:
            value = self._public_observation_commitments_by_seq.get(seq)
            return None if value is None else dict(value)

    def public_observation_receipts(
        self,
    ) -> tuple[dict[str, Any], ...]:
        """Return canonical content receipts, sorted by semantic SHA-256."""

        with self._lock:
            return tuple(
                json.loads(_canonical_json(value))
                for _digest, value in sorted(
                    self._public_observation_receipts.items()
                )
            )

    def dispatch(self, value: object) -> dict[str, Any]:
        """Validate and execute one HMAC-authenticated public operation."""

        with self._lock:
            request = _strict_object(
                value,
                required={
                    "schema",
                    "session",
                    "seq",
                    "op",
                    "mac",
                },
                optional={"action"},
                label="RPC request",
            )
            seq = self._authenticate(request)
            op = request["op"]
            if not isinstance(op, str):
                raise ArenaRpcContractError("RPC op must be a string")
            if self._closed:
                raise ArenaRpcContractError("Arena session is closed")
            self._assert_seed_immutable()

            if op == "open":
                _strict_object(
                    request,
                    required={
                        "schema",
                        "session",
                        "seq",
                        "op",
                        "mac",
                    },
                    label="open request",
                )
                if self._opened:
                    raise ArenaRpcContractError(
                        "Arena session is already open"
                    )
                self._opened = True
                result = {
                    "binding_sha256": self.binding_sha256,
                    "snapshot": _snapshot_payload(
                        self._exploration
                    ),
                }
            else:
                if not self._opened:
                    raise ArenaRpcContractError(
                        "Arena session has not been opened"
                    )
                if (
                    self._reward_boundary is not None
                    and op != "close"
                ):
                    raise ArenaRpcContractError(
                        "reward boundary is sealed; only close is permitted"
                    )
                if self._branch_invalidated and op != "close":
                    raise ArenaRpcContractError(
                        "exploration branch is invalidated; only close is "
                        "permitted"
                    )
                if op == "observe":
                    _strict_object(
                        request,
                        required={
                            "schema",
                            "session",
                            "seq",
                            "op",
                            "mac",
                        },
                        label="observe request",
                    )
                    if (
                        _normalize_path(tuple(self._exploration.path))
                        != self._exploration_path
                    ):
                        raise ArenaRpcContractError(
                            "exploration clone changed outside authenticated "
                            "RPC steps"
                        )
                    result = {
                        "snapshot": _snapshot_payload(
                            self._exploration
                        )
                    }
                elif op == "reset":
                    _strict_object(
                        request,
                        required={
                            "schema",
                            "session",
                            "seq",
                            "op",
                            "mac",
                        },
                        label="reset request",
                    )
                    if self._resets >= self.reset_cap:
                        raise ArenaRpcContractError(
                            "Arena exploration reset budget exhausted"
                        )
                    # Never call reset() on a root or exploration object.  The
                    # controller replaces the whole branch using only the
                    # pre-launch selected substrate.
                    self._exploration = self._replace_exploration()
                    self._exploration_path = (
                        self._exploration_seed_path
                    )
                    self._action7_checkpoint = None
                    self._resets += 1
                    result = {
                        "snapshot": _snapshot_payload(
                            self._exploration
                        )
                    }
                elif op == "step":
                    _strict_object(
                        request,
                        required={
                            "schema",
                            "session",
                            "seq",
                            "op",
                            "mac",
                            "action",
                        },
                        label="step request",
                    )
                    action = _normalize_action(request["action"])
                    action_id = (
                        action[0]
                        if isinstance(action, tuple)
                        else action
                    )
                    if action_id not in _normalize_actions(
                        self._exploration.actions
                    ):
                        raise ArenaRpcContractError(
                            "requested action is not currently available"
                        )
                    before = _normalize_path(
                        tuple(self._exploration.path)
                    )
                    if before != self._exploration_path:
                        raise ArenaRpcContractError(
                            "exploration clone changed outside authenticated "
                            "RPC steps"
                        )
                    if len(before) >= self.real_step_cap:
                        raise ArenaRpcContractError(
                            "600-action candidate-path budget exhausted"
                        )
                    if self._total_steps >= self.total_step_cap:
                        raise ArenaRpcContractError(
                            "total exploration step budget exhausted"
                        )
                    levels_before = self._exploration.levels_completed
                    if (
                        not _is_plain_int(levels_before)
                        or int(levels_before)
                        > self.binding.parent_level
                    ):
                        raise ArenaRpcContractError(
                            "Arena pre-step level is outside the exact "
                            "reward frontier"
                        )
                    snapshot_before = _snapshot_payload(
                        self._exploration
                    )
                    action7_checkpoint = self._action7_checkpoint
                    if action_id == 7:
                        if action7_checkpoint is None:
                            self._branch_invalidated = True
                            raise ArenaRpcContractError(
                                "ACTION7 has no immediate authenticated "
                                "rollback context"
                            )
                    else:
                        action7_checkpoint = {
                            "path": before,
                            "projection": _rollback_projection(
                                snapshot_before
                            ),
                        }
                    if isinstance(action, tuple):
                        self._exploration.step(
                            action[0], action[1], action[2]
                        )
                    else:
                        self._exploration.step(action)
                    after = _normalize_path(
                        tuple(self._exploration.path)
                    )
                    if (
                        len(after) != len(before) + 1
                        or after[:-1] != before
                        or after[-1] != action
                    ):
                        raise ArenaRpcContractError(
                            "Arena step did not append exactly the requested "
                            "action"
                        )
                    self._exploration_path = after
                    self._exploration_steps += 1
                    self._total_steps += 1
                    snapshot = _snapshot_payload(
                        self._exploration
                    )
                    if action_id == 7:
                        assert action7_checkpoint is not None
                        if (
                            _rollback_projection(snapshot)
                            != action7_checkpoint["projection"]
                        ):
                            self._branch_invalidated = True
                            try:
                                self._reconstruct_action7_branch(
                                    action7_checkpoint
                                )
                            finally:
                                self._action7_checkpoint = None
                            raise ArenaRpcContractError(
                                "ACTION7 failed exact frame/level/terminal "
                                "rollback; exploration branch was "
                                "invalidated and reconstructed"
                            )
                        self._action7_checkpoint = None
                    else:
                        self._action7_checkpoint = action7_checkpoint
                    levels_after = snapshot["levels_completed"]
                    if levels_after > self.binding.target_level:
                        raise ArenaRpcContractError(
                            "Arena crossed beyond the exact target reward"
                        )
                    if levels_after == self.binding.target_level:
                        if (
                            int(levels_before)
                            != self.binding.parent_level
                        ):
                            raise ArenaRpcContractError(
                                "target reward did not cross exact K to K+1"
                            )
                        boundary = _reward_boundary_payload(
                            binding=self.binding,
                            binding_sha256=self.binding_sha256,
                            path=after,
                            snapshot=snapshot,
                        )
                        self._reward_boundary = boundary
                        self._reward_boundary_sha256 = _json_sha256(
                            boundary
                        )
                    result = {
                        "snapshot": snapshot
                    }
                elif op == "close":
                    _strict_object(
                        request,
                        required={
                            "schema",
                            "session",
                            "seq",
                            "op",
                            "mac",
                        },
                        label="close request",
                    )
                    self._close_exploration()
                    self._closed = True
                    result = {"closed": True}
                else:
                    raise ArenaRpcContractError(
                        "unknown Arena RPC operation"
                    )
            if op != "close":
                operation: dict[str, Any] = {"op": op}
                if op == "step":
                    normalized_action = _normalize_action(
                        request["action"]
                    )
                    operation["action"] = (
                        list(normalized_action)
                        if isinstance(normalized_action, tuple)
                        else normalized_action
                    )
                self._record_public_observation(
                    seq=seq,
                    operation=operation,
                    result=result,
                )
            return self.signed_response(
                seq=seq, ok=True, result=result
            )

    def _mark_close_delivered(self, seq: int) -> None:
        """Bind host eligibility to a durably logged close response."""

        with self._lock:
            if (
                not self._opened
                or not self._closed
                or self._close_delivered
                or not _is_plain_int(seq)
                or seq != self._last_seq
            ):
                raise ArenaRpcContractError(
                    "RPC close delivery state is inconsistent"
                )
            self._close_delivered = True

    def host_result(self) -> ArenaHostResult:
        """Return trusted accounting, never engine objects, to the host."""

        with self._lock:
            if (
                not self._opened
                or not self._closed
                or not self._close_delivered
            ):
                raise ArenaRpcContractError(
                    "host result requires an authenticated clean session close"
                )
            self._assert_seed_immutable()
            levels = self._exploration.levels_completed
            path = _normalize_path(tuple(self._exploration.path))
            snapshot = _snapshot_payload(self._exploration)
            continue_parent = (
                self.binding.exploration_mode == "continue_parent"
            )
            reward_boundary = self._reward_boundary
            expected_reward_boundary = (
                _reward_boundary_payload(
                    binding=self.binding,
                    binding_sha256=self.binding_sha256,
                    path=path,
                    snapshot=snapshot,
                )
                if (
                    _is_plain_int(levels)
                    and int(levels) == self.binding.target_level
                )
                else None
            )
            if (
                self._branch_invalidated
                or
                not _is_plain_int(levels)
                or int(levels) > self.binding.target_level
                or (
                    continue_parent
                    and int(levels) < self.binding.parent_level
                )
                or path != self._exploration_path
                or (
                    continue_parent
                    and path[: len(self._parent_path)]
                    != self._parent_path
                )
                or self._parent_replay_steps
                != len(self._parent_path)
                or self._total_steps
                != (
                    self._parent_replay_steps
                    + self._exploration_steps
                )
                or reward_boundary != expected_reward_boundary
                or (
                    reward_boundary is None
                    and self._reward_boundary_sha256 is not None
                )
                or (
                    reward_boundary is not None
                    and self._reward_boundary_sha256
                    != _json_sha256(reward_boundary)
                )
            ):
                raise ArenaRpcContractError(
                    "trusted Arena lineage or step accounting disagrees"
                )
            return ArenaHostResult(
                binding_sha256=self.binding_sha256,
                game=self.game,
                exploration_mode=self.binding.exploration_mode,
                parent_level=self.binding.parent_level,
                levels_completed=int(levels),
                parent_path=self._parent_path,
                path=path,
                parent_replay_steps=self._parent_replay_steps,
                exploration_steps=self._exploration_steps,
                resets=self._resets,
                total_steps=self._total_steps,
                parent_terminal=bool(self._seed_snapshot["terminal"]),
                parent_snapshot_sha256=hashlib.sha256(
                    _canonical_json(self._seed_snapshot)
                ).hexdigest(),
            )


def _reject_aliased_path(path: Path, *, label: str, must_exist: bool) -> None:
    """Reject symlinked ancestors and nonregular/aliased final files."""

    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:-1]:
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            if must_exist:
                raise ArenaRpcContractError(f"{label} ancestor is missing: {current}")
            continue
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ArenaRpcContractError(
                f"{label} has a symlinked or non-directory ancestor"
            )
    try:
        metadata = os.lstat(absolute)
    except FileNotFoundError:
        if must_exist:
            raise ArenaRpcContractError(f"{label} is missing")
        return
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise ArenaRpcContractError(
            f"{label} must be an unaliased regular file"
        )


class HostTranscript:
    """Exclusive, no-follow host transcript not mounted into the container."""

    def __init__(
        self,
        path: Path,
        *,
        forbidden_values: Sequence[str] = (),
    ):
        self.path = path
        self._forbidden_values = tuple(
            value for value in forbidden_values if isinstance(value, str) and value
        )
        _reject_aliased_path(path, label="host transcript", must_exist=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Missing ancestors may have been created above; validate the complete
        # lexical chain again before opening the parent directory descriptor.
        _reject_aliased_path(path, label="host transcript", must_exist=False)
        try:
            parent_metadata = os.lstat(path.parent)
        except OSError as exc:
            raise ArenaRpcContractError(
                "host transcript parent is unavailable"
            ) from exc
        if (
            stat.S_ISLNK(parent_metadata.st_mode)
            or not stat.S_ISDIR(parent_metadata.st_mode)
        ):
            raise ArenaRpcContractError(
                "host transcript parent must be a regular directory"
            )
        parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        parent_flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            parent_descriptor = os.open(path.parent, parent_flags)
        except OSError as exc:
            raise ArenaRpcContractError(
                "host transcript parent cannot be opened safely"
            ) from exc
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(
                path.name, flags, 0o600, dir_fd=parent_descriptor
            )
        except OSError as exc:
            os.close(parent_descriptor)
            raise ArenaRpcContractError(
                "host transcript must be created exclusively"
            ) from exc
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            os.close(descriptor)
            os.close(parent_descriptor)
            raise ArenaRpcContractError(
                "host transcript must be an unaliased regular file"
            )
        self._file = os.fdopen(descriptor, "w", encoding="utf-8")
        self._parent_descriptor = parent_descriptor
        self._parent_identity = (
            parent_metadata.st_dev,
            parent_metadata.st_ino,
        )
        self._identity = (metadata.st_dev, metadata.st_ino)
        self._lock = threading.Lock()

    def _contains_forbidden_value(self, value: object) -> bool:
        if isinstance(value, str):
            return any(secret in value for secret in self._forbidden_values)
        if isinstance(value, Mapping):
            return any(
                self._contains_forbidden_value(key)
                or self._contains_forbidden_value(item)
                for key, item in value.items()
            )
        if isinstance(value, (list, tuple)):
            return any(self._contains_forbidden_value(item) for item in value)
        return False

    def _validate_custody(self, *, phase: str) -> None:
        metadata = os.fstat(self._file.fileno())
        try:
            parent_metadata = os.lstat(self.path.parent)
        except OSError:
            parent_metadata = None
        try:
            path_metadata = os.stat(
                self.path.name,
                dir_fd=self._parent_descriptor,
                follow_symlinks=False,
            )
        except OSError:
            path_metadata = None
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or (metadata.st_dev, metadata.st_ino) != self._identity
            or parent_metadata is None
            or stat.S_ISLNK(parent_metadata.st_mode)
            or not stat.S_ISDIR(parent_metadata.st_mode)
            or (parent_metadata.st_dev, parent_metadata.st_ino)
            != self._parent_identity
            or path_metadata is None
            or stat.S_ISLNK(path_metadata.st_mode)
            or not stat.S_ISREG(path_metadata.st_mode)
            or path_metadata.st_nlink != 1
            or (path_metadata.st_dev, path_metadata.st_ino)
            != self._identity
        ):
            raise ArenaRpcContractError(
                f"host transcript custody was lost before {phase}"
            )

    def append(self, event: Mapping[str, Any]) -> None:
        if any(
            isinstance(key, str)
            and ("token" in key.lower() or "secret" in key.lower())
            for key in event
        ):
            raise ArenaRpcContractError(
                "secret-bearing fields are forbidden in the host transcript"
            )
        if self._contains_forbidden_value(event):
            raise ArenaRpcContractError(
                "a forbidden value was rejected from the host transcript"
            )
        encoded = json.dumps(
            event,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        with self._lock:
            self._validate_custody(phase="append")
            self._file.write(encoded + "\n")
            self._file.flush()
            os.fsync(self._file.fileno())

    def close(self) -> None:
        with self._lock:
            if self._file.closed:
                return
            custody_error: ArenaRpcContractError | None = None
            try:
                self._file.flush()
                os.fsync(self._file.fileno())
                try:
                    self._validate_custody(phase="close")
                except ArenaRpcContractError as exc:
                    custody_error = exc
                open_metadata = os.fstat(self._file.fileno())
                try:
                    path_metadata = os.stat(
                        self.path.name,
                        dir_fd=self._parent_descriptor,
                        follow_symlinks=False,
                    )
                except OSError:
                    path_metadata = None
                if custody_error is None and (
                    not stat.S_ISREG(open_metadata.st_mode)
                    or open_metadata.st_nlink != 1
                    or (open_metadata.st_dev, open_metadata.st_ino)
                    != self._identity
                    or path_metadata is None
                    or stat.S_ISLNK(path_metadata.st_mode)
                    or not stat.S_ISREG(path_metadata.st_mode)
                    or path_metadata.st_nlink != 1
                    or (path_metadata.st_dev, path_metadata.st_ino)
                    != self._identity
                ):
                    custody_error = ArenaRpcContractError(
                        "host transcript custody was lost before close"
                    )
            finally:
                self._file.close()
                os.close(self._parent_descriptor)
            if custody_error is not None:
                raise custody_error

    def __enter__(self) -> "HostTranscript":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _recv_line(connection: socket.socket) -> bytes | None:
    data = bytearray()
    while True:
        chunk = connection.recv(min(65_536, MAX_MESSAGE_BYTES + 1 - len(data)))
        if not chunk:
            if data:
                raise ArenaRpcContractError(
                    "RPC message ended without a newline frame terminator"
                )
            return None
        newline = chunk.find(b"\n")
        if newline >= 0:
            data.extend(chunk[:newline])
            if chunk[newline + 1 :]:
                raise ArenaRpcContractError(
                    "multiple or pipelined RPC messages are forbidden"
                )
            return bytes(data)
        data.extend(chunk)
        if len(data) > MAX_MESSAGE_BYTES:
            raise ArenaRpcContractError("RPC message exceeds byte limit")


def _send_json(connection: socket.socket, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    if len(encoded) > MAX_MESSAGE_BYTES:
        raise ArenaRpcContractError("RPC response exceeds byte limit")
    connection.sendall(encoded)


class ArenaRpcServer:
    """Single-client Unix-socket server for one attempt generation."""

    def __init__(
        self,
        session: ArenaHostSession,
        socket_path: Path,
        transcript_path: Path,
    ):
        self.session = session
        self.socket_path = socket_path
        self.transcript_path = transcript_path
        self._listener: socket.socket | None = None
        self._started = threading.Event()
        self._finished = threading.Event()
        self._error: BaseException | None = None
        self._socket_identity: tuple[int, int] | None = None
        self._connection: socket.socket | None = None
        self._state_lock = threading.Lock()
        self._shutdown_requested = threading.Event()
        self._protocol_violation: dict[str, Any] | None = None
        self._protocol_violation_callback: (
            Callable[[Mapping[str, Any]], None] | None
        ) = None

    def set_protocol_violation_callback(
        self,
        callback: Callable[[Mapping[str, Any]], None],
    ) -> None:
        """Install the one host-only fail-closed containment callback.

        The callback is deliberately not a request handler.  It receives only
        the already-sanitized rejected-event projection, after that projection
        is durable in the host transcript and before an error response can be
        delivered to code that might catch it.
        """

        if (
            not callable(callback)
            or self._started.is_set()
            or self._protocol_violation_callback is not None
        ):
            raise ArenaRpcContractError(
                "protocol-violation callback must be installed exactly once "
                "before server start"
            )
        self._protocol_violation_callback = callback

    @property
    def protocol_violation(self) -> Mapping[str, Any] | None:
        with self._state_lock:
            return (
                None
                if self._protocol_violation is None
                else dict(self._protocol_violation)
            )

    def _invalidate_protocol(
        self, event: Mapping[str, Any]
    ) -> None:
        """Invalidate this one-shot session and invoke role containment."""

        frozen = dict(event)
        callback: Callable[[Mapping[str, Any]], None] | None
        with self._state_lock:
            if self._protocol_violation is not None:
                raise ArenaRpcContractError(
                    "Arena RPC session received more than one rejected frame"
                )
            self._protocol_violation = frozen
            callback = self._protocol_violation_callback
        if callback is not None:
            callback(frozen)

    def _remove_owned_socket(self) -> None:
        try:
            metadata = os.lstat(self.socket_path)
        except FileNotFoundError:
            if self._socket_identity is not None:
                self._socket_identity = None
                raise ArenaRpcContractError(
                    "RPC socket path disappeared before owned cleanup"
                )
            self._socket_identity = None
            return
        if (
            self._socket_identity is None
            or not stat.S_ISSOCK(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != self._socket_identity
        ):
            raise ArenaRpcContractError(
                "RPC socket path was replaced; refusing unsafe unlink"
            )
        os.unlink(self.socket_path)
        self._socket_identity = None

    def _prepare_socket(self) -> socket.socket:
        parent = self.socket_path.parent
        parent.mkdir(parents=True, exist_ok=True)
        try:
            parent_before = os.lstat(parent)
        except OSError as exc:
            raise ArenaRpcContractError(
                "RPC socket parent is unavailable"
            ) from exc
        if (
            stat.S_ISLNK(parent_before.st_mode)
            or not stat.S_ISDIR(parent_before.st_mode)
        ):
            raise ArenaRpcContractError(
                "RPC socket parent must be a regular directory"
            )
        _reject_aliased_path(
            self.socket_path, label="RPC socket", must_exist=False
        )
        if self.socket_path.exists():
            raise ArenaRpcContractError("RPC socket path already exists")
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(self.socket_path))
            socket_metadata = os.lstat(self.socket_path)
            if (
                not stat.S_ISSOCK(socket_metadata.st_mode)
                or socket_metadata.st_nlink != 1
            ):
                raise ArenaRpcContractError(
                    "RPC bind did not create an unaliased Unix socket"
                )
            self._socket_identity = (
                socket_metadata.st_dev,
                socket_metadata.st_ino,
            )
            os.chmod(self.socket_path, 0o600)
            after_chmod = os.lstat(self.socket_path)
            parent_after = os.lstat(parent)
            if (
                not stat.S_ISSOCK(after_chmod.st_mode)
                or (after_chmod.st_dev, after_chmod.st_ino)
                != self._socket_identity
                or (parent_after.st_dev, parent_after.st_ino)
                != (parent_before.st_dev, parent_before.st_ino)
                or stat.S_IMODE(after_chmod.st_mode) != 0o600
            ):
                raise ArenaRpcContractError(
                    "RPC socket or parent changed during admission"
                )
            listener.listen(1)
        except BaseException:
            listener.close()
            try:
                self._remove_owned_socket()
            except ArenaRpcContractError:
                # Never turn a detected replacement race into an unlink of
                # attacker-controlled bytes.
                pass
            raise
        return listener

    def serve_once(self) -> None:
        """Serve exactly one transport connection, then destroy the socket."""

        transcript: HostTranscript | None = None
        try:
            transcript = HostTranscript(
                self.transcript_path,
                forbidden_values=(self.session.token,),
            )
            transcript.append(self.session.binding_event())
            self._listener = self._prepare_socket()
            self._listener.settimeout(0.25)
            self._started.set()
            while True:
                if self._shutdown_requested.is_set():
                    return
                try:
                    connection, _address = self._listener.accept()
                    break
                except socket.timeout:
                    continue
            # Closing the listener immediately makes a second Arena client fail.
            self._listener.close()
            self._listener = None
            with self._state_lock:
                self._connection = connection
            if self._shutdown_requested.is_set():
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            with connection:
                while True:
                    raw: bytes | None = None
                    request: object = None
                    digest: str | None = None
                    try:
                        raw = _recv_line(connection)
                        if raw is None:
                            break
                        if len(raw) > MAX_MESSAGE_BYTES:
                            raise ArenaRpcContractError(
                                "RPC message exceeds byte limit"
                            )
                        digest = hashlib.sha256(raw).hexdigest()
                        request = _loads_json(raw, label="RPC request")
                        started = time.monotonic_ns()
                        response = self.session.dispatch(request)
                        applied_event = {
                            "schema": RPC_SCHEMA,
                            "kind": "rpc",
                            "phase": "applied",
                            "seq": response["seq"],
                            "op": _safe_operation_label(request.get("op"))
                            if isinstance(request, dict)
                            else None,
                            "request_sha256": digest,
                            "ok": True,
                            "elapsed_ns": time.monotonic_ns() - started,
                        }
                        commitment = (
                            self.session.public_observation_commitment(
                                response["seq"]
                            )
                        )
                        if commitment is not None:
                            applied_event.update(commitment)
                        transcript.append(applied_event)
                        try:
                            _send_json(connection, response)
                        except BaseException:
                            transcript.append({
                                "schema": RPC_SCHEMA,
                                "kind": "rpc_delivery",
                                "seq": response["seq"],
                                "delivered": False,
                            })
                            raise
                        transcript.append({
                            "schema": RPC_SCHEMA,
                            "kind": "rpc_delivery",
                            "seq": response["seq"],
                            "delivered": True,
                        })
                        if (
                            isinstance(request, dict)
                            and request.get("op") == "close"
                        ):
                            self.session._mark_close_delivered(
                                response["seq"]
                            )
                            break
                    except ArenaRpcContractError as exc:
                        # Do not expose engine exceptions, paths, or tracebacks.
                        seq = (
                            request.get("seq")
                            if isinstance(request, dict)
                            and _is_plain_int(request.get("seq"))
                            else -1
                        )
                        event: dict[str, Any] = {
                            "schema": RPC_SCHEMA,
                            "kind": "rpc",
                            "phase": "rejected",
                            "seq": seq,
                            "op": (
                                _safe_operation_label(request.get("op"))
                                if isinstance(request, dict)
                                else None
                            ),
                            "ok": False,
                            "error": str(exc),
                        }
                        if digest is not None:
                            event["request_sha256"] = digest
                        transcript.append(event)
                        # A rejected action/auth/framing request is terminal.
                        # Invalidate and contain before returning an error that
                        # untrusted solver code could catch and continue past.
                        self._invalidate_protocol(event)
                        _send_json(
                            connection,
                            self.session.signed_response(
                                seq=seq,
                                ok=False,
                                error=str(exc),
                            ),
                        )
                        break
        except BaseException as exc:
            if not (
                self._shutdown_requested.is_set()
                and isinstance(exc, OSError)
            ):
                self._error = exc
            self._started.set()
        finally:
            with self._state_lock:
                self._connection = None
            if self._listener is not None:
                self._listener.close()
                self._listener = None
            try:
                self._remove_owned_socket()
            except BaseException as exc:
                if self._error is None:
                    self._error = exc
            if transcript is not None:
                try:
                    transcript.close()
                except BaseException as exc:
                    if self._error is None:
                        self._error = exc
            try:
                self.session._discard_resources()
            except BaseException as exc:
                if self._error is None:
                    self._error = exc
            self._finished.set()

    def shutdown(self) -> None:
        """Interrupt accept/receive without adopting an incomplete session."""

        self._shutdown_requested.set()
        with self._state_lock:
            connection = self._connection
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def start_thread(self) -> threading.Thread:
        thread = threading.Thread(
            target=self.serve_once,
            name=f"arena-rpc-{self.game_label}",
            daemon=True,
        )
        thread.start()
        if not self._started.wait(timeout=10):
            self.shutdown()
            raise ArenaRpcError("Arena RPC server did not start")
        if self._error is not None:
            raise ArenaRpcError("Arena RPC server failed during startup") from self._error
        return thread

    @property
    def game_label(self) -> str:
        return "".join(
            character if character.isalnum() else "_"
            for character in self.session.game
        )[:64]

    def wait(self, timeout: float | None = None) -> None:
        if not self._finished.wait(timeout):
            raise TimeoutError("Arena RPC server has not finished")
        if self._error is not None:
            raise ArenaRpcError("Arena RPC server failed") from self._error


class ArenaRpcClient:
    """Container-side HMAC transport for one default exploration clone."""

    def __init__(self, socket_path: str | os.PathLike[str], token: str):
        if (
            not isinstance(token, str)
            or not 32 <= len(token) <= 256
            or any(
                ord(character) < 33 or ord(character) > 126
                for character in token
            )
        ):
            raise ArenaRpcError(
                "RPC token must be 32..256 printable non-space ASCII "
                "characters"
            )
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.settimeout(DEFAULT_SOCKET_TIMEOUT_SECONDS)
        self._token = token
        self._session_id = _wire_mac(
            token,
            {"schema": RPC_SCHEMA, "kind": "session"},
        )
        self._seq = 0
        self._lock = threading.RLock()
        self._closed = False
        try:
            self._socket.connect(os.fspath(socket_path))
            opened = self._call("open")
            opened = _strict_object(
                opened,
                required={"binding_sha256", "snapshot"},
                label="Arena open result",
            )
            if (
                not isinstance(opened["binding_sha256"], str)
                or _SHA256_RE.fullmatch(
                    opened["binding_sha256"]
                )
                is None
            ):
                raise ArenaRpcError(
                    "Arena open result lacks a valid binding"
                )
            self.binding_sha256 = opened["binding_sha256"]
            self.root = RemoteArena(self, opened["snapshot"])
        except BaseException:
            self._closed = True
            self._token = ""
            self._socket.close()
            raise

    def _call(self, op: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise ArenaRpcError("Arena RPC client is closed")
            if op not in _PUBLIC_OPERATIONS:
                raise ArenaRpcError(
                    "unknown local Arena RPC operation"
                )
            try:
                seq = self._seq
                unsigned = {
                    "schema": RPC_SCHEMA,
                    "session": self._session_id,
                    "seq": seq,
                    "op": op,
                    **fields,
                }
                request = {
                    **unsigned,
                    "mac": _wire_mac(self._token, unsigned),
                }
                _send_json(self._socket, request)
                raw = _recv_line(self._socket)
                if raw is None:
                    raise ArenaRpcError(
                        "Arena RPC server closed without a response"
                    )
                response = _loads_json(raw, label="RPC response")
                response = _strict_object(
                    response,
                    required={
                        "schema",
                        "session",
                        "seq",
                        "ok",
                        "mac",
                    },
                    optional={"result", "error"},
                    label="RPC response",
                )
                observed_mac = response["mac"]
                unsigned_response = dict(response)
                unsigned_response.pop("mac")
                if (
                    response["schema"] != RPC_SCHEMA
                    or response["session"] != self._session_id
                    or not _is_plain_int(response["seq"])
                    or response["seq"] != seq
                    or not isinstance(response["ok"], bool)
                    or not isinstance(observed_mac, str)
                    or _SHA256_RE.fullmatch(observed_mac) is None
                    or not hmac.compare_digest(
                        observed_mac,
                        _wire_mac(self._token, unsigned_response),
                    )
                ):
                    raise ArenaRpcError(
                        "Arena RPC response identity or HMAC mismatch"
                    )
                self._seq += 1
                if response["ok"] is False:
                    if set(response) != {
                        "schema",
                        "session",
                        "seq",
                        "ok",
                        "error",
                        "mac",
                    }:
                        raise ArenaRpcError(
                            "malformed Arena error response"
                        )
                    error = response.get("error")
                    raise ArenaRpcError(
                        error
                        if isinstance(error, str)
                        else "Arena RPC request failed"
                    )
                if set(response) != {
                    "schema",
                    "session",
                    "seq",
                    "ok",
                    "result",
                    "mac",
                }:
                    raise ArenaRpcError(
                        "malformed Arena success response"
                    )
                result = response.get("result")
                if not isinstance(result, dict):
                    raise ArenaRpcError(
                        "Arena RPC result must be an object"
                    )
                return dict(_validate_operation_result(op, result))
            except BaseException:
                self._abort_transport()
                raise

    def _abort_transport(self) -> None:
        self._closed = True
        self._token = ""
        self._socket.close()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                result = _strict_object(
                    self._call("close"),
                    required={"closed"},
                    label="Arena close result",
                )
                if result["closed"] is not True:
                    raise ArenaRpcError(
                        "Arena close was not acknowledged"
                    )
            finally:
                self._closed = True
                self._token = ""
                self._socket.close()

    def __enter__(self) -> "ArenaRpcClient":
        return self

    def __exit__(
        self,
        exception_type: object,
        _exception: object,
        _traceback: object,
    ) -> None:
        if exception_type is None:
            self.close()
            return
        with self._lock:
            self._abort_transport()


def _decode_snapshot(value: object) -> dict[str, Any]:
    snapshot = _strict_object(
        value,
        required={"frame", "actions", "levels_completed", "terminal"},
        label="Arena snapshot",
    )
    frame = _normalize_frame(snapshot["frame"])
    actions = tuple(_normalize_actions(snapshot["actions"]))
    levels = snapshot["levels_completed"]
    terminal = snapshot["terminal"]
    if not _is_plain_int(levels) or int(levels) < 0:
        raise ArenaRpcError("snapshot levels_completed is invalid")
    if not isinstance(terminal, bool):
        raise ArenaRpcError("snapshot terminal is invalid")
    return {
        "frame": frame,
        "actions": actions,
        "levels_completed": int(levels),
        "terminal": terminal,
    }


def _validate_operation_result(
    op: str,
    value: object,
) -> Mapping[str, Any]:
    if op == "open":
        return _strict_object(
            value,
            required={"binding_sha256", "snapshot"},
            label=f"Arena {op} result",
        )
    if op in {"observe", "reset", "step"}:
        return _strict_object(
            value,
            required={"snapshot"},
            label=f"Arena {op} result",
        )
    if op == "close":
        return _strict_object(
            value,
            required={"closed"},
            label=f"Arena {op} result",
        )
    raise ArenaRpcError("unknown local Arena RPC operation")


class RemoteArena:
    """One default exploration proxy; clone handles are intentionally absent."""

    __slots__ = ("_client", "_snapshot")

    def __init__(
        self,
        client: ArenaRpcClient,
        snapshot: Mapping[str, Any],
    ):
        self._client = client
        self._snapshot = _decode_snapshot(snapshot)

    def _ensure_open(self) -> None:
        if self._client._closed:
            raise ArenaRpcError("remote Arena session is closed")

    @staticmethod
    def _frame_array(frame: list[list[int]]) -> Any:
        try:
            import numpy as np
        except ImportError:
            return [list(row) for row in frame]
        return np.asarray(frame, dtype=np.uint8)

    def _replace_snapshot(self, value: object) -> None:
        self._snapshot = _decode_snapshot(value)

    def reset(self) -> Any:
        self._ensure_open()
        result = _strict_object(
            self._client._call("reset"),
            required={"snapshot"},
            label="Arena reset result",
        )
        self._replace_snapshot(result["snapshot"])
        return self.frame()

    def observe(self) -> Any:
        self._ensure_open()
        result = _strict_object(
            self._client._call("observe"),
            required={"snapshot"},
            label="Arena observe result",
        )
        self._replace_snapshot(result["snapshot"])
        return self.frame()

    def frame(self) -> Any:
        self._ensure_open()
        # Return a copy so solver-side mutation cannot alter cached state.
        return self._frame_array(self._snapshot["frame"])

    def step(
        self,
        action: Any,
        x: int | None = None,
        y: int | None = None,
    ) -> Any:
        self._ensure_open()
        # Forward even malformed, JSON-serializable action tokens to the
        # trusted host.  The host validates before touching the engine and
        # durably records a rejected RPC.  Client-side range rejection would
        # let solver code catch the exception and continue without leaving
        # authoritative evidence of the protocol escape.
        if x is not None or y is not None:
            wire_action: Any = [action, x, y]
        elif isinstance(action, tuple):
            wire_action = list(action)
        else:
            wire_action = action
        result = self._client._call("step", action=wire_action)
        result = _strict_object(
            result,
            required={"snapshot"},
            label="Arena step result",
        )
        self._replace_snapshot(result["snapshot"])
        return self.frame()

    @property
    def actions(self) -> tuple[int, ...]:
        self._ensure_open()
        return tuple(self._snapshot["actions"])

    @property
    def levels_completed(self) -> int:
        self._ensure_open()
        return int(self._snapshot["levels_completed"])

    def terminal(self) -> bool:
        self._ensure_open()
        return bool(self._snapshot["terminal"])
