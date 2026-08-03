#!/usr/bin/env python3
"""Canonical proposer-side client for the ARC-AGI-3 Arena RPC.

The trusted host implementation remains in :mod:`arc_agi3_arena_rpc`.  This
module is the only RPC code installed in a solver/proposer container: it has no
repository imports, engine imports, directory traversal, or file-reading API.
It exposes only authenticated public observations and public actions over one
pre-provisioned Unix socket.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import numbers
import re
import socket
import threading
from typing import Any, Mapping


RPC_SCHEMA = "arc-agi3-arena-rpc/v1"
MAX_MESSAGE_BYTES = 256 * 1024
MAX_FRAME_SIDE = 64
MAX_TOTAL_CELLS = MAX_FRAME_SIDE * MAX_FRAME_SIDE
DEFAULT_SOCKET_TIMEOUT_SECONDS = 30.0
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PUBLIC_OPERATIONS = frozenset({
    "open",
    "observe",
    "reset",
    "step",
    "close",
})


class ArenaRpcError(RuntimeError):
    """A local transport or remote public-contract failure."""


class ArenaRpcContractError(ArenaRpcError):
    """A malformed or unauthorized RPC request."""


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
        raise ArenaRpcContractError(f"{label} fields mismatch")
    return value


def _loads_json(raw: bytes, *, label: str) -> object:
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


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _wire_mac(token: str, value: Mapping[str, Any]) -> str:
    return hmac.new(
        token.encode("ascii"),
        _canonical_json(value),
        hashlib.sha256,
    ).hexdigest()


def _recv_line(connection: socket.socket) -> bytes | None:
    data = bytearray()
    while True:
        chunk = connection.recv(
            min(65_536, MAX_MESSAGE_BYTES + 1 - len(data))
        )
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


def _normalize_frame(frame: Any) -> list[list[int]]:
    if hasattr(frame, "tolist"):
        frame = frame.tolist()
    if not isinstance(frame, (list, tuple)) or not frame:
        raise ArenaRpcContractError(
            "Arena frame must be a nonempty row sequence"
        )
    if len(frame) > MAX_FRAME_SIDE:
        raise ArenaRpcContractError("Arena frame exceeds 64 rows")
    rows: list[list[int]] = []
    width: int | None = None
    cells = 0
    for raw_row in frame:
        if not isinstance(raw_row, (list, tuple)) or not raw_row:
            raise ArenaRpcContractError(
                "Arena frame rows must be nonempty sequences"
            )
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
            if not 0 <= cell <= 15:
                raise ArenaRpcContractError(
                    "Arena frame cells must be colour indices in 0..15"
                )
            row.append(cell)
        rows.append(row)
        cells += len(row)
    if cells > MAX_TOTAL_CELLS:
        raise ArenaRpcContractError(
            "Arena frame exceeds the 64x64 cell bound"
        )
    return rows


def _normalize_actions(actions: object) -> list[int]:
    if not isinstance(actions, (list, tuple)):
        raise ArenaRpcContractError("Arena actions must be a sequence")
    output: list[int] = []
    for raw in actions:
        if not _is_plain_int(raw) or not 1 <= int(raw) <= 7:
            raise ArenaRpcContractError(
                "Arena action IDs must be integers in 1..7"
            )
        action = int(raw)
        if action in output:
            raise ArenaRpcContractError("Arena action IDs must be unique")
        output.append(action)
    if not output:
        raise ArenaRpcContractError(
            "Arena must expose at least one action"
        )
    return output


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
    operation: str,
    value: object,
) -> Mapping[str, Any]:
    if operation == "open":
        return _strict_object(
            value,
            required={"binding_sha256", "snapshot"},
            label="Arena open result",
        )
    if operation in {"observe", "reset", "step"}:
        return _strict_object(
            value,
            required={"snapshot"},
            label=f"Arena {operation} result",
        )
    if operation == "close":
        return _strict_object(
            value,
            required={"closed"},
            label="Arena close result",
        )
    raise ArenaRpcError("unknown local Arena RPC operation")


class ArenaRpcClient:
    """Container-side HMAC transport for one exploration branch."""

    def __init__(self, socket_path: object, token: str):
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
            if not isinstance(socket_path, (str, bytes)):
                try:
                    socket_path = socket_path.__fspath__()  # type: ignore[attr-defined]
                except AttributeError as exc:
                    raise TypeError(
                        "socket path must be str, bytes, or path-like"
                    ) from exc
                if not isinstance(socket_path, (str, bytes)):
                    raise TypeError(
                        "socket path __fspath__ returned a non-path value"
                    )
            self._socket.connect(socket_path)
            opened = self._call("open")
            opened = _strict_object(
                opened,
                required={"binding_sha256", "snapshot"},
                label="Arena open result",
            )
            if (
                not isinstance(opened["binding_sha256"], str)
                or _SHA256_RE.fullmatch(opened["binding_sha256"]) is None
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

    def _call(self, operation: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise ArenaRpcError("Arena RPC client is closed")
            if operation not in _PUBLIC_OPERATIONS:
                raise ArenaRpcError("unknown local Arena RPC operation")
            try:
                sequence = self._seq
                unsigned = {
                    "schema": RPC_SCHEMA,
                    "session": self._session_id,
                    "seq": sequence,
                    "op": operation,
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
                response = _strict_object(
                    _loads_json(raw, label="RPC response"),
                    required={
                        "schema", "session", "seq", "ok", "mac"
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
                    or response["seq"] != sequence
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
                        "schema", "session", "seq", "ok", "error", "mac"
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
                    "schema", "session", "seq", "ok", "result", "mac"
                }:
                    raise ArenaRpcError(
                        "malformed Arena success response"
                    )
                result = response.get("result")
                if not isinstance(result, dict):
                    raise ArenaRpcError(
                        "Arena RPC result must be an object"
                    )
                return dict(
                    _validate_operation_result(operation, result)
                )
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


class RemoteArena:
    """One default exploration proxy; engine/private handles are absent."""

    __slots__ = ("_client", "_snapshot")

    def __init__(self, client: ArenaRpcClient, snapshot: Mapping[str, Any]):
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
        return self._frame_array(self._snapshot["frame"])

    def step(
        self,
        action: Any,
        x: int | None = None,
        y: int | None = None,
    ) -> Any:
        self._ensure_open()
        if x is not None or y is not None:
            wire_action: Any = [action, x, y]
        elif isinstance(action, tuple):
            wire_action = list(action)
        else:
            wire_action = action
        result = _strict_object(
            self._client._call("step", action=wire_action),
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


__all__ = (
    "ArenaRpcClient",
    "ArenaRpcContractError",
    "ArenaRpcError",
    "RemoteArena",
)
