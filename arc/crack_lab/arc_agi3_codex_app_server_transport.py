#!/usr/bin/env python3
"""Pinned host mediator for the isolated ARC-AGI-3 Codex controller.

The authenticated Codex app-server runs in a separate digest-pinned controller
container.  This trusted host mediator owns credentials and the one
attempt-bound Unix bridge, attaches to the controller only through stdio, and
serves the closed dynamic-tool inventory.  The controller receives neither a
bridge/Arena endpoint nor credential source files, argv, environment values,
or retained receipts.

This module intentionally exposes contracts as frozen dataclasses.  The Docker
backend implements :class:`ProbeExecutor`; the controller can therefore run
model-authored probes only in a fresh, credential-free child container rather
than under the bridge worker's identity.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import ctypes
import selectors
import signal
import socket
import stat
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence

try:
    from arc.crack_lab import arc_agi3_proposer_boundary as Boundary
except ModuleNotFoundError:  # pragma: no cover - direct-script fallback
    import arc_agi3_proposer_boundary as Boundary


SCHEMA = 1
BRIDGE_PROTOCOL_VERSION = 1
MAX_AUTH_SOURCE_BYTES = 64 * 1024
MAX_BRIDGE_LINE_BYTES = 8 * 1024 * 1024
BRIDGE_RESPONSE_TIMEOUT_SECONDS = 30.0
MAX_BRIDGE_RESPONSE_REPLAYS = 2
MAX_PROBE_FILES = 128
MAX_PROBE_ARGUMENTS = 64
MAX_PROBE_ARGUMENT_BYTES = 16 * 1024
MAX_PROBE_TIMEOUT_SECONDS = 600
MAX_PINNED_CODEX_BINARY_BYTES = 320 * 1024 * 1024
MAX_APP_SERVER_STATE_FILES = 8_192
MAX_APP_SERVER_STATE_FILE_BYTES = 320 * 1024 * 1024
MAX_APP_SERVER_STATE_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
MAX_APP_SERVER_STATE_DEPTH = 64
CODEX_STATE_DATABASE_NAME = "state_5.sqlite"
SQLITE3_HEADER = b"SQLite format 3\x00"
APP_SERVER_HARD_SAFETY_SECONDS = 21_600
MAX_AUTH_REFRESHES = min(
    8,
    (APP_SERVER_HARD_SAFETY_SECONDS + 3_599) // 3_600 + 1,
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_RELATIVE_RE = re.compile(
    r"^(?!/)(?!\.)(?!.*\/\.)(?!.*(?:\.\.?)(?:\/|$))"
    r"(?!.*//)[A-Za-z0-9][A-Za-z0-9._/-]{0,1023}$"
)

INITIALIZE_PARAMS = {
    "clientInfo": {
        "name": "gkm-arc-agi3-contiguous",
        "title": "GKM ARC-AGI-3 contiguous controller",
        "version": "1",
    },
    "capabilities": {
        "experimentalApi": True,
        "mcpServerOpenaiFormElicitation": False,
        "optOutNotificationMethods": [
            "remoteControl/status/changed",
        ],
        "requestAttestation": False,
    },
}
INITIALIZE_PARAMS_SHA256 = hashlib.sha256(
    json.dumps(
        INITIALIZE_PARAMS,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()

PREFLIGHT_REQUEST_SEQUENCE = (
    "initialize",
    "account/login/start",
    "account/read",
    "account/rateLimits/read",
    "model/list",
    "modelProvider/capabilities/read",
    "config/read",
    "skills/list",
    "hooks/list",
    "plugin/list",
    "app/list",
    "experimentalFeature/list",
    "mcpServerStatus/list",
)
PREFLIGHT_NOTIFICATION_CARDINALITY = {
    "initialized": 1,
    "account/login/completed": 1,
    "account/updated": 1,
}
TURN_REQUEST_SEQUENCE = (
    "thread/start",
    "thread/resume",
    "turn/start",
    "turn/interrupt",
    "account/rateLimits/read",
)
TURN_REQUEST_METHODS = frozenset(TURN_REQUEST_SEQUENCE)
ACTIVE_SERVER_REQUEST_METHODS = frozenset(
    {
        "account/chatgptAuthTokens/refresh",
        "item/tool/call",
    }
)

# Capability-bearing app-server features that must be explicitly disabled in
# the private projection and observed disabled in the live feature inventory.
# This is deliberately a denylist: experimentalFeature/list also reports
# implementation/runtime features (for example compression) that do not grant
# model-visible authority.
SECURITY_DISABLED_FEATURES = (
    "apps",
    "auth_elicitation",
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "code_mode",
    "code_mode_buffered_exec",
    "code_mode_host",
    "code_mode_only",
    "computer_use",
    "default_mode_request_user_input",
    "deferred_executor",
    "enable_fanout",
    "enable_mcp_apps",
    "executor_capability_discovery",
    "external_agent_memory_import",
    "goals",
    "guardian_approval",
    "hooks",
    "image_generation",
    "in_app_browser",
    "js_repl",
    "memories",
    "mentions_v2",
    "multi_agent",
    "multi_agent_v2",
    "network_proxy",
    "non_prefixed_mcp_tool_names",
    "plugin_hooks",
    "plugin_sharing",
    "plugins",
    "remote_control",
    "remote_plugin",
    "request_permissions_tool",
    "respect_system_proxy",
    "shell_snapshot",
    "shell_tool",
    "skill_mcp_dependency_install",
    "skill_search",
    "standalone_web_search",
    "tool_call_mcp_elicitation",
    "tool_search",
    "tool_suggest",
    "unified_exec",
    "web_search_cached",
    "web_search_request",
    "workspace_dependencies",
)

PROJECT_DISCOVERY_MARKERS = frozenset(
    {
        ".agents",
        ".codex",
        ".git",
        ".hg",
        ".svn",
        "AGENT.md",
        "AGENTS.md",
        "Cargo.toml",
        "MODULE.bazel",
        "WORKSPACE",
        "WORKSPACE.bazel",
        "go.mod",
        "package.json",
        "pyproject.toml",
        "setup.cfg",
        "setup.py",
    }
)

DISABLED_SYSTEM_SKILLS = (
    "imagegen",
    "openai-docs",
    "plugin-creator",
    "review-agent",
    "skill-creator",
    "skill-installer",
)

BASE_INSTRUCTIONS = """\
You are the isolated proposer for exactly one ARC-AGI-3 frontier. Use only the
declared contiguous_lane tools. Treat all observations as public game data.
Publish candidate or WIP bytes only through the declared publication tool.
Final prose is non-authoritative and never becomes a candidate artifact.
"""
DEVELOPER_INSTRUCTIONS = """\
Do not request shell, process, filesystem, web, app, memory, skill, plugin,
multi-agent, configuration, authentication, or arbitrary MCP capabilities.
Never inspect implementation source, private adapter state, tokens, host paths,
or process metadata. Work only in the attempt workspace and public Arena clone.
"""


class AppServerTransportError(RuntimeError):
    """The pinned transport, credential, transcript, or bridge failed closed."""


class DeterministicControllerConfigurationError(
    AppServerTransportError
):
    """A controller configuration defect that cannot improve by polling."""

    substrate_failure_class = "DETERMINISTIC_CONFIGURATION"

    def __init__(self, message: str, *, failure_code: str) -> None:
        super().__init__(message)
        if (
            not isinstance(failure_code, str)
            or re.fullmatch(r"[a-z][a-z0-9_]{0,127}", failure_code)
            is None
        ):
            raise AppServerTransportError(
                "deterministic controller failure code is malformed"
            )
        self.substrate_failure_code = failure_code


class BridgeResponseLost(AppServerTransportError):
    """The bridge closed before an in-flight response was received."""


class _ProtocolEof(Exception):
    """Both app-server output pipes reached exact EOF after terminal turn."""


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


PROBE_STDERR_VISIBILITY_KIND = (
    "arc_agi3_contiguous_probe_stderr_visibility"
)
PROBE_STDERR_SANITIZED_LINE = (
    "Probe stderr was withheld by the host; use the terminal status and "
    "retained audit digest.\n"
)
_PYTHON_TRACEBACK_MARKERS = (
    "Traceback (most recent call last):",
    'File "',
    "KeyboardInterrupt",
)


def _probe_stderr_visibility_projection(
    stderr: bytes,
) -> tuple[str, dict[str, object]]:
    """Separate model-visible status from immutable raw probe stderr.

    Probe stderr is an untrusted execution surface.  In particular, Python
    tracebacks can quote private harness filenames and implementation lines.
    The model therefore receives either the empty string or one fixed line;
    the exact raw bytes remain host evidence and are classified separately.
    """

    if not isinstance(stderr, bytes):
        raise AppServerTransportError(
            "probe stderr visibility input is not bytes"
        )
    decoded = stderr.decode("utf-8", errors="replace")
    classification = (
        "empty"
        if not stderr
        else (
            "python_or_harness_traceback"
            if any(marker in decoded for marker in _PYTHON_TRACEBACK_MARKERS)
            else "other_stderr"
        )
    )
    visible = "" if not stderr else PROBE_STDERR_SANITIZED_LINE
    body: dict[str, object] = {
        "schema": 1,
        "kind": PROBE_STDERR_VISIBILITY_KIND,
        "raw_stderr_sha256": sha256_bytes(stderr),
        "raw_stderr_bytes": len(stderr),
        "raw_surface_classification": classification,
        "raw_bytes_host_only": True,
        "proposer_visible_stderr": visible,
        "proposer_visible_stderr_sha256": sha256_bytes(
            visible.encode("utf-8")
        ),
        "proposer_visible_traceback_absent": True,
        "proposer_visible_taint_status": "CLEAN",
    }
    return visible, body


def _retain_probe_stderr_visibility_receipt(
    stderr_path: Path,
    body: Mapping[str, object],
) -> tuple[str, str]:
    """Write or reopen the immutable host-only stderr classification."""

    path = Path(stderr_path).with_name(
        "stderr_visibility_receipt.json"
    )
    payload = canonical_json(dict(body)) + b"\n"
    if path.exists() or path.is_symlink():
        observed = _bounded_regular_bytes(
            path,
            max_bytes=64 * 1024,
            private_owner=True,
        )
        if observed != payload:
            raise AppServerTransportError(
                "retained probe stderr visibility receipt differs"
            )
    else:
        _write_new_bytes(path, payload, mode=0o400)
    return str(path), sha256_bytes(payload)


def _observed_executable_identity(path: Path) -> dict[str, Any]:
    """Descriptor-bind the executable named by the operating system."""
    selected = Path(path)
    descriptor = -1
    try:
        before = selected.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size > MAX_PINNED_CODEX_BINARY_BYTES
        ):
            raise AppServerTransportError(
                "observed process executable is not a bounded regular file"
            )
        descriptor = os.open(
            selected, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
        current = os.fstat(descriptor)
        stable = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(before, name) != getattr(current, name)
            for name in stable
        ):
            raise AppServerTransportError(
                "observed process executable changed during identity read"
            )
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
        if any(
            getattr(current, name) != getattr(after, name)
            for name in stable
        ):
            raise AppServerTransportError(
                "observed process executable changed during hashing"
            )
        return {
            "path": str(selected),
            "device": current.st_dev,
            "inode": current.st_ino,
            "size": current.st_size,
            "mtime_ns": current.st_mtime_ns,
            "ctime_ns": current.st_ctime_ns,
            "sha256": digest.hexdigest(),
        }
    except OSError as exc:
        raise AppServerTransportError(
            "observed process executable cannot be authenticated"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def observe_os_process_start_identity(pid: int) -> str:
    """Return an OS-birth/executable identity safe against PID reuse."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        raise AppServerTransportError("process identity pid is malformed")
    platform_name = os.uname().sysname
    if platform_name == "Darwin":
        class _ProcBsdInfo(ctypes.Structure):
            _fields_ = [
                ("pbi_flags", ctypes.c_uint32),
                ("pbi_status", ctypes.c_uint32),
                ("pbi_xstatus", ctypes.c_uint32),
                ("pbi_pid", ctypes.c_uint32),
                ("pbi_ppid", ctypes.c_uint32),
                ("pbi_uid", ctypes.c_uint32),
                ("pbi_gid", ctypes.c_uint32),
                ("pbi_ruid", ctypes.c_uint32),
                ("pbi_rgid", ctypes.c_uint32),
                ("pbi_svuid", ctypes.c_uint32),
                ("pbi_svgid", ctypes.c_uint32),
                ("rfu_1", ctypes.c_uint32),
                ("pbi_comm", ctypes.c_char * 16),
                ("pbi_name", ctypes.c_char * 32),
                ("pbi_nfiles", ctypes.c_uint32),
                ("pbi_pgid", ctypes.c_uint32),
                ("pbi_pjobc", ctypes.c_uint32),
                ("e_tdev", ctypes.c_uint32),
                ("e_tpgid", ctypes.c_uint32),
                ("pbi_nice", ctypes.c_int32),
                ("pbi_start_tvsec", ctypes.c_uint64),
                ("pbi_start_tvusec", ctypes.c_uint64),
            ]

        try:
            library = ctypes.CDLL(
                "/usr/lib/libproc.dylib", use_errno=True
            )
            library.proc_pidinfo.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_uint64,
                ctypes.c_void_p,
                ctypes.c_int,
            ]
            library.proc_pidinfo.restype = ctypes.c_int
            info = _ProcBsdInfo()
            observed = library.proc_pidinfo(
                pid, 3, 0, ctypes.byref(info), ctypes.sizeof(info)
            )
            if observed != ctypes.sizeof(info) or info.pbi_pid != pid:
                raise AppServerTransportError(
                    "Darwin process birth record is unavailable"
                )
            path_buffer = ctypes.create_string_buffer(4096)
            library.proc_pidpath.argtypes = [
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_uint32,
            ]
            library.proc_pidpath.restype = ctypes.c_int
            path_bytes = library.proc_pidpath(
                pid, path_buffer, len(path_buffer)
            )
            if path_bytes <= 0:
                raise AppServerTransportError(
                    "Darwin process executable path is unavailable"
                )
            executable_path = Path(
                os.fsdecode(path_buffer.value)
            )
            facts = {
                "schema": 1,
                "platform": "Darwin",
                "pid": pid,
                "process_group_id": int(info.pbi_pgid),
                "start_seconds": int(info.pbi_start_tvsec),
                "start_microseconds": int(info.pbi_start_tvusec),
                "executable":
                    _observed_executable_identity(executable_path),
            }
        except (OSError, ValueError) as exc:
            raise AppServerTransportError(
                "Darwin process identity cannot be observed"
            ) from exc
    elif platform_name == "Linux":
        try:
            boot_id = Path(
                "/proc/sys/kernel/random/boot_id"
            ).read_text(encoding="ascii").strip()
            stat_raw = Path(f"/proc/{pid}/stat").read_text(
                encoding="ascii"
            )
            closing = stat_raw.rfind(")")
            if closing < 0:
                raise ValueError("malformed /proc stat")
            fields = stat_raw[closing + 2 :].split()
            process_group_id = int(fields[2])
            start_ticks = int(fields[19])
            executable_path = Path(
                os.readlink(f"/proc/{pid}/exe")
            )
            facts = {
                "schema": 1,
                "platform": "Linux",
                "boot_id": boot_id,
                "pid": pid,
                "process_group_id": process_group_id,
                "start_ticks": start_ticks,
                "executable":
                    _observed_executable_identity(executable_path),
            }
        except (OSError, UnicodeError, ValueError, IndexError) as exc:
            raise AppServerTransportError(
                "Linux process identity cannot be observed"
            ) from exc
    else:
        raise AppServerTransportError(
            "OS process birth identity is unsupported on this platform"
        )
    return sha256_bytes(canonical_json(facts))


def is_safe_relative_path(value: object) -> bool:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or "\\" in value
        or len(value.encode("utf-8")) > 1024
        or not SAFE_RELATIVE_RE.fullmatch(value)
    ):
        return False
    path = PurePosixPath(value)
    return bool(
        not path.is_absolute()
        and 0 < len(path.parts) <= 12
        and str(path) == value
        and all(
            part not in {"", ".", ".."} and not part.startswith(".")
            for part in path.parts
        )
    )


def _reject_duplicate_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AppServerTransportError(
                f"duplicate JSON object key: {key}"
            )
        result[key] = value
    return result


def strict_json_loads(raw: bytes | str) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                AppServerTransportError(
                    f"non-finite JSON number: {value}"
                )
            ),
        )
    except AppServerTransportError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AppServerTransportError("malformed JSON") from exc


def _bounded_regular_bytes(
    path: Path,
    *,
    max_bytes: int,
    private_owner: bool = False,
    allow_empty: bool = False,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AppServerTransportError(
            f"could not descriptor-open regular file: {path}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (before.st_size == 0 and not allow_empty)
            or before.st_size > max_bytes
            or (
                private_owner
                and (
                    before.st_uid != os.getuid()
                    or stat.S_IMODE(before.st_mode) & 0o077
                )
            )
        ):
            raise AppServerTransportError(
                f"file ownership/type/bounds are inadmissible: {path}"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                raise AppServerTransportError(
                    f"file changed while reading: {path}"
                )
            chunks.append(block)
            remaining -= len(block)
        after = os.fstat(descriptor)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_uid",
            "st_gid",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(before, name) != getattr(after, name)
            for name in stable_fields
        ):
            raise AppServerTransportError(
                f"file metadata changed while reading: {path}"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class ExternalChatGptCredentials:
    """Ephemeral external-token tuple.

    ``leak_sentinels`` is deliberately excluded from every receipt.  It exists
    only so the supervisor can scan live and retained bytes for every nonempty
    source credential, including credentials it never injects.
    """

    access_token: str
    account_id: str
    plan_type: str | None
    leak_sentinels: tuple[str, ...]
    source_path: str

    def login_params(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": "chatgptAuthTokens",
            "accessToken": self.access_token,
            "chatgptAccountId": self.account_id,
        }
        if self.plan_type is not None:
            result["chatgptPlanType"] = self.plan_type
        return result

    def redacted_login_params(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": "chatgptAuthTokens",
            "accessToken": "REDACTED",
            "chatgptAccountId": "REDACTED",
        }
        if self.plan_type is not None:
            result["chatgptPlanType"] = self.plan_type
        return result

    def redacted_request_sha256(self) -> str:
        return sha256_bytes(
            canonical_json(
                {
                    "schema": SCHEMA,
                    "protocol_variant":
                        "codex-app-server-0.145.0-external-chatgpt-tokens",
                    "method": "account/login/start",
                    "cardinality": 1,
                    "params": self.redacted_login_params(),
                }
            )
        )


def load_external_chatgpt_credentials(
    source: Path,
) -> ExternalChatGptCredentials:
    """Load the one pinned Codex ``auth.json`` schema without fallback."""

    raw = _bounded_regular_bytes(
        Path(source),
        max_bytes=MAX_AUTH_SOURCE_BYTES,
        private_owner=True,
    )
    value = strict_json_loads(raw)
    if (
        not isinstance(value, dict)
        or set(value)
        != {"OPENAI_API_KEY", "auth_mode", "last_refresh", "tokens"}
        or value["auth_mode"] != "chatgpt"
        or not (
            value["OPENAI_API_KEY"] is None
            or isinstance(value["OPENAI_API_KEY"], str)
        )
        or not (
            value["last_refresh"] is None
            or isinstance(value["last_refresh"], str)
        )
        or not isinstance(value["tokens"], dict)
        or set(value["tokens"])
        != {"access_token", "account_id", "id_token", "refresh_token"}
    ):
        raise AppServerTransportError(
            "credential source does not match the pinned chatgpt schema"
        )
    tokens = value["tokens"]
    if (
        not isinstance(tokens["access_token"], str)
        or not tokens["access_token"]
        or not isinstance(tokens["account_id"], str)
        or not tokens["account_id"]
        or not isinstance(tokens["id_token"], str)
        or not tokens["id_token"]
        or not isinstance(tokens["refresh_token"], str)
        or not tokens["refresh_token"]
    ):
        raise AppServerTransportError(
            "credential source has incomplete external-token fields"
        )
    sentinels = tuple(
        item
        for item in (
            value["OPENAI_API_KEY"],
            tokens["access_token"],
            tokens["account_id"],
            tokens["id_token"],
            tokens["refresh_token"],
        )
        if isinstance(item, str) and item
    )
    return ExternalChatGptCredentials(
        access_token=tokens["access_token"],
        account_id=tokens["account_id"],
        plan_type=None,
        leak_sentinels=sentinels,
        source_path=str(Path(source).resolve(strict=True)),
    )


def _object_schema(
    properties: Mapping[str, Any],
    required: Sequence[str],
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": False,
    }


PATH_SCHEMA = {
    "type": "string",
    "minLength": 1,
    "maxLength": 1024,
    "pattern": SAFE_RELATIVE_RE.pattern,
}
ACTION_SCHEMA = {
    "oneOf": [
        {"type": "integer", "minimum": 1, "maximum": 7},
        {
            "type": "array",
            "prefixItems": [
                {"const": 6},
                {"type": "integer"},
                {"type": "integer"},
            ],
            "items": False,
            "minItems": 3,
            "maxItems": 3,
        },
    ]
}
EXPORTS_SCHEMA = {
    "type": "object",
    "minProperties": 1,
    "maxProperties": 512,
    "propertyNames": PATH_SCHEMA,
    "additionalProperties": PATH_SCHEMA,
}

_TOOL_ROWS = (
    (
        "arena_observe",
        "Read the current public observation and legal actions.",
        _object_schema({}, ()),
    ),
    (
        "arena_reset",
        "Discard the current exploration clone and recreate it from the "
        "immutable seeded parent frontier.",
        _object_schema({}, ()),
    ),
    (
        "arena_step",
        "Apply one legal public Arena action to the current exploration clone.",
        _object_schema({"action": ACTION_SCHEMA}, ("action",)),
    ),
    (
        "candidate_publish",
        "Publish one declared full candidate action path and UTF-8 source set.",
        _object_schema(
            {
                "candidate_path": {
                    "type": "array",
                    "items": ACTION_SCHEMA,
                    "minItems": 1,
                    "maxItems": 600,
                },
                "exports": EXPORTS_SCHEMA,
            },
            ("candidate_path", "exports"),
        ),
    ),
    (
        "progress",
        "Record a bounded non-authoritative progress note.",
        _object_schema(
            {"message": {"type": "string", "minLength": 1, "maxLength": 4096}},
            ("message",),
        ),
    ),
    (
        "wip_publish",
        (
            "Publish clean UTF-8 WIP under wip/: reusable solver source is "
            "required as the flat source-schema-valid "
            "wip/solver_source/{...} tree; optional broader notes/probes may "
            "appear only below wip/context/."
        ),
        _object_schema({"exports": EXPORTS_SCHEMA}, ("exports",)),
    ),
    (
        "workspace_list",
        "List one descriptor-confined workspace directory.",
        _object_schema(
            {"path": {"oneOf": [PATH_SCHEMA, {"type": "null"}]}},
            ("path",),
        ),
    ),
    (
        "workspace_mkdir",
        "Create one descriptor-confined workspace directory.",
        _object_schema({"path": PATH_SCHEMA}, ("path",)),
    ),
    (
        "workspace_read",
        "Read one bounded UTF-8 workspace file.",
        _object_schema({"path": PATH_SCHEMA}, ("path",)),
    ),
    (
        "workspace_remove",
        "Remove one descriptor-confined regular workspace file.",
        _object_schema({"path": PATH_SCHEMA}, ("path",)),
    ),
    (
        "workspace_write",
        "Atomically write one bounded UTF-8 workspace file.",
        _object_schema(
            {"path": PATH_SCHEMA, "text": {"type": "string"}},
            ("path", "text"),
        ),
    ),
)

BASE_DYNAMIC_TOOL_FUNCTIONS = tuple(
    {
        "type": "function",
        "name": name,
        "description": description,
        "inputSchema": schema,
        "deferLoading": False,
    }
    for name, description, schema in _TOOL_ROWS
)

SAFE_PROBE_TOOL_SPEC = {
    "type": "function",
    "name": "workspace_run_python",
    "description": (
        "Run a declared immutable workspace snapshot in a fresh offline probe "
        "container; no bytes are written back to the proposer workspace."
    ),
    "inputSchema": _object_schema(
        {
            "entrypoint": PATH_SCHEMA,
            "files": {
                "type": "array",
                "items": PATH_SCHEMA,
                "minItems": 1,
                "maxItems": MAX_PROBE_FILES,
                "uniqueItems": True,
            },
            "arguments": {
                "type": "array",
                "items": {
                    "type": "string",
                    "maxLength": MAX_PROBE_ARGUMENT_BYTES,
                },
                "maxItems": MAX_PROBE_ARGUMENTS,
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_PROBE_TIMEOUT_SECONDS,
            },
        },
        ("entrypoint", "files", "arguments", "timeout_seconds"),
    ),
    "deferLoading": False,
}

PRODUCTION_DYNAMIC_TOOL_FUNCTIONS = (
    *BASE_DYNAMIC_TOOL_FUNCTIONS,
    SAFE_PROBE_TOOL_SPEC,
)
DYNAMIC_TOOL_NAMES = tuple(
    tool["name"] for tool in PRODUCTION_DYNAMIC_TOOL_FUNCTIONS
)
DYNAMIC_TOOL_SPECS = (
    {
        "type": "namespace",
        "name": "contiguous_lane",
        "description": (
            "The only attempt-bound ARC-AGI-3 proposer capability. All calls "
            "are correlated to this lane's thread, turn, bridge, and receipts."
        ),
        "tools": list(PRODUCTION_DYNAMIC_TOOL_FUNCTIONS),
    },
)
DYNAMIC_TOOL_SPECS_SHA256 = sha256_bytes(
    canonical_json(DYNAMIC_TOOL_SPECS)
)
BRIDGE_OPERATION_ALLOWLIST = (
    "arena_observe",
    "arena_reset",
    "arena_step",
    "candidate_publish",
    "handshake",
    "probe_snapshot",
    "progress",
    "wip_publish",
    "workspace_list",
    "workspace_mkdir",
    "workspace_read",
    "workspace_remove",
    "workspace_write",
)
BRIDGE_EXEC_ALLOWLIST: tuple[str, ...] = ()
BASE_INSTRUCTIONS_SHA256 = sha256_bytes(
    BASE_INSTRUCTIONS.encode("utf-8")
)
DEVELOPER_INSTRUCTIONS_SHA256 = sha256_bytes(
    DEVELOPER_INSTRUCTIONS.encode("utf-8")
)


@dataclass(frozen=True)
class ProbeResourceLimits:
    cpus: float
    memory_bytes: int
    pids: int
    tmpfs_bytes: int


@dataclass(frozen=True)
class ProbeExecutionRequest:
    schema: Literal[1]
    campaign_id: str
    generation_id: str
    attempt_id: str
    dynamic_request_id: str | int
    dynamic_call_id: str
    thread_id: str
    turn_id: str
    workspace_snapshot_manifest_path: str
    workspace_snapshot_manifest_sha256: str
    workspace_snapshot_tree_sha256: str
    entrypoint: str
    arguments: tuple[str, ...]
    timeout_seconds: int
    stdout_limit_bytes: int
    stderr_limit_bytes: int
    resource_limits: ProbeResourceLimits
    arena_mode: Literal["disabled"]
    arena_session_id: None = None

    def sha256(self) -> str:
        return sha256_bytes(canonical_json(asdict(self)))


@dataclass(frozen=True)
class ProbeExecutionResult:
    schema: Literal[1]
    request_sha256: str
    probe_container_id: str
    image_reference: str
    image_digest: str
    containment_attestation_path: str
    containment_attestation_sha256: str
    snapshot_tree_sha256: str
    stdout_path: str
    stdout_sha256: str
    stdout_bytes: int
    stdout_truncated: bool
    stderr_path: str
    stderr_sha256: str
    stderr_bytes: int
    stderr_truncated: bool
    exit_code: int | None
    timed_out: bool
    output_overflow: bool
    arena_session_id: None
    arena_transcript_path: None
    arena_transcript_sha256: None
    started_monotonic_ns: int
    finished_monotonic_ns: int
    teardown_receipt_path: str
    teardown_receipt_sha256: str
    container_absent: bool
    process_group_absent: bool
    descendants_absent: bool
    no_writeback: Literal[True]


class ProbeExecutor(Protocol):
    def run_probe(
        self,
        *,
        spec: Any,
        launched: Any,
        request: ProbeExecutionRequest,
    ) -> ProbeExecutionResult:
        ...


@dataclass(frozen=True)
class WorkspaceSnapshotEntry:
    path: str
    sha256: str
    bytes: int
    device: int
    inode: int


@dataclass(frozen=True)
class WorkspaceSnapshotManifest:
    schema: Literal[1]
    kind: Literal["arc_agi3_contiguous_workspace_snapshot"]
    campaign_id: str
    generation_id: str
    attempt_id: str
    dynamic_request_id: str | int
    dynamic_call_id: str
    thread_id: str
    turn_id: str
    generation_dir: str
    call_dir: str
    snapshot_root: str
    snapshot_device: int
    snapshot_inode: int
    tree_sha256: str
    entries: tuple[WorkspaceSnapshotEntry, ...]
    source_workspace_tree_sha256: str
    no_writeback: Literal[True]


def workspace_snapshot_manifest_from_dict(
    value: object,
) -> WorkspaceSnapshotManifest:
    if not isinstance(value, dict):
        raise AppServerTransportError("snapshot manifest is not an object")
    required = {
        "schema",
        "kind",
        "campaign_id",
        "generation_id",
        "attempt_id",
        "dynamic_request_id",
        "dynamic_call_id",
        "thread_id",
        "turn_id",
        "generation_dir",
        "call_dir",
        "snapshot_root",
        "snapshot_device",
        "snapshot_inode",
        "tree_sha256",
        "entries",
        "source_workspace_tree_sha256",
        "no_writeback",
    }
    if (
        set(value) != required
        or value["schema"] != 1
        or value["kind"]
        != "arc_agi3_contiguous_workspace_snapshot"
        or value["no_writeback"] is not True
        or not all(
            isinstance(value[name], str)
            and Path(value[name]).is_absolute()
            for name in ("generation_dir", "call_dir", "snapshot_root")
        )
        or Path(value["call_dir"]).parent.parent
        != Path(value["generation_dir"]) / "probe_calls"
        or Path(value["snapshot_root"])
        != Path(value["call_dir"]) / "snapshot"
        or not isinstance(value["snapshot_device"], int)
        or isinstance(value["snapshot_device"], bool)
        or value["snapshot_device"] <= 0
        or not isinstance(value["snapshot_inode"], int)
        or isinstance(value["snapshot_inode"], bool)
        or value["snapshot_inode"] <= 0
        or not SHA256_RE.fullmatch(str(value["tree_sha256"]))
        or not SHA256_RE.fullmatch(
            str(value["source_workspace_tree_sha256"])
        )
        or not isinstance(value["entries"], list)
        or not 1 <= len(value["entries"]) <= MAX_PROBE_FILES
    ):
        raise AppServerTransportError("snapshot manifest schema mismatch")
    entries: list[WorkspaceSnapshotEntry] = []
    prior = ""
    total = 0
    for row in value["entries"]:
        if (
            not isinstance(row, dict)
            or set(row)
            != {"path", "sha256", "bytes", "device", "inode"}
            or not isinstance(row["path"], str)
            or not is_safe_relative_path(row["path"])
            or row["path"] <= prior
            or not isinstance(row["sha256"], str)
            or not SHA256_RE.fullmatch(row["sha256"])
            or not isinstance(row["bytes"], int)
            or isinstance(row["bytes"], bool)
            or row["bytes"] < 0
            or not isinstance(row["device"], int)
            or isinstance(row["device"], bool)
            or row["device"] <= 0
            or not isinstance(row["inode"], int)
            or isinstance(row["inode"], bool)
            or row["inode"] <= 0
        ):
            raise AppServerTransportError(
                "snapshot entry is malformed or unsorted"
            )
        total += row["bytes"]
        if total > MAX_BRIDGE_LINE_BYTES:
            raise AppServerTransportError("snapshot aggregate is oversized")
        entries.append(WorkspaceSnapshotEntry(**row))
        prior = row["path"]
    try:
        manifest = WorkspaceSnapshotManifest(
            **{
                **value,
                "entries": tuple(entries),
            }
        )
    except TypeError as exc:
        raise AppServerTransportError(
            "snapshot manifest constructor mismatch"
        ) from exc
    return manifest


class ChainedTranscript:
    """Append-only canonical transcript with a digest chain."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.sequence = 0
        self.head: str | None = None
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        self._descriptor = os.open(self.path, flags, 0o600)

    def append(
        self,
        *,
        direction: Literal[
            "client_request",
            "client_notification",
            "client_response",
            "server_response",
            "server_request",
            "server_notification",
            "server_stderr",
        ],
        payload: object,
    ) -> str:
        body = {
            "schema": SCHEMA,
            "sequence": self.sequence + 1,
            "previous_digest": self.head,
            "direction": direction,
            "payload": payload,
        }
        digest = sha256_bytes(canonical_json(body))
        line = canonical_json({**body, "digest": digest}) + b"\n"
        view = memoryview(line)
        while view:
            written = os.write(self._descriptor, view)
            view = view[written:]
        os.fsync(self._descriptor)
        self.sequence += 1
        self.head = digest
        return digest

    def close(self) -> None:
        os.close(self._descriptor)


def _validate_target_boundary_result(
    result: object,
    *,
    attempt_id: str,
    request: Mapping[str, Any],
    target_level: object,
) -> str:
    """Validate the closed target result before it can reach the model."""

    boundary_fields = {
        "schema",
        "kind",
        "attempt_id",
        "game",
        "target_level",
        "levels_before",
        "levels_completed",
        "arena_binding_sha256",
        "bridge_request_id",
        "bridge_sequence",
        "bridge_mutation_id",
        "crossing_action_sha256",
        "exploration_suffix_sha256",
        "exploration_suffix_length",
        "workspace_tree_sha256",
        "workspace_inventory_sha256",
        "workspace_file_count",
        "workspace_total_bytes",
    }
    if (
        not isinstance(result, dict)
        or set(result)
        != {"target_reached", "boundary", "boundary_sha256"}
        or result.get("target_reached") is not True
        or not isinstance(result.get("boundary"), dict)
        or set(result["boundary"]) != boundary_fields
        or not isinstance(result.get("boundary_sha256"), str)
        or SHA256_RE.fullmatch(result["boundary_sha256"]) is None
    ):
        raise AppServerTransportError(
            "target boundary response is malformed"
        )
    boundary = result["boundary"]
    integer_names = (
        "target_level",
        "levels_before",
        "levels_completed",
        "bridge_sequence",
        "exploration_suffix_length",
        "workspace_file_count",
        "workspace_total_bytes",
    )
    if (
        boundary.get("schema") != SCHEMA
        or isinstance(boundary.get("schema"), bool)
        or boundary.get("kind")
        != "arc_agi3_contiguous_target_boundary"
        or boundary.get("attempt_id") != attempt_id
        or not isinstance(boundary.get("game"), str)
        or not boundary["game"]
        or any(
            not isinstance(boundary.get(name), int)
            or isinstance(boundary.get(name), bool)
            or boundary[name] < 0
            for name in integer_names
        )
        or boundary["target_level"] != target_level
        or boundary["levels_completed"] != target_level
        or boundary["levels_before"] != target_level - 1
        or boundary["bridge_request_id"]
        != request.get("request_id")
        or boundary["bridge_sequence"] != request.get("sequence")
        or boundary["bridge_mutation_id"]
        != request.get("mutation_id")
        or boundary["exploration_suffix_length"] <= 0
        or boundary["workspace_file_count"] <= 0
        or any(
            not isinstance(boundary.get(name), str)
            or SHA256_RE.fullmatch(boundary[name]) is None
            for name in (
                "arena_binding_sha256",
                "crossing_action_sha256",
                "exploration_suffix_sha256",
                "workspace_tree_sha256",
                "workspace_inventory_sha256",
            )
        )
        or boundary["crossing_action_sha256"]
        != sha256_bytes(canonical_json(
            request.get("arguments", {}).get("action")
        ))
        or result["boundary_sha256"]
        != sha256_bytes(canonical_json(boundary))
    ):
        raise AppServerTransportError(
            "target boundary response binding differs"
        )
    return result["boundary_sha256"]


class BridgeClient:
    """Sequential HMAC-authenticated client for one proposer worker."""

    _MUTATING = frozenset(
        {
            "arena_reset",
            "arena_step",
            "candidate_publish",
            "progress",
            "wip_publish",
            "workspace_mkdir",
            "workspace_remove",
            "workspace_write",
        }
    )

    def __init__(
        self,
        *,
        socket_path: Path,
        token_file: Path,
        attempt_id: str,
        evidence_callback: Callable[[str, Mapping[str, Any]], None]
        | None = None,
        target_boundary_callback: Callable[
            [Mapping[str, Any], Mapping[str, Any]], None
        ]
        | None = None,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.attempt_id = attempt_id
        self._token = _bounded_regular_bytes(
            Path(token_file),
            max_bytes=4096,
            private_owner=True,
        ).decode("ascii").strip()
        if len(self._token) < 32 or not re.fullmatch(
            r"[0-9a-f]+", self._token
        ):
            raise AppServerTransportError("bridge token is malformed")
        self._callback = evidence_callback
        self._target_boundary_callback = target_boundary_callback
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.settimeout(BRIDGE_RESPONSE_TIMEOUT_SECONDS)
        self._socket.connect(str(self.socket_path))
        self._recv_buffer = bytearray()
        self._completed_responses: dict[str, dict[str, Any]] = {}
        challenge = self._read_record()
        if (
            not isinstance(challenge, dict)
            or set(challenge)
            != {
                "schema",
                "kind",
                "protocol_version",
                "attempt_id",
                "challenge_nonce",
            }
            or challenge["schema"] != SCHEMA
            or challenge["kind"]
            != "arc_agi3_contiguous_bridge_challenge"
            or challenge["protocol_version"] != BRIDGE_PROTOCOL_VERSION
            or challenge["attempt_id"] != self.attempt_id
            or not isinstance(challenge["challenge_nonce"], str)
            or not re.fullmatch(
                r"[0-9a-f]{32}", challenge["challenge_nonce"]
            )
        ):
            raise AppServerTransportError("bridge challenge mismatch")
        self.challenge_nonce = challenge["challenge_nonce"]
        self.sequence = 0
        self.mutation_sequence = 0
        self.session_nonce: str | None = None
        self.handshake_request_sha256: str | None = None
        self.handshake_response_sha256: str | None = None
        self.handshake_result: dict[str, Any] | None = None
        self.target_boundary_sha256: str | None = None
        response = self.call(
            "handshake",
            {},
            idempotency_key="handshake",
        )
        if (
            not isinstance(response, dict)
            or set(response)
            != {
                "attempt_id",
                "campaign_id",
                "environment_names",
                "exec_allowlist",
                "frontier_sha256",
                "game",
                "generation_id",
                "operation_allowlist",
                "policy_sha256",
                "protocol_version",
                "provider_credential_names",
                "session_nonce",
                "target_level",
            }
            or response.get("attempt_id") != self.attempt_id
            or response.get("protocol_version")
            != BRIDGE_PROTOCOL_VERSION
            or not isinstance(response.get("session_nonce"), str)
            or not re.fullmatch(
                r"[0-9a-f]{32}", response["session_nonce"]
            )
            or not all(
                isinstance(response.get(name), str)
                and bool(response[name])
                for name in (
                    "campaign_id",
                    "generation_id",
                    "game",
                )
            )
            or not isinstance(response.get("target_level"), int)
            or isinstance(response["target_level"], bool)
            or response["target_level"] <= 0
            or not SHA256_RE.fullmatch(
                str(response.get("frontier_sha256"))
            )
            or not SHA256_RE.fullmatch(
                str(response.get("policy_sha256"))
            )
            or response.get("operation_allowlist")
            != list(BRIDGE_OPERATION_ALLOWLIST)
            or response.get("exec_allowlist")
            != list(BRIDGE_EXEC_ALLOWLIST)
            or response.get("provider_credential_names") != []
            or not isinstance(response.get("environment_names"), list)
            or not all(
                isinstance(name, str)
                for name in response["environment_names"]
            )
            or response["environment_names"]
            != sorted(set(response["environment_names"]))
        ):
            raise AppServerTransportError("bridge handshake mismatch")
        self.session_nonce = response["session_nonce"]
        self.handshake_result = dict(response)
        if (
            self.handshake_request_sha256 is None
            or self.handshake_response_sha256 is None
        ):
            raise AppServerTransportError(
                "bridge handshake evidence was not retained"
            )

    def _read_record(self) -> Any:
        while b"\n" not in self._recv_buffer:
            block = self._socket.recv(65536)
            if not block:
                raise BridgeResponseLost(
                    "bridge closed before complete response"
                )
            self._recv_buffer.extend(block)
            if len(self._recv_buffer) > MAX_BRIDGE_LINE_BYTES:
                raise AppServerTransportError(
                    "bridge response exceeds byte bound"
                )
        line, separator, remainder = self._recv_buffer.partition(
            b"\n"
        )
        if not separator:
            raise AppServerTransportError(
                "bridge emitted a partial record"
            )
        self._recv_buffer[:] = remainder
        return strict_json_loads(line)

    def _reconnect(self) -> None:
        try:
            self._socket.close()
        except OSError:
            pass
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.settimeout(BRIDGE_RESPONSE_TIMEOUT_SECONDS)
        self._socket.connect(str(self.socket_path))
        self._recv_buffer.clear()
        challenge = self._read_record()
        if (
            not isinstance(challenge, dict)
            or set(challenge)
            != {
                "schema",
                "kind",
                "protocol_version",
                "attempt_id",
                "challenge_nonce",
            }
            or challenge.get("schema") != SCHEMA
            or challenge.get("kind")
            != "arc_agi3_contiguous_bridge_challenge"
            or challenge.get("protocol_version")
            != BRIDGE_PROTOCOL_VERSION
            or challenge.get("attempt_id") != self.attempt_id
            or challenge.get("challenge_nonce")
            != self.challenge_nonce
        ):
            raise AppServerTransportError(
                "bridge reconnect challenge mismatch"
            )

    def _read_matching_response(
        self,
        *,
        request_id: str,
        sequence: int,
    ) -> dict[str, Any]:
        while True:
            response = self._read_record()
            if not isinstance(response, dict):
                raise AppServerTransportError(
                    "bridge response is not an object"
                )
            observed_id = response.get("request_id")
            if observed_id == request_id:
                if response.get("sequence") != sequence:
                    raise AppServerTransportError(
                        "bridge response sequence mismatch"
                    )
                return response
            prior = self._completed_responses.get(str(observed_id))
            if prior is None or prior != response:
                raise AppServerTransportError(
                    "bridge emitted an unknown out-of-order response"
                )

    def call(
        self,
        operation: str,
        arguments: Mapping[str, Any],
        *,
        idempotency_key: str,
    ) -> Any:
        boundary_hits = Boundary.dynamic_tool_boundary_hits(
            operation, arguments
        )
        if boundary_hits:
            raise AppServerTransportError(
                "workspace source violates the clean-room filesystem "
                "boundary: " + ",".join(boundary_hits)
            )
        if (
            getattr(self, "target_boundary_sha256", None) is not None
            and operation
            in {
                "arena_observe",
                "arena_reset",
                "arena_step",
                "workspace_mkdir",
                "workspace_remove",
                "workspace_write",
                "wip_publish",
            }
        ):
            raise AppServerTransportError(
                "target boundary is frozen; post-target mutation/"
                "observation is forbidden"
            )
        mutating = operation in self._MUTATING
        next_sequence = self.sequence + 1
        next_mutation_sequence = (
            self.mutation_sequence + 1
            if mutating
            else self.mutation_sequence
        )
        request_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"gkm:{self.attempt_id}:{idempotency_key}",
            )
        )
        body: dict[str, Any] = {
            "schema": SCHEMA,
            "kind": "arc_agi3_contiguous_bridge_request",
            "protocol_version": BRIDGE_PROTOCOL_VERSION,
            "attempt_id": self.attempt_id,
            "request_id": request_id,
            "sequence": next_sequence,
            "session_nonce": self.session_nonce,
            "operation": operation,
            "mutation_id": (
                f"{self.attempt_id}:{next_mutation_sequence:08d}"
                if mutating
                else None
            ),
            "challenge_nonce": self.challenge_nonce,
            "arguments": dict(arguments),
        }
        request = {
            **body,
            "auth_hmac": hmac.new(
                self._token.encode("ascii"),
                canonical_json(body),
                hashlib.sha256,
            ).hexdigest(),
        }
        if operation == "handshake":
            self.handshake_request_sha256 = sha256_bytes(
                canonical_json(body)
            )
        if self._callback is not None:
            self._callback("bridge_request", request)
        wire = canonical_json(request) + b"\n"
        response: dict[str, Any] | None = None
        for replay_index in range(MAX_BRIDGE_RESPONSE_REPLAYS + 1):
            try:
                self._socket.sendall(wire)
                response = self._read_matching_response(
                    request_id=request_id,
                    sequence=next_sequence,
                )
                break
            except socket.timeout as exc:
                if replay_index >= MAX_BRIDGE_RESPONSE_REPLAYS:
                    raise AppServerTransportError(
                        "bridge response replay bound exhausted"
                    ) from exc
            except (
                BrokenPipeError,
                ConnectionError,
                OSError,
                BridgeResponseLost,
            ) as exc:
                if replay_index >= MAX_BRIDGE_RESPONSE_REPLAYS:
                    if isinstance(exc, BridgeResponseLost):
                        raise
                    raise AppServerTransportError(
                        "bridge reconnect replay bound exhausted"
                    ) from exc
                self._reconnect()
        if response is None:
            raise AppServerTransportError(
                "bridge produced no replayable response"
            )
        if operation == "handshake":
            self.handshake_response_sha256 = sha256_bytes(
                canonical_json(response)
            )
        if self._callback is not None:
            self._callback("bridge_response", response)
        if (
            not isinstance(response, dict)
            or set(response)
            != {
                "schema",
                "kind",
                "attempt_id",
                "request_id",
                "sequence",
                "success",
                "result",
                "error",
            }
            or response["schema"] != SCHEMA
            or response["kind"]
            != "arc_agi3_contiguous_bridge_response"
            or response["attempt_id"] != self.attempt_id
            or response["request_id"] != request_id
            or response["sequence"] != next_sequence
            or not isinstance(response["success"], bool)
        ):
            raise AppServerTransportError("bridge response mismatch")
        self._completed_responses[request_id] = dict(response)
        self.sequence = next_sequence
        if mutating:
            self.mutation_sequence = next_mutation_sequence
        if not response["success"]:
            raise AppServerTransportError(
                "proposer bridge rejected operation: "
                + str(response["error"])
            )
        result = response["result"]
        boundary_sha256: str | None = None
        if (
            operation == "arena_step"
            and isinstance(result, dict)
            and result.get("target_reached") is True
        ):
            boundary_sha256 = _validate_target_boundary_result(
                result,
                attempt_id=self.attempt_id,
                request=request,
                target_level=(
                    self.handshake_result.get("target_level")
                    if isinstance(self.handshake_result, dict)
                    else None
                ),
            )
            callback = getattr(
                self, "_target_boundary_callback", None
            )
            if callback is not None:
                callback(request, response)
        if boundary_sha256 is not None:
            self.target_boundary_sha256 = boundary_sha256
        return result

    def close(self) -> None:
        self._socket.close()
        self._token = ""


def render_strict_config(
    *,
    model: str,
    model_provider: str,
    effort: str,
) -> str:
    """Render the complete zero-ambient app-server config projection."""

    if model != "gpt-5.6-sol" or model_provider != "openai":
        raise AppServerTransportError(
            "model/provider differs from the frozen projection"
        )
    if effort not in {"medium", "high", "xhigh", "max"}:
        raise AppServerTransportError("reasoning effort is unsupported")
    lines = [
        f'model = "{model}"',
        f'model_provider = "{model_provider}"',
        f'model_reasoning_effort = "{effort}"',
        'approval_policy = "never"',
        'sandbox_mode = "read-only"',
        'web_search = "disabled"',
        "",
        "[history]",
        # Thread/resume across a fresh app-server process is a production
        # requirement.  The state root is a private, lane-scoped staged copy,
        # so retaining the completed thread locally does not expose it to
        # another lane or to the proposer container.
        'persistence = "save-all"',
        "",
        "[memories]",
        "use_memories = false",
        "generate_memories = false",
        "",
        "[shell_environment_policy]",
        'inherit = "none"',
        "",
        "[features]",
    ]
    lines.extend(
        f"{feature} = false"
        for feature in SECURITY_DISABLED_FEATURES
    )
    for skill in DISABLED_SYSTEM_SKILLS:
        lines.extend(
            (
                "",
                "[[skills.config]]",
                f'name = "{skill}"',
                "enabled = false",
            )
        )
    return "\n".join(lines) + "\n"


def strict_config_projection(
    *,
    model: str,
    model_provider: str,
    effort: str,
) -> dict[str, Any]:
    """Return the exact config/read layer represented by the renderer."""

    # Reuse the renderer's validation contract.
    render_strict_config(
        model=model,
        model_provider=model_provider,
        effort=effort,
    )
    return {
        "approval_policy": "never",
        "features": {
            feature: False
            for feature in SECURITY_DISABLED_FEATURES
        },
        "history": {"persistence": "save-all"},
        "memories": {
            "generate_memories": False,
            "use_memories": False,
        },
        "model": model,
        "model_provider": model_provider,
        "model_reasoning_effort": effort,
        "sandbox_mode": "read-only",
        "shell_environment_policy": {"inherit": "none"},
        "skills": {
            "config": [
                {"enabled": False, "name": name}
                for name in DISABLED_SYSTEM_SKILLS
            ]
        },
        "web_search": "disabled",
    }


@dataclass(frozen=True)
class ControllerBinding:
    campaign_id: str
    generation_id: str
    attempt_id: str
    generation_dir: str
    state_root: str
    neutral_cwd: str
    neutral_host_cwd: str
    model: Literal["gpt-5.6-sol"]
    model_provider: Literal["openai"]
    reasoning_effort: Literal["medium", "high", "xhigh", "max"]
    thread_mode: Literal["new", "resume"]
    resume_thread_id: str | None
    hard_safety_seconds: Literal[21600]
    max_auth_refreshes: Literal[7]
    app_server_control_dir: str
    attempt_spec_sha256: str
    controller_canary_escrow_path: str
    controller_canary_escrow_sha256: str
    controller_canary_escrow_identity_sha256: str
    controller_canary_commitments_json: str
    controller_canary_commitments_sha256: str
    controller_canary_placement_descriptors_json: str
    controller_canary_placement_descriptors_sha256: str


@dataclass(frozen=True)
class ControllerContainerStart:
    """Host-observed authority for one attached controller container."""

    process: subprocess.Popen[bytes]
    controller_container_id: str
    controller_image_digest: str
    egress_proxy_container_id: str
    egress_proxy_image_digest: str
    egress_policy_sha256: str
    launch_intent_sha256: str
    launch_receipt_path: str
    launch_receipt_sha256: str
    guardian_start_receipt_path: str
    guardian_start_receipt_sha256: str
    codex_binary_sha256: str
    codex_binary_bytes: int
    supply_chain_manifest_sha256: str


@dataclass(frozen=True)
class ControllerContainerStop:
    """Authoritative full-ID/cgroup absence after controller containment."""

    controller_container_id: str
    egress_proxy_container_id: str
    controller_inspect_absent: Literal[True]
    controller_identity_query_empty: Literal[True]
    controller_top_absent: Literal[True]
    controller_no_descendants: Literal[True]
    egress_proxy_inspect_absent: Literal[True]
    egress_proxy_identity_query_empty: Literal[True]
    egress_proxy_top_absent: Literal[True]
    egress_proxy_no_descendants: Literal[True]
    absence_receipt_path: str
    absence_receipt_sha256: str


class ControllerContainerLauncher(Protocol):
    """Trusted Docker boundary; host PID facts are diagnostic only."""

    def start(
        self,
        *,
        binding: ControllerBinding,
        probe_spec: Any,
    ) -> ControllerContainerStart:
        ...

    def contain(
        self,
        *,
        binding: ControllerBinding,
        started: ControllerContainerStart,
    ) -> ControllerContainerStop:
        ...


@dataclass(frozen=True)
class ProviderUsageWindow:
    """Authenticated, typed provider budget observation.

    Denominations are intentionally not convertible.  In particular, a
    subscription percentage is never relabeled as credits, tokens, or USD.
    The canonical redacted JSON retains the complete admitted provider
    response without retaining credentials.
    """

    schema: Literal[1]
    phase: Literal["preflight", "postflight"]
    observation_sequence: int
    authenticated_response_sha256: str
    transcript_chain_sha256: str
    redacted_raw_snapshot_json: str
    redacted_raw_snapshot_sha256: str
    limit_id: str
    window_name: str
    window_duration_mins: int | None
    resets_at: int | None
    plan_type: str | None
    reset_credits_available: int | None
    authority: Literal[
        "explicit_unlimited",
        "explicit_finite",
        "legacy_percentage",
    ]
    denomination: Literal[
        "credits",
        "tokens",
        "usd",
        "subscription_percent",
    ]
    limit: float | None
    used: float | None
    remaining: float | None
    credits_unlimited: bool | None
    spend_control_reached: bool | None
    cost_control_enabled: bool
    cost_window_id: str
    window_sha256: str


@dataclass(frozen=True)
class ProviderUsageSettlement:
    """Attempt charge derived only from authenticated typed observations."""

    schema: Literal[1]
    pre_window_sha256: str
    post_window_sha256: str
    pre_authenticated_response_sha256: str
    post_authenticated_response_sha256: str
    cost_window_id: str
    post_cost_window_id: str
    denomination: Literal[
        "credits",
        "tokens",
        "usd",
        "subscription_percent",
    ]
    transition: Literal[
        "same_window",
        "cached_unlimited_legacy_postflight",
        "finite_window_reenabled",
    ]
    cost_control_enabled: bool
    limit: float | None
    charge: float
    requires_readmission: bool
    next_cost_window_id: str | None
    token_usage_observations_sha256: str
    token_total: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_output_tokens: int
    settlement_sha256: str


@dataclass(frozen=True)
class PreflightEvidence:
    schema: Literal[1]
    pid: int
    process_group_id: int
    process_start_identity: str
    codex_binary_sha256: str
    codex_binary_bytes: int
    initialize_params_sha256: str
    redacted_login_request_sha256: str
    dynamic_tool_specs_sha256: str
    base_instructions_sha256: str
    developer_instructions_sha256: str
    model: str
    model_provider: str
    reasoning_effort: str
    hard_safety_seconds: int
    max_auth_refreshes: int
    process_start_receipt_sha256: str
    process_identity_authority: Literal["controller_container_cgroup"]
    controller_container_id: str
    controller_image_digest: str
    egress_proxy_container_id: str
    egress_proxy_image_digest: str
    egress_policy_sha256: str
    controller_launch_intent_sha256: str
    controller_launch_receipt_path: str
    controller_launch_receipt_sha256: str
    guardian_start_receipt_path: str
    guardian_start_receipt_sha256: str
    supply_chain_manifest_sha256: str
    request_methods: tuple[str, ...]
    notification_counts: tuple[tuple[str, int], ...]
    response_sha256: tuple[tuple[str, str], ...]
    provider_usage_window: ProviderUsageWindow
    auth_mode: Literal["chatgptAuthTokens"]
    model_effort_supported: Literal[True]
    system_skills_disabled: Literal[True]
    hooks_empty: Literal[True]
    plugins_empty: Literal[True]
    apps_empty: Literal[True]
    experimental_features_disabled: Literal[True]
    mcp_servers_empty: Literal[True]
    stderr_empty: Literal[True]
    stderr_sha256: str
    stderr_bytes: Literal[0]
    path_alias_setup_status: Literal["PASS"]
    state_root: str
    initial_state_tree_sha256: str
    initialized_state_tree_sha256: str
    initialized_state_inventory_sha256: str
    initialized_state_file_count: int
    initialized_state_total_bytes: int
    state_database_path: str
    state_database_sha256: str
    state_database_bytes: int
    state_database_header_sha256: str
    state_database_initialized: Literal[True]
    transcript_chain_sha256: str


@dataclass(frozen=True)
class TurnStartEvidence:
    schema: Literal[1]
    thread_id: str
    turn_id: str
    thread_mode: Literal["new", "resume"]
    thread_request_sha256: str
    turn_request_sha256: str
    prompt_sha256: str
    transcript_chain_sha256: str


@dataclass(frozen=True)
class TurnFinalEvidence:
    schema: Literal[1]
    thread_id: str
    turn_id: str
    turn_status: Literal["completed", "interrupted", "failed"]
    provider_outcome: Literal[
        "completed",
        "capacity",
        "rate_limit",
        "provider_failure",
        "containment_fault",
    ]
    token_usage_observations: tuple[dict[str, Any], ...]
    pre_provider_usage_window: ProviderUsageWindow
    post_provider_usage_window: ProviderUsageWindow
    provider_usage_settlement: ProviderUsageSettlement
    final_model_text_sha256: str
    final_model_text: str
    tool_call_count: int
    hard_safety_seconds: int
    max_auth_refreshes: int
    auth_refresh_count: int
    redacted_auth_refresh_response_sha256: tuple[str, ...]
    credential_sentinel_scan_passed: Literal[True]
    post_turn_event_count: int
    stdout_bytes: int
    stderr_bytes: int
    pipes_drained_to_eof: Literal[True]
    transcript_chain_sha256: str
    transcript_event_count: int


@dataclass(frozen=True)
class ControllerTeardownEvidence:
    schema: Literal[1]
    pid: int
    process_group_id: int
    exit_code: int
    process_absent: bool
    process_group_absent: bool
    process_absent_receipt_sha256: str
    process_start_receipt_removed: Literal[True]
    ephemeral_tmp_purged: Literal[True]
    stderr_sha256: str
    stderr_bytes: int
    state_tree_sha256: str
    transcript_chain_sha256: str
    process_identity_authority: Literal["controller_container_cgroup"]
    controller_container_id: str
    egress_proxy_container_id: str
    controller_inspect_absent: Literal[True]
    controller_identity_query_empty: Literal[True]
    controller_top_absent: Literal[True]
    controller_no_descendants: Literal[True]
    egress_proxy_inspect_absent: Literal[True]
    egress_proxy_identity_query_empty: Literal[True]
    egress_proxy_top_absent: Literal[True]
    egress_proxy_no_descendants: Literal[True]
    controller_absence_receipt_sha256: str


def _write_new_bytes(path: Path, payload: bytes, *, mode: int) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        target,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(
        target.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return sha256_bytes(payload)


def _purge_ephemeral_directory(path: Path) -> None:
    """Descriptor-confined purge of the declared app-server tmp subtree."""

    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        root_fd = os.open(path, flags)
    except OSError as exc:
        raise AppServerTransportError(
            "app-server tmp root is missing or aliased"
        ) from exc

    def purge(directory_fd: int) -> None:
        try:
            names = sorted(os.listdir(directory_fd))
        except OSError as exc:
            raise AppServerTransportError(
                "app-server tmp directory cannot be enumerated"
            ) from exc
        for name in names:
            if (
                not isinstance(name, str)
                or name in {"", ".", ".."}
                or "/" in name
                or "\x00" in name
            ):
                raise AppServerTransportError(
                    "app-server tmp entry name is malformed"
                )
            try:
                metadata = os.stat(
                    name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise AppServerTransportError(
                    "app-server tmp entry cannot be inspected"
                ) from exc
            if stat.S_ISDIR(metadata.st_mode):
                try:
                    child_fd = os.open(
                        name, flags, dir_fd=directory_fd
                    )
                except OSError as exc:
                    raise AppServerTransportError(
                        "app-server tmp directory is aliased"
                    ) from exc
                try:
                    purge(child_fd)
                finally:
                    os.close(child_fd)
                try:
                    os.rmdir(name, dir_fd=directory_fd)
                except OSError as exc:
                    raise AppServerTransportError(
                        "app-server tmp directory survived purge"
                    ) from exc
            elif stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(
                metadata.st_mode
            ):
                try:
                    os.unlink(name, dir_fd=directory_fd)
                except OSError as exc:
                    raise AppServerTransportError(
                        "app-server tmp entry survived purge"
                    ) from exc
            else:
                raise AppServerTransportError(
                    "app-server tmp contains an inadmissible object"
                )
        try:
            os.fsync(directory_fd)
        except OSError as exc:
            raise AppServerTransportError(
                "app-server tmp purge could not be synchronized"
            ) from exc

    try:
        root_stat = os.fstat(root_fd)
        if (
            not stat.S_ISDIR(root_stat.st_mode)
            or root_stat.st_uid != os.getuid()
            or stat.S_IMODE(root_stat.st_mode) & 0o077
        ):
            raise AppServerTransportError(
                "app-server tmp root ownership/mode is unsafe"
            )
        purge(root_fd)
        if os.listdir(root_fd):
            raise AppServerTransportError(
                "app-server tmp root is not empty after purge"
            )
    finally:
        os.close(root_fd)


@dataclass(frozen=True)
class ControllerStateInventory:
    tree_sha256: str
    inventory_sha256: str
    file_count: int
    total_bytes: int
    files: tuple[tuple[str, str, int], ...]
    secret_occurrences: int

    def as_receipt(self) -> dict[str, Any]:
        return {
            "tree_sha256": self.tree_sha256,
            "inventory_sha256": self.inventory_sha256,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "files": [
                {"path": path, "sha256": digest, "bytes": byte_count}
                for path, digest, byte_count in self.files
            ],
        }


def inventory_controller_state(
    root: Path,
    *,
    sentinels: Sequence[str] = (),
) -> ControllerStateInventory:
    """Descriptor-confined bounded inventory and live-only secret scan."""

    selected = Path(root)
    if not selected.is_absolute():
        raise AppServerTransportError(
            "controller state root is not absolute"
        )
    encoded_sentinels = tuple(
        sorted(
            {
                value.encode("utf-8")
                for value in sentinels
                if isinstance(value, str) and value
            }
        )
    )
    root_descriptor = os.open(
        selected,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    rows: list[tuple[str, str, int]] = []
    total_bytes = 0
    secret_occurrences = 0

    def visit(
        directory_descriptor: int,
        relative_parent: PurePosixPath,
        depth: int,
    ) -> None:
        nonlocal total_bytes, secret_occurrences
        if depth > MAX_APP_SERVER_STATE_DEPTH:
            raise AppServerTransportError(
                "controller state exceeds its depth bound"
            )
        try:
            names = sorted(os.listdir(directory_descriptor))
        except OSError as exc:
            raise AppServerTransportError(
                "controller state directory cannot be enumerated"
            ) from exc
        for name in names:
            if (
                not isinstance(name, str)
                or not name
                or name in {".", ".."}
                or "/" in name
                or "\x00" in name
            ):
                raise AppServerTransportError(
                    "controller state entry name is unsafe"
                )
            relative = relative_parent / name
            relative_text = relative.as_posix()
            if len(relative_text.encode("utf-8")) > 4096:
                raise AppServerTransportError(
                    "controller state path exceeds its byte bound"
                )
            metadata = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if stat.S_ISDIR(metadata.st_mode):
                child_descriptor = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_descriptor,
                )
                try:
                    child_metadata = os.fstat(child_descriptor)
                    if (
                        not stat.S_ISDIR(child_metadata.st_mode)
                        or (child_metadata.st_dev, child_metadata.st_ino)
                        != (metadata.st_dev, metadata.st_ino)
                    ):
                        raise AppServerTransportError(
                            "controller state directory identity changed"
                        )
                    visit(child_descriptor, relative, depth + 1)
                finally:
                    os.close(child_descriptor)
                continue
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_size < 0
                or metadata.st_size > MAX_APP_SERVER_STATE_FILE_BYTES
            ):
                raise AppServerTransportError(
                    "controller state contains an aliased, special, or "
                    "oversized file"
                )
            if len(rows) >= MAX_APP_SERVER_STATE_FILES:
                raise AppServerTransportError(
                    "controller state exceeds its file-count bound"
                )
            total_bytes += metadata.st_size
            if total_bytes > MAX_APP_SERVER_STATE_TOTAL_BYTES:
                raise AppServerTransportError(
                    "controller state exceeds its aggregate byte bound"
                )
            descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_descriptor,
            )
            try:
                opened = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or (opened.st_dev, opened.st_ino, opened.st_size)
                    != (metadata.st_dev, metadata.st_ino, metadata.st_size)
                ):
                    raise AppServerTransportError(
                        "controller state file identity changed"
                    )
                digest = hashlib.sha256()
                observed = 0
                tail = b""
                maximum_sentinel = max(
                    (len(value) for value in encoded_sentinels),
                    default=0,
                )
                while True:
                    block = os.read(descriptor, 1024 * 1024)
                    if not block:
                        break
                    observed += len(block)
                    if observed > MAX_APP_SERVER_STATE_FILE_BYTES:
                        raise AppServerTransportError(
                            "controller state file grew beyond its bound"
                        )
                    digest.update(block)
                    searchable = tail + block
                    secret_occurrences += sum(
                        searchable.count(value) - tail.count(value)
                        for value in encoded_sentinels
                    )
                    tail = (
                        searchable[-(maximum_sentinel - 1):]
                        if maximum_sentinel > 1
                        else b""
                    )
                after = os.fstat(descriptor)
                if (
                    observed != metadata.st_size
                    or (
                        after.st_dev,
                        after.st_ino,
                        after.st_size,
                        after.st_mtime_ns,
                    )
                    != (
                        metadata.st_dev,
                        metadata.st_ino,
                        metadata.st_size,
                        metadata.st_mtime_ns,
                    )
                ):
                    raise AppServerTransportError(
                        "controller state changed during inventory"
                    )
                rows.append(
                    (relative_text, digest.hexdigest(), observed)
                )
            finally:
                os.close(descriptor)

    try:
        root_metadata = os.fstat(root_descriptor)
        if not stat.S_ISDIR(root_metadata.st_mode):
            raise AppServerTransportError(
                "controller state root is not a directory"
            )
        visit(root_descriptor, PurePosixPath("."), 0)
    finally:
        os.close(root_descriptor)
    normalized_rows = tuple(
        sorted(
            (
                path.removeprefix("./"),
                digest,
                byte_count,
            )
            for path, digest, byte_count in rows
        )
    )
    tree_digest = hashlib.sha256()
    for path, digest, _byte_count in normalized_rows:
        tree_digest.update(path.encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(digest.encode("ascii"))
        tree_digest.update(b"\n")
    inventory_payload = canonical_json(
        {
            "files": [
                {"path": path, "sha256": digest, "bytes": byte_count}
                for path, digest, byte_count in normalized_rows
            ],
            "file_count": len(normalized_rows),
            "total_bytes": total_bytes,
        }
    )
    return ControllerStateInventory(
        tree_sha256=tree_digest.hexdigest(),
        inventory_sha256=sha256_bytes(inventory_payload),
        file_count=len(normalized_rows),
        total_bytes=total_bytes,
        files=normalized_rows,
        secret_occurrences=secret_occurrences,
    )


def _regular_tree_sha256(root: Path) -> str:
    return inventory_controller_state(root).tree_sha256


def _result_rows(value: object) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in (
            "data",
            "items",
            "models",
            "skills",
            "hooks",
            "plugins",
            "apps",
            "features",
            "servers",
        ):
            rows = value.get(key)
            if isinstance(rows, list):
                return rows
    raise AppServerTransportError(
        "preflight inventory response has no exact list"
    )


def _provider_cost_window_id(
    *,
    limit_id: str,
    window_name: str,
    authority: str,
    denomination: str,
    window_duration_mins: int | None,
    resets_at: int | None,
) -> str:
    return sha256_bytes(canonical_json({
        "domain": "arc_agi3_provider_cost_window_v1",
        "limit_id": limit_id,
        "window_name": window_name,
        "authority": authority,
        "denomination": denomination,
        "window_duration_mins": window_duration_mins,
        "resets_at": resets_at,
    }))


def normalize_provider_usage_window(
    response: object,
    *,
    phase: Literal["preflight", "postflight"],
    observation_sequence: int,
    authenticated_response_sha256: str,
    transcript_chain_sha256: str,
) -> ProviderUsageWindow:
    """Strictly normalize one authenticated app-server rate-limit result."""

    if (
        phase not in {"preflight", "postflight"}
        or not isinstance(observation_sequence, int)
        or isinstance(observation_sequence, bool)
        or observation_sequence < 1
        or not SHA256_RE.fullmatch(authenticated_response_sha256)
        or not SHA256_RE.fullmatch(transcript_chain_sha256)
        or not isinstance(response, dict)
        or not set(response).issubset({
            "rateLimitsByLimitId",
            "rateLimits",
            "rateLimitResetCredits",
        })
        or "rateLimitsByLimitId" not in response
        or "rateLimits" in response
    ):
        raise AppServerTransportError(
            "rate-limit observation lacks an explicit authenticated identity"
        )
    buckets_value = response["rateLimitsByLimitId"]
    if (
        not isinstance(buckets_value, dict)
        or not buckets_value
        or any(
            not isinstance(limit_id, str)
            or re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", limit_id)
            is None
            for limit_id in buckets_value
        )
    ):
        raise AppServerTransportError(
            "rate-limit bucket identity schema differs"
        )
    reset_credits: int | None = None
    reset = response.get("rateLimitResetCredits")
    if reset is not None:
        if (
            not isinstance(reset, dict)
            or set(reset) != {"availableCount"}
            or not isinstance(reset["availableCount"], int)
            or isinstance(reset["availableCount"], bool)
            or reset["availableCount"] < 0
        ):
            raise AppServerTransportError(
                "rate-limit reset-credit schema differs"
            )
        reset_credits = reset["availableCount"]

    explicit: list[dict[str, Any]] = []
    weekly: list[dict[str, Any]] = []
    for limit_id, raw_bucket in sorted(buckets_value.items()):
        if (
            not isinstance(raw_bucket, dict)
            or not set(raw_bucket).issubset({
                "planType",
                "primary",
                "secondary",
                "credits",
                "spendControlReached",
            })
        ):
            raise AppServerTransportError(
                "rate-limit bucket schema differs"
            )
        plan_type = raw_bucket.get("planType")
        if plan_type is not None and (
            not isinstance(plan_type, str) or not plan_type
        ):
            raise AppServerTransportError(
                "rate-limit plan type is malformed"
            )
        spend_reached = raw_bucket.get("spendControlReached")
        if spend_reached is not None and not isinstance(
            spend_reached, bool
        ):
            raise AppServerTransportError(
                "rate-limit spend-control flag is malformed"
            )
        credits = raw_bucket.get("credits")
        if credits is not None:
            if (
                not isinstance(credits, dict)
                or set(credits)
                != {"hasCredits", "unlimited", "balance"}
                or not isinstance(credits["hasCredits"], bool)
                or not isinstance(credits["unlimited"], bool)
                or not (
                    credits["balance"] is None
                    or (
                        isinstance(credits["balance"], (int, float))
                        and not isinstance(credits["balance"], bool)
                        and math.isfinite(float(credits["balance"]))
                        and float(credits["balance"]) >= 0
                    )
                )
            ):
                raise AppServerTransportError(
                    "rate-limit credits schema differs"
                )
            # This exact conjunction is the only authority for limit=None.
            if (
                credits["unlimited"] is True
                and spend_reached is False
            ):
                explicit.append({
                    "limit_id": limit_id,
                    "window_name": "credits",
                    "window_duration_mins": None,
                    "resets_at": None,
                    "plan_type": plan_type,
                    "authority": "explicit_unlimited",
                    "denomination": "credits",
                    "limit": None,
                    "used": None,
                    "remaining": None,
                    "credits_unlimited": True,
                    "spend_control_reached": False,
                    "cost_control_enabled": False,
                })
            elif (
                credits["unlimited"] is False
                and credits["balance"] is not None
                and isinstance(spend_reached, bool)
            ):
                remaining = (
                    0.0
                    if spend_reached
                    else float(credits["balance"])
                )
                explicit.append({
                    "limit_id": limit_id,
                    "window_name": "credits",
                    "window_duration_mins": None,
                    "resets_at": None,
                    "plan_type": plan_type,
                    "authority": "explicit_finite",
                    "denomination": "credits",
                    "limit": remaining,
                    "used": 0.0,
                    "remaining": remaining,
                    "credits_unlimited": False,
                    "spend_control_reached": spend_reached,
                    "cost_control_enabled": True,
                })
        for window_name in ("primary", "secondary"):
            window = raw_bucket.get(window_name)
            if window is None:
                continue
            if (
                not isinstance(window, dict)
                or set(window)
                != {
                    "usedPercent",
                    "resetsAt",
                    "windowDurationMins",
                }
                or not isinstance(window["usedPercent"], int)
                or isinstance(window["usedPercent"], bool)
                or not 0 <= window["usedPercent"] <= 100
                or not isinstance(window["windowDurationMins"], int)
                or isinstance(window["windowDurationMins"], bool)
                or window["windowDurationMins"] <= 0
                or not (
                    window["resetsAt"] is None
                    or (
                        isinstance(window["resetsAt"], int)
                        and not isinstance(window["resetsAt"], bool)
                        and window["resetsAt"] > 0
                    )
                )
            ):
                raise AppServerTransportError(
                    "rate-limit window schema differs"
                )
            if window["windowDurationMins"] >= 7 * 24 * 60:
                used = float(window["usedPercent"])
                weekly.append({
                    "limit_id": limit_id,
                    "window_name": window_name,
                    "window_duration_mins":
                        window["windowDurationMins"],
                    "resets_at": window["resetsAt"],
                    "plan_type": plan_type,
                    "authority": "legacy_percentage",
                    "denomination": "subscription_percent",
                    "limit": 100.0 - used,
                    "used": used,
                    "remaining": 100.0 - used,
                    "credits_unlimited": (
                        credits["unlimited"]
                        if isinstance(credits, dict)
                        else None
                    ),
                    "spend_control_reached": spend_reached,
                    "cost_control_enabled": True,
                })

    if len(explicit) > 1:
        raise AppServerTransportError(
            "multiple explicit provider cost authorities are ambiguous"
        )
    if explicit:
        selected = explicit[0]
    else:
        if not weekly:
            raise AppServerTransportError(
                "rate-limit response proves no weekly or explicit cost window"
            )
        longest = max(
            item["window_duration_mins"] for item in weekly
        )
        finalists = [
            item for item in weekly
            if item["window_duration_mins"] == longest
        ]
        if len(finalists) != 1:
            raise AppServerTransportError(
                "multiple longest provider windows are ambiguous"
            )
        selected = finalists[0]

    raw_json = canonical_json(response).decode("ascii")
    cost_window_id = _provider_cost_window_id(
        limit_id=selected["limit_id"],
        window_name=selected["window_name"],
        authority=selected["authority"],
        denomination=selected["denomination"],
        window_duration_mins=selected["window_duration_mins"],
        resets_at=selected["resets_at"],
    )
    body = {
        "schema": 1,
        "phase": phase,
        "observation_sequence": observation_sequence,
        "authenticated_response_sha256":
            authenticated_response_sha256,
        "transcript_chain_sha256": transcript_chain_sha256,
        "redacted_raw_snapshot_json": raw_json,
        "redacted_raw_snapshot_sha256": sha256_bytes(
            raw_json.encode("ascii")
        ),
        **selected,
        "reset_credits_available": reset_credits,
        "cost_window_id": cost_window_id,
    }
    return ProviderUsageWindow(
        **body,
        window_sha256=sha256_bytes(canonical_json(body)),
    )


def provider_usage_window_to_dict(
    window: ProviderUsageWindow,
) -> dict[str, Any]:
    return asdict(window)


def provider_usage_window_from_dict(
    value: object,
) -> ProviderUsageWindow:
    if (
        not isinstance(value, dict)
        or set(value)
        != set(ProviderUsageWindow.__dataclass_fields__)
    ):
        raise AppServerTransportError(
            "provider usage window schema differs"
        )
    raw_json = value.get("redacted_raw_snapshot_json")
    if not isinstance(raw_json, str):
        raise AppServerTransportError(
            "provider usage window lacks its redacted raw snapshot"
        )
    raw = strict_json_loads(raw_json)
    expected = normalize_provider_usage_window(
        raw,
        phase=value.get("phase"),  # type: ignore[arg-type]
        observation_sequence=value.get("observation_sequence"),  # type: ignore[arg-type]
        authenticated_response_sha256=value.get(
            "authenticated_response_sha256"
        ),  # type: ignore[arg-type]
        transcript_chain_sha256=value.get(
            "transcript_chain_sha256"
        ),  # type: ignore[arg-type]
    )
    if asdict(expected) != value:
        raise AppServerTransportError(
            "provider usage window is stale, mutated, or unbound"
        )
    return expected


def settle_provider_usage(
    pre: ProviderUsageWindow,
    post: ProviderUsageWindow,
    *,
    token_usage_observations: Sequence[Mapping[str, Any]],
) -> ProviderUsageSettlement:
    """Settle without denomination conversion or window rotation."""

    if (
        not isinstance(pre, ProviderUsageWindow)
        or not isinstance(post, ProviderUsageWindow)
        or pre.phase != "preflight"
        or post.phase != "postflight"
        or post.observation_sequence <= pre.observation_sequence
        or pre.limit_id != post.limit_id
        or not token_usage_observations
    ):
        raise AppServerTransportError(
            "provider usage settlement inputs are stale or incompatible"
        )
    # Reparse both raw snapshots so a caller cannot hand-author typed fields.
    provider_usage_window_from_dict(asdict(pre))
    provider_usage_window_from_dict(asdict(post))
    final = token_usage_observations[-1]
    totals = final.get("total") if isinstance(final, Mapping) else None
    required_totals = {
        "inputTokens",
        "cachedInputTokens",
        "outputTokens",
        "reasoningOutputTokens",
        "totalTokens",
    }
    if (
        not isinstance(totals, Mapping)
        or set(totals) != required_totals
        or any(
            not isinstance(item, int)
            or isinstance(item, bool)
            or item < 0
            for item in totals.values()
        )
    ):
        raise AppServerTransportError(
            "provider usage settlement lacks exact raw token classes"
        )
    try:
        observations_sha256 = sha256_bytes(canonical_json(
            [dict(item) for item in token_usage_observations]
        ))
    except (TypeError, ValueError) as exc:
        raise AppServerTransportError(
            "provider token observations are not canonical JSON"
        ) from exc

    transition: Literal[
        "same_window",
        "cached_unlimited_legacy_postflight",
        "finite_window_reenabled",
    ]
    requires_readmission = False
    next_window: str | None = None
    if pre.authority == "explicit_unlimited":
        if (
            post.authority == "explicit_unlimited"
            and post.denomination == "credits"
            and post.cost_window_id == pre.cost_window_id
        ):
            transition = "same_window"
        elif (
            post.authority == "legacy_percentage"
            and post.denomination == "subscription_percent"
            and post.used == 0.0
            and post.remaining == 100.0
        ):
            # A legacy "100% remaining" result cannot itself authorize
            # unlimited.  It also cannot revoke the newer explicit preflight
            # assertion for this same limit ID.
            transition = "cached_unlimited_legacy_postflight"
        elif (
            post.authority == "explicit_finite"
            and post.denomination == "credits"
        ):
            transition = "finite_window_reenabled"
            requires_readmission = True
            next_window = post.cost_window_id
        else:
            raise AppServerTransportError(
                "unlimited provider authority changed ambiguously"
            )
        denomination = "credits"
        charge = 0.0
        limit = None
        cost_control_enabled = False
    else:
        if (
            post.authority != pre.authority
            or post.denomination != pre.denomination
            or post.cost_window_id != pre.cost_window_id
        ):
            raise AppServerTransportError(
                "provider window rotated or mixed denominations"
            )
        transition = "same_window"
        denomination = pre.denomination
        limit = pre.limit
        cost_control_enabled = True
        if denomination == "credits":
            if (
                pre.remaining is None
                or post.remaining is None
                or post.remaining > pre.remaining
            ):
                raise AppServerTransportError(
                    "provider credit balance moved backwards"
                )
            charge = pre.remaining - post.remaining
        elif denomination == "subscription_percent":
            if (
                pre.used is None
                or post.used is None
                or post.used < pre.used
            ):
                raise AppServerTransportError(
                    "provider percentage usage moved backwards"
                )
            charge = post.used - pre.used
        else:
            # The current pinned response does not expose authenticated
            # token/USD limits.  They remain distinct types for future pinned
            # schemas and can never enter through this parser by conversion.
            raise AppServerTransportError(
                "provider denomination has no pinned settlement rule"
            )
    if not math.isfinite(charge) or charge < 0:
        raise AppServerTransportError(
            "provider usage charge is invalid"
        )
    body = {
        "schema": 1,
        "pre_window_sha256": pre.window_sha256,
        "post_window_sha256": post.window_sha256,
        "pre_authenticated_response_sha256":
            pre.authenticated_response_sha256,
        "post_authenticated_response_sha256":
            post.authenticated_response_sha256,
        "cost_window_id": pre.cost_window_id,
        "post_cost_window_id": post.cost_window_id,
        "denomination": denomination,
        "transition": transition,
        "cost_control_enabled": cost_control_enabled,
        "limit": limit,
        "charge": float(charge),
        "requires_readmission": requires_readmission,
        "next_cost_window_id": next_window,
        "token_usage_observations_sha256": observations_sha256,
        "token_total": totals["totalTokens"],
        "input_tokens": totals["inputTokens"],
        "cached_input_tokens": totals["cachedInputTokens"],
        "output_tokens": totals["outputTokens"],
        "reasoning_output_tokens": totals["reasoningOutputTokens"],
    }
    return ProviderUsageSettlement(
        **body,
        settlement_sha256=sha256_bytes(canonical_json(body)),
    )


def provider_usage_settlement_from_dict(
    value: object,
    *,
    pre: ProviderUsageWindow,
    post: ProviderUsageWindow,
    token_usage_observations: Sequence[Mapping[str, Any]],
) -> ProviderUsageSettlement:
    if (
        not isinstance(value, dict)
        or set(value)
        != set(ProviderUsageSettlement.__dataclass_fields__)
    ):
        raise AppServerTransportError(
            "provider usage settlement schema differs"
        )
    expected = settle_provider_usage(
        pre,
        post,
        token_usage_observations=token_usage_observations,
    )
    if asdict(expected) != value:
        raise AppServerTransportError(
            "provider usage settlement is mutated or forged"
        )
    return expected


def _identifier_component(value: str | int) -> str:
    text = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", text):
        raise AppServerTransportError(
            "dynamic request/call identity is not path-safe"
        )
    return text


class CodexAppServerController:
    """One exact pinned app-server process and one active proposer turn."""

    def __init__(
        self,
        *,
        binary_path: Path | None,
        binding: ControllerBinding,
        transcript_path: Path,
        credentials: ExternalChatGptCredentials,
        bridge: BridgeClient,
        probe_executor: ProbeExecutor,
        probe_spec: Any,
        probe_launch: Any,
        container_launcher: ControllerContainerLauncher | None = None,
        response_timeout_seconds: float = 120.0,
    ) -> None:
        if (
            binding.hard_safety_seconds
            != APP_SERVER_HARD_SAFETY_SECONDS
            or binding.max_auth_refreshes != MAX_AUTH_REFRESHES
            or not SHA256_RE.fullmatch(
                str(binding.attempt_spec_sha256)
            )
            or not Path(binding.app_server_control_dir).is_absolute()
        ):
            raise AppServerTransportError(
                "controller binding weakens the frozen safety policy"
            )
        if container_launcher is None:
            raise AppServerTransportError(
                "production controller requires an isolated container launcher"
            )
        self.binary_path = (
            None if binary_path is None else Path(binary_path)
        )
        self.binding = binding
        self.transcript = ChainedTranscript(transcript_path)
        self.credentials = credentials
        self.bridge = bridge
        self.probe_executor = probe_executor
        self.probe_spec = probe_spec
        self.probe_launch = probe_launch
        self.container_launcher = container_launcher
        self.response_timeout_seconds = response_timeout_seconds
        self.process: subprocess.Popen[bytes] | None = None
        self._selector: selectors.BaseSelector | None = None
        self._stdout_buffer = bytearray()
        self._stderr_buffer = bytearray()
        self._stderr_complete = bytearray()
        self._stdout_bytes_observed = 0
        self._stderr_bytes_observed = 0
        self._stdout_eof = False
        self._stderr_eof = False
        self._allow_protocol_eof = False
        self._next_id = 0
        self._outstanding: dict[int, str] = {}
        self._server_requests: set[str | int] = set()
        self._request_methods: list[str] = []
        self._response_hashes: list[tuple[str, str]] = []
        self._notifications: dict[str, int] = {}
        self._login_notification_sequence: list[str] = []
        self._notification_emitted_at_ms = -1
        self._preflight_complete = False
        self._preflight_provider_usage_window: (
            ProviderUsageWindow | None
        ) = None
        self._turn_active = False
        self._thread_id: str | None = None
        self._turn_id: str | None = None
        self._token_usage: list[dict[str, Any]] = []
        self._model_text_parts: list[str] = []
        self._tool_call_count = 0
        self._refresh_count = 0
        self._credential_sentinels = set(credentials.leak_sentinels)
        self._credential_access_token_sha256 = {
            sha256_bytes(credentials.access_token.encode("utf-8"))
        }
        self._started_identity = ""
        self._process_group_id: int | None = None
        self._refresh_redacted_response_sha256: list[str] = []
        self._codex_binary_sha256 = ""
        self._codex_binary_bytes = 0
        self._process_start_receipt_path: Path | None = None
        self._process_start_receipt_sha256 = ""
        self._post_turn_event_count = 0
        self._container_start: ControllerContainerStart | None = None
        self._initial_state_inventory: ControllerStateInventory | None = None

    def _minimal_environment(self) -> dict[str, str]:
        state = Path(self.binding.state_root)
        temporary = state / "tmp"
        temporary.mkdir(parents=True, exist_ok=True, mode=0o700)
        return {
            "CODEX_HOME": str(state),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "TMPDIR": str(temporary),
        }

    def _prepare_state(self) -> None:
        state = Path(self.binding.state_root)
        neutral = Path(self.binding.neutral_host_cwd)
        generation = Path(self.binding.generation_dir)
        if (
            not state.is_absolute()
            or not neutral.is_absolute()
            or not generation.is_absolute()
            or state.is_symlink()
            or neutral.is_symlink()
            or generation.is_symlink()
            or not state.is_dir()
            or not neutral.is_dir()
            or not generation.is_dir()
            or any(neutral.iterdir())
        ):
            raise AppServerTransportError(
                "state root or neutral cwd is not pre-created and isolated"
            )
        if self.binding.neutral_cwd != "/controller-neutral":
            raise AppServerTransportError(
                "controller neutral cwd is not the fixed in-container root"
            )
        self._validate_nonproject_roots(
            (generation, state, neutral)
        )
        config = state / "config.toml"
        rendered = render_strict_config(
            model=self.binding.model,
            model_provider=self.binding.model_provider,
            effort=self.binding.reasoning_effort,
        ).encode("utf-8")
        if config.exists():
            observed = _bounded_regular_bytes(
                config, max_bytes=1024 * 1024
            )
            admitted_prior_configs = {
                render_strict_config(
                    model=self.binding.model,
                    model_provider=self.binding.model_provider,
                    effort=effort,
                ).encode("utf-8")
                for effort in ("medium", "high", "xhigh", "max")
            }
            if observed not in admitted_prior_configs:
                raise AppServerTransportError(
                    "existing lane config differs from frozen projection"
                )
            if observed != rendered:
                pending = state / (
                    ".config.toml.pending-" + uuid.uuid4().hex
                )
                _write_new_bytes(pending, rendered, mode=0o400)
                os.replace(pending, config)
                directory = os.open(
                    state,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                )
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
        else:
            _write_new_bytes(config, rendered, mode=0o400)

    def _state_initialization_evidence(
        self,
    ) -> tuple[ControllerStateInventory, str, int, str]:
        """Prove that the pinned app-server initialized its SQLite runtime.

        A successful JSON-RPC ``initialize`` response alone is not sufficient:
        the live host incident that motivated this boundary reached process
        startup but could not create the state database.  Reopen the exact
        bind-mounted state tree after initialization, require the pinned
        database artifact and its SQLite header, and bind its complete digest
        through the descriptor-confined inventory.
        """

        state_root = Path(self.binding.state_root)
        inventory = inventory_controller_state(state_root)
        database_rows = {
            path: (digest, byte_count)
            for path, digest, byte_count in inventory.files
            if path == CODEX_STATE_DATABASE_NAME
        }
        if set(database_rows) != {CODEX_STATE_DATABASE_NAME}:
            raise DeterministicControllerConfigurationError(
                "app-server initialization did not create the pinned "
                "SQLite state database",
                failure_code="controller_sqlite_state_initialization",
            )
        database = state_root / CODEX_STATE_DATABASE_NAME
        try:
            metadata = database.stat(follow_symlinks=False)
            descriptor = os.open(
                database,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                opened = os.fstat(descriptor)
                header = os.read(descriptor, len(SQLITE3_HEADER))
            finally:
                os.close(descriptor)
        except OSError as exc:
            raise DeterministicControllerConfigurationError(
                "app-server SQLite state database cannot be reopened",
                failure_code="controller_sqlite_state_initialization",
            ) from exc
        digest, byte_count = database_rows[CODEX_STATE_DATABASE_NAME]
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_nlink != 1
            or (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
            )
            != (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_size,
            )
            or metadata.st_size != byte_count
            or byte_count < len(SQLITE3_HEADER)
            or header != SQLITE3_HEADER
        ):
            raise DeterministicControllerConfigurationError(
                "app-server SQLite state database is not a complete "
                "unaliasable SQLite artifact",
                failure_code="controller_sqlite_state_initialization",
            )
        return (
            inventory,
            digest,
            byte_count,
            sha256_bytes(header),
        )

    @staticmethod
    def _module_source_root() -> Path | None:
        source = Path(__file__).resolve()
        for candidate in (source.parent, *source.parents):
            if (candidate / ".git").exists():
                return candidate
        return None

    @classmethod
    def _validate_nonproject_roots(
        cls, roots: Sequence[Path]
    ) -> None:
        source_root = cls._module_source_root()
        for raw_root in roots:
            root = Path(raw_root)
            try:
                resolved = root.resolve(strict=True)
            except OSError as exc:
                raise AppServerTransportError(
                    "isolated root cannot be resolved"
                ) from exc
            cursor = Path(resolved.anchor)
            for component in resolved.parts[1:]:
                cursor = cursor / component
                try:
                    if stat.S_ISLNK(os.lstat(cursor).st_mode):
                        raise AppServerTransportError(
                            "isolated root traverses a symlink"
                        )
                except OSError as exc:
                    raise AppServerTransportError(
                        "isolated root ancestry cannot be inspected"
                    ) from exc
            if source_root is not None:
                try:
                    resolved.relative_to(source_root)
                except ValueError:
                    pass
                else:
                    raise AppServerTransportError(
                        "campaign/state/neutral root is inside source tree"
                    )
            # config.toml is intentionally not a discovery marker.  Scan the
            # state root itself as well as every ancestor so a staged/resumed
            # state cannot smuggle project discovery through `.git`, AGENTS,
            # `.agents`, or `.codex`.
            marker_roots = (resolved, *resolved.parents)
            for ancestor in marker_roots:
                if any(
                    (ancestor / marker).exists()
                    for marker in PROJECT_DISCOVERY_MARKERS
                ):
                    raise AppServerTransportError(
                        "isolated root ancestry exposes project discovery"
                    )

    def start(self) -> None:
        if self.process is not None:
            raise AppServerTransportError(
                "app-server controller already started"
            )
        self._prepare_state()
        self._initial_state_inventory = inventory_controller_state(
            Path(self.binding.state_root)
        )
        expected_transport = getattr(
            self.probe_spec, "proposer_transport", None
        )
        expected_binary_sha256 = getattr(
            expected_transport, "codex_binary_sha256", None
        )
        expected_binary_bytes = getattr(
            expected_transport, "codex_binary_bytes", None
        )
        started = self.container_launcher.start(
            binding=self.binding,
            probe_spec=self.probe_spec,
        )
        if (
            not isinstance(started, ControllerContainerStart)
            or re.fullmatch(
                r"[0-9a-f]{64}", started.controller_container_id
            )
            is None
            or re.fullmatch(
                r"[0-9a-f]{64}", started.egress_proxy_container_id
            )
            is None
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                started.controller_image_digest,
            )
            is None
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                started.egress_proxy_image_digest,
            )
            is None
            or not SHA256_RE.fullmatch(started.egress_policy_sha256)
            or not SHA256_RE.fullmatch(started.launch_intent_sha256)
            or not SHA256_RE.fullmatch(started.launch_receipt_sha256)
            or not SHA256_RE.fullmatch(
                started.guardian_start_receipt_sha256
            )
            or not SHA256_RE.fullmatch(
                started.supply_chain_manifest_sha256
            )
            or started.codex_binary_sha256
            != expected_binary_sha256
            or started.codex_binary_bytes != expected_binary_bytes
            or started.controller_image_digest
            != getattr(
                expected_transport, "controller_image_digest", None
            )
            or started.egress_proxy_image_digest
            != getattr(
                expected_transport,
                "controller_egress_proxy_image_digest",
                None,
            )
            or started.egress_policy_sha256
            != getattr(
                expected_transport,
                "controller_egress_policy_sha256",
                None,
            )
        ):
            raise AppServerTransportError(
                "controller-container authority differs from attempt pins"
            )
        process = started.process
        if not isinstance(process, subprocess.Popen):
            raise AppServerTransportError(
                "controller attach handle is not a Popen process"
            )
        assert (
            process.stdin is not None
            and process.stdout is not None
            and process.stderr is not None
        )
        self.process = process
        self._container_start = started
        self._codex_binary_sha256 = started.codex_binary_sha256
        self._codex_binary_bytes = started.codex_binary_bytes
        try:
            self._process_group_id = os.getpgid(process.pid)
            if self._process_group_id != process.pid:
                raise AppServerTransportError(
                    "controller attach did not enter its private process group"
                )
            self._started_identity = (
                observe_os_process_start_identity(process.pid)
            )
        except BaseException:
            try:
                self.container_launcher.contain(
                    binding=self.binding,
                    started=started,
                )
            except BaseException:
                pass
            raise
        control_root = Path(
            self.binding.app_server_control_dir
        )
        if (
            control_root.is_symlink()
            or not control_root.is_dir()
            or stat.S_IMODE(
                control_root.stat(follow_symlinks=False).st_mode
            )
            & 0o077
        ):
            raise AppServerTransportError(
                "app-server process control root is unsafe"
            )
        self._process_start_receipt_sha256 = (
            started.launch_receipt_sha256
        )
        self._process_start_receipt_path = None
        os.set_blocking(process.stdout.fileno(), False)
        os.set_blocking(process.stderr.fileno(), False)
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        self._selector = selector

    def _write_wire(
        self,
        payload: Mapping[str, Any],
        *,
        direction: Literal[
            "client_request",
            "client_notification",
            "client_response",
        ],
        transcript_payload: Mapping[str, Any] | None = None,
    ) -> None:
        if self.process is None or self.process.stdin is None:
            raise AppServerTransportError("app-server is not running")
        raw = canonical_json(dict(payload)) + b"\n"
        try:
            self.process.stdin.write(raw)
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise AppServerTransportError(
                "app-server input pipe failed"
            ) from exc
        self.transcript.append(
            direction=direction,
            payload=(
                dict(transcript_payload)
                if transcript_payload is not None
                else dict(payload)
            ),
        )

    def _send_request(
        self,
        method: str,
        params: Mapping[str, Any],
        *,
        redacted_params: Mapping[str, Any] | None = None,
    ) -> int:
        self._next_id += 1
        request_id = self._next_id
        payload = {
            "id": request_id,
            "method": method,
            "params": dict(params),
        }
        logged = (
            {
                "id": request_id,
                "method": method,
                "params": dict(redacted_params),
            }
            if redacted_params is not None
            else payload
        )
        self._write_wire(
            payload,
            direction="client_request",
            transcript_payload=logged,
        )
        self._outstanding[request_id] = method
        self._request_methods.append(method)
        return request_id

    def _send_notification(
        self, method: str, params: Mapping[str, Any] | None = None
    ) -> None:
        payload: dict[str, Any] = {"method": method}
        if params is not None:
            payload["params"] = dict(params)
        self._write_wire(payload, direction="client_notification")
        self._notifications[method] = (
            self._notifications.get(method, 0) + 1
        )

    def _read_ready_line(self, timeout: float) -> tuple[str, bytes]:
        assert self._selector is not None
        deadline = time.monotonic() + timeout
        while True:
            # A single pipe read commonly coalesces several app-server JSONL
            # events.  Drain complete retained lines before waiting for a new
            # kernel readiness edge.  Stderr-first is the frozen cross-stream
            # ordering policy; order within each stream remains byte order.
            for stream_name, buffer in (
                ("stderr", self._stderr_buffer),
                ("stdout", self._stdout_buffer),
            ):
                if b"\n" in buffer:
                    line, _, remainder = buffer.partition(b"\n")
                    buffer[:] = remainder
                    if stream_name == "stderr":
                        self._stderr_complete.extend(line + b"\n")
                    return stream_name, bytes(line)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AppServerTransportError(
                    "app-server response timed out"
                )
            events = self._selector.select(remaining)
            if not events:
                continue
            for key, _mask in sorted(
                events,
                key=lambda event: (
                    0 if event[0].data == "stderr" else 1
                ),
            ):
                stream = key.fileobj
                try:
                    block = os.read(stream.fileno(), 65536)
                except BlockingIOError:
                    continue
                if not block:
                    try:
                        self._selector.unregister(stream)
                    except (KeyError, ValueError):
                        pass
                    if key.data == "stdout":
                        self._stdout_eof = True
                    else:
                        self._stderr_eof = True
                    if self._stdout_eof and self._stderr_eof:
                        if self._stdout_buffer or self._stderr_buffer:
                            raise AppServerTransportError(
                                "app-server ended with a partial output line"
                            )
                        if self._allow_protocol_eof:
                            raise _ProtocolEof
                        raise AppServerTransportError(
                            "app-server exited before protocol completion"
                        )
                    continue
                if key.data == "stdout":
                    self._stdout_bytes_observed += len(block)
                else:
                    self._stderr_bytes_observed += len(block)
                buffer = (
                    self._stdout_buffer
                    if key.data == "stdout"
                    else self._stderr_buffer
                )
                buffer.extend(block)
                if len(buffer) > MAX_BRIDGE_LINE_BYTES:
                    raise AppServerTransportError(
                        "app-server emitted an oversized line"
                    )

    def _read_protocol_event(self, timeout: float) -> dict[str, Any]:
        while True:
            stream, line = self._read_ready_line(timeout)
            if stream == "stderr":
                try:
                    text = line.decode("utf-8")
                except UnicodeError as exc:
                    raise AppServerTransportError(
                        "app-server stderr is not UTF-8"
                    ) from exc
                self.transcript.append(
                    direction="server_stderr", payload=text
                )
                continue
            value = strict_json_loads(line)
            if not isinstance(value, dict):
                raise AppServerTransportError(
                    "app-server JSONL event is not an object"
                )
            has_method = "method" in value
            has_id = "id" in value
            if has_method and has_id:
                direction = "server_request"
            elif has_method:
                direction = "server_notification"
            elif has_id:
                direction = "server_response"
            else:
                raise AppServerTransportError(
                    "app-server event has no method or response id"
                )
            transcript_value: Mapping[str, Any] = value
            if (
                direction == "server_request"
                and value.get("method")
                == "account/chatgptAuthTokens/refresh"
            ):
                params = value.get("params")
                if isinstance(params, dict):
                    redacted_params = dict(params)
                    if "previousAccountId" in redacted_params:
                        redacted_params["previousAccountId"] = "REDACTED"
                    transcript_value = {
                        **value,
                        "params": redacted_params,
                    }
            self.transcript.append(
                direction=direction, payload=transcript_value
            )
            if direction == "server_notification":
                self._observe_notification(value)
            return value

    def _observe_notification(self, value: Mapping[str, Any]) -> None:
        method = value.get("method")
        if not isinstance(method, str):
            raise AppServerTransportError(
                "server notification method is malformed"
            )
        emitted = value.get("emittedAtMs")
        if emitted is not None:
            if (
                not isinstance(emitted, int)
                or isinstance(emitted, bool)
                or emitted < self._notification_emitted_at_ms
            ):
                raise AppServerTransportError(
                    "notification emittedAtMs is invalid/nonmonotone"
                )
            self._notification_emitted_at_ms = emitted
        self._notifications[method] = (
            self._notifications.get(method, 0) + 1
        )
        params = value.get("params")
        if method == "account/login/completed":
            if params != {
                "error": None,
                "loginId": None,
                "success": True,
            }:
                raise AppServerTransportError(
                    "external ChatGPT login notification failed"
                )
            self._login_notification_sequence.append(method)
        elif method == "account/updated":
            if (
                not isinstance(params, dict)
                or set(params) != {"authMode", "planType"}
                or params.get("authMode") != "chatgptAuthTokens"
                or not isinstance(params.get("planType"), str)
                or not params["planType"]
            ):
                raise AppServerTransportError(
                    "external ChatGPT account update is malformed"
                )
            if not self._preflight_complete:
                self._login_notification_sequence.append(method)
        if self._login_notification_sequence not in (
            [],
            ["account/login/completed"],
            ["account/login/completed", "account/updated"],
        ):
            raise AppServerTransportError(
                "external ChatGPT login notification order/cardinality failed"
            )
        if method == "thread/tokenUsage/updated":
            if not isinstance(params, dict):
                raise AppServerTransportError(
                    "token usage notification is malformed"
                )
            self._token_usage.append(dict(params))
        elif method == "item/agentMessage/delta":
            if (
                isinstance(params, dict)
                and isinstance(params.get("delta"), str)
            ):
                self._model_text_parts.append(params["delta"])

    def _wait_response(
        self,
        request_id: int,
        *,
        allow_tool_calls: bool = False,
    ) -> Any:
        deadline = time.monotonic() + self.response_timeout_seconds
        while True:
            event = self._read_protocol_event(
                max(0.001, deadline - time.monotonic())
            )
            if "method" in event and "id" in event:
                if not allow_tool_calls:
                    raise AppServerTransportError(
                        "server request occurred outside active turn"
                    )
                self._handle_server_request(event)
                continue
            if "method" in event:
                continue
            response_id = event.get("id")
            if response_id not in self._outstanding:
                raise AppServerTransportError(
                    "server response id is unknown or duplicated"
                )
            method = self._outstanding.pop(response_id)
            if response_id != request_id:
                raise AppServerTransportError(
                    "server responses arrived out of admitted order"
                )
            if set(event) not in (
                {"id", "result"},
                {"id", "error"},
            ):
                raise AppServerTransportError(
                    "server response envelope is not exact"
                )
            self._response_hashes.append(
                (method, sha256_bytes(canonical_json(event)))
            )
            if "error" in event:
                raise AppServerTransportError(
                    f"app-server request failed: {method}"
                )
            return event["result"]

    def _request(
        self,
        method: str,
        params: Mapping[str, Any],
        *,
        redacted_params: Mapping[str, Any] | None = None,
        allow_tool_calls: bool = False,
    ) -> Any:
        request_id = self._send_request(
            method, params, redacted_params=redacted_params
        )
        return self._wait_response(
            request_id, allow_tool_calls=allow_tool_calls
        )

    def _request_provider_usage_window(
        self,
        phase: Literal["preflight", "postflight"],
    ) -> ProviderUsageWindow:
        result = self._request("account/rateLimits/read", {})
        if (
            not self._response_hashes
            or self._response_hashes[-1][0]
            != "account/rateLimits/read"
            or self.transcript.head is None
        ):
            raise AppServerTransportError(
                "provider usage response lacks transcript authentication"
            )
        return normalize_provider_usage_window(
            result,
            phase=phase,
            observation_sequence=self._next_id,
            authenticated_response_sha256=(
                self._response_hashes[-1][1]
            ),
            transcript_chain_sha256=self.transcript.head,
        )

    def _validate_preflight_results(
        self, results: Mapping[str, Any]
    ) -> None:
        initialize = results["initialize"]
        if (
            not isinstance(initialize, dict)
            or initialize.get("codexHome")
            != self.binding.state_root
        ):
            raise AppServerTransportError(
                "initialize result loaded another Codex home"
            )
        login = results["account/login/start"]
        if (
            not isinstance(login, dict)
            or login.get("type") != "chatgptAuthTokens"
        ):
            raise AppServerTransportError(
                "external-token login result is not exact"
            )
        account = results["account/read"]
        if (
            not isinstance(account, dict)
            or set(account) != {"account", "requiresOpenaiAuth"}
            or account["requiresOpenaiAuth"] is not True
            or not isinstance(account["account"], dict)
            or account["account"].get("type") != "chatgpt"
        ):
            raise AppServerTransportError(
                "account/read did not prove external ChatGPT auth"
            )
        models = _result_rows(results["model/list"])
        matching = [
            row
            for row in models
            if isinstance(row, dict)
            and (
                row.get("id") == self.binding.model
                or row.get("model") == self.binding.model
                or row.get("slug") == self.binding.model
            )
        ]
        if len(matching) != 1:
            raise AppServerTransportError(
                "model/list did not return the exact pinned model once"
            )
        efforts = matching[0].get("supportedReasoningEfforts")
        if (
            not isinstance(efforts, list)
            or sum(
                1
                for row in efforts
                if isinstance(row, dict)
                and row.get("reasoningEffort")
                == self.binding.reasoning_effort
            )
            != 1
        ):
            raise AppServerTransportError(
                "model/list did not prove requested effort support"
            )
        model_result = results["model/list"]
        if (
            not isinstance(model_result, dict)
            or model_result.get("nextCursor") is not None
        ):
            raise AppServerTransportError(
                "model/list pagination is incomplete"
            )
        capabilities = results["modelProvider/capabilities/read"]
        if (
            not isinstance(capabilities, dict)
            or set(capabilities)
            != {"imageGeneration", "namespaceTools", "webSearch"}
            or capabilities.get("namespaceTools") is not True
            or not isinstance(capabilities.get("imageGeneration"), bool)
            or not isinstance(capabilities.get("webSearch"), bool)
        ):
            raise AppServerTransportError(
                "model provider capability inventory is malformed"
            )
        config = results["config/read"]
        if (
            not isinstance(config, dict)
            or set(config) != {"config", "layers", "origins"}
            or not isinstance(config.get("config"), dict)
            or not isinstance(config.get("origins"), dict)
            or not isinstance(config.get("layers"), list)
        ):
            raise AppServerTransportError("config/read result is malformed")
        effective = config["config"]
        exact_effective = {
            "model": self.binding.model,
            "model_provider": self.binding.model_provider,
            "model_reasoning_effort": self.binding.reasoning_effort,
            "approval_policy": "never",
            "sandbox_mode": "read-only",
            "web_search": "disabled",
        }
        if any(
            effective.get(name) != expected
            for name, expected in exact_effective.items()
        ):
            raise AppServerTransportError(
                "config/read differs from exact critical projection"
            )
        apps_config = effective.get("apps")
        if apps_config not in (None, {}, {"_default": None}):
            raise AppServerTransportError(
                "config/read exposed app authority"
            )
        effective_features = effective.get("features")
        if (
            not isinstance(effective_features, dict)
            or any(
                effective_features.get(name) is not False
                for name in SECURITY_DISABLED_FEATURES
            )
        ):
            raise AppServerTransportError(
                "config/read did not disable the security feature set"
            )
        expected_layer = strict_config_projection(
            model=self.binding.model,
            model_provider=self.binding.model_provider,
            effort=self.binding.reasoning_effort,
        )
        state_config = str(
            (Path(self.binding.state_root) / "config.toml").resolve()
        )
        user_layers = []
        for layer in config["layers"]:
            if (
                not isinstance(layer, dict)
                or set(layer) != {"config", "name", "version"}
                or not isinstance(layer["config"], dict)
                or not isinstance(layer["name"], dict)
                or not isinstance(layer["version"], str)
            ):
                raise AppServerTransportError(
                    "config/read layer is malformed"
                )
            layer_type = layer["name"].get("type")
            if layer_type == "system":
                if layer["config"] != {}:
                    raise AppServerTransportError(
                        "system config layer is not empty"
                    )
            elif layer_type == "user":
                if (
                    layer["name"].get("file") != state_config
                    or layer["config"] != expected_layer
                ):
                    raise AppServerTransportError(
                        "user config layer is not the pinned projection"
                    )
                user_layers.append(layer)
            else:
                raise AppServerTransportError(
                    "project/profile config layer is forbidden"
                )
        if len(user_layers) != 1:
            raise AppServerTransportError(
                "exactly one private user config layer is required"
            )
        for origin in config["origins"].values():
            if (
                not isinstance(origin, dict)
                or not isinstance(origin.get("name"), dict)
                or origin["name"].get("type") != "user"
                or origin["name"].get("file") != state_config
            ):
                raise AppServerTransportError(
                    "config origin escaped the private projection"
                )
        skill_response = results["skills/list"]
        if (
            not isinstance(skill_response, dict)
            or set(skill_response) != {"data"}
            or not isinstance(skill_response["data"], list)
            or len(skill_response["data"]) != 1
        ):
            raise AppServerTransportError(
                "skills/list response is not one exact cwd inventory"
            )
        skill_entry = skill_response["data"][0]
        if (
            not isinstance(skill_entry, dict)
            or set(skill_entry) != {"cwd", "errors", "skills"}
            or skill_entry["cwd"] != self.binding.neutral_cwd
            or skill_entry["errors"] != []
            or not isinstance(skill_entry["skills"], list)
        ):
            raise AppServerTransportError(
                "skills/list cwd/errors are unsafe"
            )
        skills = skill_entry["skills"]
        skill_names: set[str] = set()
        for row in skills:
            if not isinstance(row, dict):
                raise AppServerTransportError(
                    "skills/list contains a malformed row"
                )
            name = row.get("name")
            enabled = row.get("enabled")
            if (
                not isinstance(name, str)
                or name in skill_names
                or enabled is not False
                or row.get("scope") != "system"
            ):
                raise AppServerTransportError(
                    "system skill is unknown, duplicated, or enabled"
                )
            skill_names.add(name)
        if skill_names != set(DISABLED_SYSTEM_SKILLS):
            raise AppServerTransportError(
                "skills/list differs from exact disabled system inventory"
            )
        hooks = results["hooks/list"]
        if (
            not isinstance(hooks, dict)
            or set(hooks) != {"data"}
            or hooks.get("data")
            != [
                {
                    "cwd": self.binding.neutral_cwd,
                    "errors": [],
                    "hooks": [],
                    "warnings": [],
                }
            ]
        ):
            raise AppServerTransportError(
                "hooks/list returned errors or hook authority"
            )
        plugins = results["plugin/list"]
        if (
            not isinstance(plugins, dict)
            or set(plugins)
            - {
                "featuredPluginIds",
                "marketplaceLoadErrors",
                "marketplaces",
            }
            or plugins.get("marketplaces") != []
            or plugins.get("featuredPluginIds", []) != []
            or plugins.get("marketplaceLoadErrors", []) != []
        ):
            raise AppServerTransportError(
                "plugin/list returned plugin authority or errors"
            )
        for method in (
            "app/list",
            "mcpServerStatus/list",
        ):
            response = results[method]
            if (
                not isinstance(response, dict)
                or set(response) - {"data", "nextCursor"}
                or response.get("data") != []
                or response.get("nextCursor") is not None
            ):
                raise AppServerTransportError(
                    f"{method} returned authority or incomplete pagination"
                )
        feature_response = results["experimentalFeature/list"]
        if (
            not isinstance(feature_response, dict)
            or set(feature_response) - {"data", "nextCursor"}
            or feature_response.get("nextCursor") is not None
        ):
            raise AppServerTransportError(
                "experimental feature pagination is incomplete"
            )
        features = feature_response.get("data")
        if not isinstance(features, list):
            raise AppServerTransportError(
                "experimental feature inventory is malformed"
            )
        feature_by_name: dict[str, Mapping[str, Any]] = {}
        for row in features:
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("name"), str)
                or row["name"] in feature_by_name
                or not isinstance(row.get("enabled"), bool)
            ):
                raise AppServerTransportError(
                    "experimental feature inventory is malformed"
                )
            feature_by_name[row["name"]] = row
        if any(
            name not in feature_by_name
            or feature_by_name[name].get("enabled") is not False
            for name in SECURITY_DISABLED_FEATURES
        ):
            raise AppServerTransportError(
                "security feature is absent or enabled"
            )

    def start_and_preflight(self) -> PreflightEvidence:
        self.start()
        results: dict[str, Any] = {}
        results["initialize"] = self._request(
            "initialize", INITIALIZE_PARAMS
        )
        self._send_notification("initialized", {})
        results["account/login/start"] = self._request(
            "account/login/start",
            self.credentials.login_params(),
            redacted_params=self.credentials.redacted_login_params(),
        )
        login_deadline = (
            time.monotonic() + self.response_timeout_seconds
        )
        while any(
            self._notifications.get(method, 0) != 1
            for method in (
                "account/login/completed",
                "account/updated",
            )
        ):
            event = self._read_protocol_event(
                max(0.001, login_deadline - time.monotonic())
            )
            if (
                "id" in event
                or event.get("method")
                not in {
                    "account/login/completed",
                    "account/updated",
                    "account/rateLimits/updated",
                }
            ):
                raise AppServerTransportError(
                    "login notification barrier received another phase"
                )
        if self._login_notification_sequence != [
            "account/login/completed",
            "account/updated",
        ]:
            raise AppServerTransportError(
                "login notification barrier is incomplete"
            )
        probe_params: tuple[tuple[str, dict[str, Any]], ...] = (
            ("account/read", {"refreshToken": False}),
            ("account/rateLimits/read", {}),
            (
                "model/list",
                {
                    "cursor": None,
                    "includeHidden": True,
                    "limit": 100,
                },
            ),
            ("modelProvider/capabilities/read", {}),
            (
                "config/read",
                {
                    "cwd": self.binding.neutral_cwd,
                    "includeLayers": True,
                },
            ),
            (
                "skills/list",
                {
                    "cwds": [self.binding.neutral_cwd],
                    "forceReload": True,
                },
            ),
            ("hooks/list", {"cwds": [self.binding.neutral_cwd]}),
            (
                "plugin/list",
                {
                    "cwds": [self.binding.neutral_cwd],
                    "marketplaceKinds": [
                        "local",
                        "vertical",
                        "workspace-directory",
                    ],
                },
            ),
            (
                "app/list",
                {
                    "cursor": None,
                    "forceRefetch": False,
                    "limit": 100,
                    "threadId": None,
                },
            ),
            (
                "experimentalFeature/list",
                {
                    "cursor": None,
                    "limit": 100,
                    "threadId": None,
                },
            ),
            (
                "mcpServerStatus/list",
                {
                    "cursor": None,
                    "detail": "full",
                    "limit": 100,
                    "threadId": None,
                },
            ),
        )
        for method, params in probe_params:
            if method == "account/rateLimits/read":
                usage_window = self._request_provider_usage_window(
                    "preflight"
                )
                self._preflight_provider_usage_window = usage_window
                results[method] = strict_json_loads(
                    usage_window.redacted_raw_snapshot_json
                )
            else:
                results[method] = self._request(method, params)
        self._validate_preflight_results(results)
        if (
            tuple(self._request_methods)
            != PREFLIGHT_REQUEST_SEQUENCE
            or self._preflight_provider_usage_window is None
        ):
            raise AppServerTransportError(
                "preflight request sequence lacks provider budget evidence"
            )
        for method, required in (
            ("initialized", 1),
            ("account/login/completed", 1),
            ("account/updated", 1),
        ):
            if self._notifications.get(method, 0) != required:
                raise AppServerTransportError(
                    f"preflight notification cardinality failed: {method}"
                )
        if self._stderr_complete:
            raise DeterministicControllerConfigurationError(
                "app-server PATH-alias/startup preflight wrote stderr",
                failure_code=(
                    "controller_path_alias_or_startup_stderr"
                ),
            )
        if self._initial_state_inventory is None:
            raise AppServerTransportError(
                "app-server preflight lacks its initial state inventory"
            )
        (
            initialized_state,
            state_database_sha256,
            state_database_bytes,
            state_database_header_sha256,
        ) = self._state_initialization_evidence()
        self._credential_sentinel_scan()
        self._preflight_complete = True
        assert (
            self.process is not None
            and self.transcript.head is not None
            and self._container_start is not None
        )
        container_start = self._container_start
        return PreflightEvidence(
            schema=1,
            pid=self.process.pid,
            process_group_id=int(self._process_group_id),
            process_start_identity=self._started_identity,
            codex_binary_sha256=self._codex_binary_sha256,
            codex_binary_bytes=self._codex_binary_bytes,
            initialize_params_sha256=INITIALIZE_PARAMS_SHA256,
            redacted_login_request_sha256=
                self.credentials.redacted_request_sha256(),
            dynamic_tool_specs_sha256=DYNAMIC_TOOL_SPECS_SHA256,
            base_instructions_sha256=BASE_INSTRUCTIONS_SHA256,
            developer_instructions_sha256=
                DEVELOPER_INSTRUCTIONS_SHA256,
            model=self.binding.model,
            model_provider=self.binding.model_provider,
            reasoning_effort=self.binding.reasoning_effort,
            hard_safety_seconds=self.binding.hard_safety_seconds,
            max_auth_refreshes=self.binding.max_auth_refreshes,
            process_start_receipt_sha256=(
                self._process_start_receipt_sha256
            ),
            process_identity_authority="controller_container_cgroup",
            controller_container_id=(
                container_start.controller_container_id
            ),
            controller_image_digest=(
                container_start.controller_image_digest
            ),
            egress_proxy_container_id=(
                container_start.egress_proxy_container_id
            ),
            egress_proxy_image_digest=(
                container_start.egress_proxy_image_digest
            ),
            egress_policy_sha256=(
                container_start.egress_policy_sha256
            ),
            controller_launch_intent_sha256=(
                container_start.launch_intent_sha256
            ),
            controller_launch_receipt_sha256=(
                container_start.launch_receipt_sha256
            ),
            controller_launch_receipt_path=(
                container_start.launch_receipt_path
            ),
            guardian_start_receipt_sha256=(
                container_start.guardian_start_receipt_sha256
            ),
            guardian_start_receipt_path=(
                container_start.guardian_start_receipt_path
            ),
            supply_chain_manifest_sha256=(
                container_start.supply_chain_manifest_sha256
            ),
            request_methods=tuple(self._request_methods),
            notification_counts=tuple(
                sorted(self._notifications.items())
            ),
            response_sha256=tuple(self._response_hashes),
            provider_usage_window=(
                self._preflight_provider_usage_window
            ),
            auth_mode="chatgptAuthTokens",
            model_effort_supported=True,
            system_skills_disabled=True,
            hooks_empty=True,
            plugins_empty=True,
            apps_empty=True,
            experimental_features_disabled=True,
            mcp_servers_empty=True,
            stderr_empty=True,
            stderr_sha256=sha256_bytes(b""),
            stderr_bytes=0,
            path_alias_setup_status="PASS",
            state_root=str(Path(self.binding.state_root)),
            initial_state_tree_sha256=(
                self._initial_state_inventory.tree_sha256
            ),
            initialized_state_tree_sha256=(
                initialized_state.tree_sha256
            ),
            initialized_state_inventory_sha256=(
                initialized_state.inventory_sha256
            ),
            initialized_state_file_count=(
                initialized_state.file_count
            ),
            initialized_state_total_bytes=(
                initialized_state.total_bytes
            ),
            state_database_path=CODEX_STATE_DATABASE_NAME,
            state_database_sha256=state_database_sha256,
            state_database_bytes=state_database_bytes,
            state_database_header_sha256=(
                state_database_header_sha256
            ),
            state_database_initialized=True,
            transcript_chain_sha256=self.transcript.head,
        )

    def _thread_params(self) -> tuple[str, dict[str, Any]]:
        common = {
            "approvalPolicy": "never",
            "baseInstructions": BASE_INSTRUCTIONS,
            "cwd": self.binding.neutral_cwd,
            "developerInstructions": DEVELOPER_INSTRUCTIONS,
            "model": self.binding.model,
            "modelProvider": self.binding.model_provider,
            "runtimeWorkspaceRoots": [self.binding.neutral_cwd],
            "sandbox": "read-only",
        }
        if self.binding.thread_mode == "resume":
            if not self.binding.resume_thread_id:
                raise AppServerTransportError(
                    "resume mode lacks exact thread id"
                )
            return (
                "thread/resume",
                {
                    **common,
                    "threadId": self.binding.resume_thread_id,
                    "excludeTurns": False,
                },
            )
        return (
            "thread/start",
            {
                **common,
                "allowProviderModelFallback": False,
                "dynamicTools": list(DYNAMIC_TOOL_SPECS),
                "environments": [],
                "ephemeral": False,
                "experimentalRawEvents": False,
                "historyMode": "paginated",
                "selectedCapabilityRoots": [],
            },
        )

    def start_turn(
        self, *, frontier_brief: Mapping[str, Any]
    ) -> TurnStartEvidence:
        if not self._preflight_complete or self._turn_active:
            raise AppServerTransportError(
                "turn cannot start before sealed preflight"
            )
        method, params = self._thread_params()
        thread_request_sha = sha256_bytes(
            canonical_json({"method": method, "params": params})
        )
        result = self._request(method, params)
        if (
            not isinstance(result, dict)
            or not isinstance(result.get("thread"), dict)
            or not isinstance(result["thread"].get("id"), str)
        ):
            raise AppServerTransportError(
                "thread start/resume result is malformed"
            )
        thread_id = result["thread"]["id"]
        if (
            self.binding.thread_mode == "resume"
            and thread_id != self.binding.resume_thread_id
        ):
            raise AppServerTransportError(
                "resumed app-server thread identity changed"
            )
        prompt = (
            "Solve exactly this receipt-bound ARC-AGI-3 frontier using only "
            "the contiguous_lane namespace. Immutable frontier:\n"
            + canonical_json(dict(frontier_brief)).decode("ascii")
        )
        prompt_sha = sha256_bytes(prompt.encode("utf-8"))
        turn_params = {
            "threadId": thread_id,
            "input": [
                {
                    "type": "text",
                    "text": prompt,
                    "text_elements": [],
                }
            ],
            "approvalPolicy": "never",
            "cwd": self.binding.neutral_cwd,
            "effort": self.binding.reasoning_effort,
            "environments": [],
            "model": self.binding.model,
            "runtimeWorkspaceRoots": [self.binding.neutral_cwd],
            "sandboxPolicy": {
                "type": "readOnly",
                "networkAccess": False,
            },
        }
        turn_request_sha = sha256_bytes(
            canonical_json(
                {"method": "turn/start", "params": turn_params}
            )
        )
        turn_result = self._request("turn/start", turn_params)
        if (
            not isinstance(turn_result, dict)
            or not isinstance(turn_result.get("turn"), dict)
            or not isinstance(turn_result["turn"].get("id"), str)
        ):
            raise AppServerTransportError(
                "turn/start result is malformed"
            )
        self._thread_id = thread_id
        self._turn_id = turn_result["turn"]["id"]
        self._turn_active = True
        assert self.transcript.head is not None
        return TurnStartEvidence(
            schema=1,
            thread_id=thread_id,
            turn_id=self._turn_id,
            thread_mode=self.binding.thread_mode,
            thread_request_sha256=thread_request_sha,
            turn_request_sha256=turn_request_sha,
            prompt_sha256=prompt_sha,
            transcript_chain_sha256=self.transcript.head,
        )

    def _materialize_probe_snapshot(
        self,
        *,
        request_id: str | int,
        call_id: str,
        entrypoint: str,
        files: list[str],
    ) -> tuple[WorkspaceSnapshotManifest, str]:
        assert self._thread_id is not None and self._turn_id is not None
        if (
            not is_safe_relative_path(entrypoint)
            or not isinstance(files, list)
            or entrypoint not in files
            or not 1 <= len(files) <= MAX_PROBE_FILES
            or len(files) != len(set(files))
            or not all(is_safe_relative_path(path) for path in files)
        ):
            raise AppServerTransportError(
                "probe declared paths are malformed"
            )
        binding = {
            "dynamic_request_id": request_id,
            "dynamic_call_id": call_id,
            "thread_id": self._thread_id,
            "turn_id": self._turn_id,
        }
        result = self.bridge.call(
            "probe_snapshot",
            {"paths": files, "binding": binding},
            idempotency_key="probe-snapshot:" + call_id,
        )
        if (
            not isinstance(result, dict)
            or result.get("binding") != binding
            or result.get("quiescent") is not True
            or result.get("no_writeback") is not True
            or not isinstance(result.get("entries"), list)
        ):
            raise AppServerTransportError(
                "bridge probe snapshot response is malformed"
            )
        generation = Path(self.binding.generation_dir)
        call_dir = (
            generation
            / "probe_calls"
            / _identifier_component(request_id)
            / _identifier_component(call_id)
        )
        snapshot = call_dir / "snapshot"
        snapshot.mkdir(parents=True, mode=0o700, exist_ok=False)
        inventory: list[dict[str, Any]] = []
        total = 0
        for row in result["entries"]:
            if (
                not isinstance(row, dict)
                or set(row) != {"path", "sha256", "bytes", "base64"}
                or not is_safe_relative_path(row["path"])
                or not SHA256_RE.fullmatch(str(row["sha256"]))
                or not isinstance(row["bytes"], int)
                or isinstance(row["bytes"], bool)
                or row["bytes"] < 0
                or not isinstance(row["base64"], str)
            ):
                raise AppServerTransportError(
                    "bridge probe snapshot entry is malformed"
                )
            try:
                import base64

                raw = base64.b64decode(
                    row["base64"], validate=True
                )
            except (ValueError, UnicodeError) as exc:
                raise AppServerTransportError(
                    "bridge snapshot base64 is invalid"
                ) from exc
            if (
                len(raw) != row["bytes"]
                or sha256_bytes(raw) != row["sha256"]
            ):
                raise AppServerTransportError(
                    "bridge snapshot content hash mismatch"
                )
            total += len(raw)
            if total > MAX_BRIDGE_LINE_BYTES:
                raise AppServerTransportError(
                    "bridge snapshot exceeds aggregate bound"
                )
            destination = snapshot.joinpath(
                *PurePosixPath(row["path"]).parts
            )
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o500)
            _write_new_bytes(destination, raw, mode=0o400)
            metadata = destination.stat(follow_symlinks=False)
            inventory.append(
                {
                    "path": row["path"],
                    "sha256": row["sha256"],
                    "bytes": row["bytes"],
                    "device": metadata.st_dev,
                    "inode": metadata.st_ino,
                }
            )
        inventory.sort(key=lambda row: row["path"])
        if [row["path"] for row in inventory] != sorted(files):
            raise AppServerTransportError(
                "bridge snapshot omitted or added declared paths"
            )
        for directory in sorted(
            (path for path in snapshot.rglob("*") if path.is_dir()),
            reverse=True,
        ):
            os.chmod(directory, 0o500, follow_symlinks=False)
        os.chmod(snapshot, 0o500, follow_symlinks=False)
        snapshot_stat = snapshot.stat(follow_symlinks=False)
        tree_sha = _regular_tree_sha256(snapshot)
        manifest_value = {
            "schema": 1,
            "kind": "arc_agi3_contiguous_workspace_snapshot",
            "campaign_id": self.binding.campaign_id,
            "generation_id": self.binding.generation_id,
            "attempt_id": self.binding.attempt_id,
            "dynamic_request_id": request_id,
            "dynamic_call_id": call_id,
            "thread_id": self._thread_id,
            "turn_id": self._turn_id,
            "generation_dir": str(generation),
            "call_dir": str(call_dir),
            "snapshot_root": str(snapshot),
            "snapshot_device": snapshot_stat.st_dev,
            "snapshot_inode": snapshot_stat.st_ino,
            "tree_sha256": tree_sha,
            "entries": inventory,
            "source_workspace_tree_sha256": str(
                result.get("inventory_sha256")
            ),
            "no_writeback": True,
        }
        manifest = workspace_snapshot_manifest_from_dict(manifest_value)
        manifest_path = call_dir / "snapshot_manifest.json"
        manifest_sha = _write_new_bytes(
            manifest_path,
            canonical_json(manifest_value) + b"\n",
            mode=0o400,
        )
        return manifest, manifest_sha

    def _execute_safe_probe(
        self,
        *,
        request_id: str | int,
        call_id: str,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        if (
            set(arguments)
            != {
                "entrypoint",
                "files",
                "arguments",
                "timeout_seconds",
            }
            or not isinstance(arguments["files"], list)
            or not isinstance(arguments["arguments"], list)
            or not all(
                isinstance(value, str)
                and "\x00" not in value
                and not value.startswith("/")
                and len(value.encode("utf-8"))
                <= MAX_PROBE_ARGUMENT_BYTES
                for value in arguments["arguments"]
            )
            or len(arguments["arguments"]) > MAX_PROBE_ARGUMENTS
            or not isinstance(arguments["timeout_seconds"], int)
            or isinstance(arguments["timeout_seconds"], bool)
            or not 1 <= arguments["timeout_seconds"]
            <= MAX_PROBE_TIMEOUT_SECONDS
        ):
            raise AppServerTransportError(
                "safe probe arguments are malformed"
            )
        manifest, manifest_sha = self._materialize_probe_snapshot(
            request_id=request_id,
            call_id=call_id,
            entrypoint=arguments["entrypoint"],
            files=arguments["files"],
        )
        request = ProbeExecutionRequest(
            schema=1,
            campaign_id=self.binding.campaign_id,
            generation_id=self.binding.generation_id,
            attempt_id=self.binding.attempt_id,
            dynamic_request_id=request_id,
            dynamic_call_id=call_id,
            thread_id=self._thread_id or "",
            turn_id=self._turn_id or "",
            workspace_snapshot_manifest_path=str(
                Path(manifest.call_dir) / "snapshot_manifest.json"
            ),
            workspace_snapshot_manifest_sha256=manifest_sha,
            workspace_snapshot_tree_sha256=manifest.tree_sha256,
            entrypoint=arguments["entrypoint"],
            arguments=tuple(arguments["arguments"]),
            timeout_seconds=arguments["timeout_seconds"],
            stdout_limit_bytes=1024 * 1024,
            stderr_limit_bytes=1024 * 1024,
            resource_limits=ProbeResourceLimits(
                cpus=1.0,
                memory_bytes=512 * 1024 * 1024,
                pids=32,
                tmpfs_bytes=64 * 1024 * 1024,
            ),
            arena_mode="disabled",
        )
        observed = self.probe_executor.run_probe(
            spec=self.probe_spec,
            launched=self.probe_launch,
            request=request,
        )
        if (
            not isinstance(observed, ProbeExecutionResult)
            or observed.request_sha256 != request.sha256()
            or observed.snapshot_tree_sha256 != manifest.tree_sha256
            or observed.no_writeback is not True
            or not all(
                (
                    observed.container_absent,
                    observed.process_group_absent,
                    observed.descendants_absent,
                )
            )
        ):
            raise AppServerTransportError(
                "probe executor result lacks containment proof"
            )
        stdout = _bounded_regular_bytes(
            Path(observed.stdout_path),
            max_bytes=request.stdout_limit_bytes + 65536,
            allow_empty=True,
        )
        stderr = _bounded_regular_bytes(
            Path(observed.stderr_path),
            max_bytes=request.stderr_limit_bytes + 65536,
            allow_empty=True,
        )
        if (
            sha256_bytes(stdout) != observed.stdout_sha256
            or len(stdout) != observed.stdout_bytes
            or sha256_bytes(stderr) != observed.stderr_sha256
            or len(stderr) != observed.stderr_bytes
        ):
            raise AppServerTransportError(
                "probe terminal streams changed or are incomplete"
            )
        visible_stderr, stderr_visibility = (
            _probe_stderr_visibility_projection(stderr)
        )
        _, stderr_visibility_receipt_sha256 = (
            _retain_probe_stderr_visibility_receipt(
            Path(observed.stderr_path), stderr_visibility
            )
        )
        return {
            "exit_code": observed.exit_code,
            "timed_out": observed.timed_out,
            "output_overflow": observed.output_overflow,
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stdout_sha256": observed.stdout_sha256,
            "stdout_bytes": observed.stdout_bytes,
            "stdout_truncated": observed.stdout_truncated,
            # Raw stderr remains immutable host evidence.  Model-visible
            # context receives only this fixed projection, never traceback
            # filenames, source lines, or exception prose.
            "stderr": visible_stderr,
            "stderr_sha256": observed.stderr_sha256,
            "stderr_bytes": observed.stderr_bytes,
            "stderr_truncated": observed.stderr_truncated,
            "stderr_visibility_receipt_sha256":
                stderr_visibility_receipt_sha256,
            "stderr_raw_surface_classification":
                stderr_visibility["raw_surface_classification"],
            "snapshot_tree_sha256": manifest.tree_sha256,
            "no_writeback": True,
            "teardown_receipt_sha256":
                observed.teardown_receipt_sha256,
        }

    def _handle_server_request(
        self, event: Mapping[str, Any]
    ) -> None:
        if not self._turn_active:
            raise AppServerTransportError(
                "server request occurred outside ACTIVE_TURN"
            )
        if set(event) != {"id", "method", "params"}:
            raise AppServerTransportError(
                "server request envelope is not exact"
            )
        request_id = event["id"]
        if (
            not isinstance(request_id, (str, int))
            or isinstance(request_id, bool)
            or request_id in self._server_requests
        ):
            raise AppServerTransportError(
                "server request is duplicated or forbidden"
            )
        self._server_requests.add(request_id)
        if event["method"] == "account/chatgptAuthTokens/refresh":
            self._handle_auth_refresh(event)
            return
        if event["method"] != "item/tool/call":
            raise AppServerTransportError(
                "server request method is forbidden"
            )
        params = event["params"]
        allowed_keys = {
            "arguments",
            "callId",
            "namespace",
            "threadId",
            "tool",
            "turnId",
        }
        if (
            not isinstance(params, dict)
            or set(params) != allowed_keys
            or params["namespace"] != "contiguous_lane"
            or params["threadId"] != self._thread_id
            or params["turnId"] != self._turn_id
            or not isinstance(params["callId"], str)
            or not isinstance(params["tool"], str)
            or params["tool"] not in DYNAMIC_TOOL_NAMES
            or not isinstance(params["arguments"], dict)
        ):
            raise AppServerTransportError(
                "dynamic tool call binding/schema mismatch"
            )
        try:
            if params["tool"] == "workspace_run_python":
                result = self._execute_safe_probe(
                    request_id=request_id,
                    call_id=params["callId"],
                    arguments=params["arguments"],
                )
            else:
                result = self.bridge.call(
                    params["tool"],
                    params["arguments"],
                    idempotency_key="dynamic:" + params["callId"],
                )
            response = {
                "id": request_id,
                "result": {
                    "contentItems": [
                        {
                            "type": "inputText",
                            "text": canonical_json(result).decode("ascii"),
                        }
                    ],
                    "success": True,
                },
            }
        except Exception as exc:
            response = {
                "id": request_id,
                "result": {
                    "contentItems": [
                        {
                            "type": "inputText",
                            "text": canonical_json(
                                {"error": type(exc).__name__}
                            ).decode("ascii"),
                        }
                    ],
                    "success": False,
                },
            }
        self._write_wire(response, direction="client_response")
        self._tool_call_count += 1

    def _handle_auth_refresh(
        self, event: Mapping[str, Any]
    ) -> None:
        if self._refresh_count >= self.binding.max_auth_refreshes:
            raise AppServerTransportError(
                "external auth refresh cardinality exceeded"
            )
        params = event.get("params")
        if (
            not isinstance(params, dict)
            or set(params)
            not in (
                {"reason"},
                {"reason", "previousAccountId"},
            )
            or params.get("reason") != "unauthorized"
            or params.get(
                "previousAccountId", self.credentials.account_id
            )
            not in (None, self.credentials.account_id)
        ):
            raise AppServerTransportError(
                "external auth refresh lineage/reason mismatch"
            )
        refreshed = load_external_chatgpt_credentials(
            Path(self.credentials.source_path)
        )
        if (
            refreshed.account_id != self.credentials.account_id
            or refreshed.source_path != self.credentials.source_path
        ):
            raise AppServerTransportError(
                "external auth refresh changed account lineage"
            )
        refreshed_token_sha256 = sha256_bytes(
            refreshed.access_token.encode("utf-8")
        )
        if (
            refreshed_token_sha256
            in self._credential_access_token_sha256
        ):
            raise AppServerTransportError(
                "external auth refresh did not rotate the access token"
            )
        self._credential_access_token_sha256.add(
            refreshed_token_sha256
        )
        self._credential_sentinels.update(
            refreshed.leak_sentinels
        )
        result: dict[str, Any] = {
            "accessToken": refreshed.access_token,
            "chatgptAccountId": refreshed.account_id,
        }
        redacted_result: dict[str, Any] = {
            "accessToken": "REDACTED",
            "chatgptAccountId": "REDACTED",
        }
        if refreshed.plan_type is not None:
            result["chatgptPlanType"] = refreshed.plan_type
            redacted_result["chatgptPlanType"] = refreshed.plan_type
        response = {"id": event["id"], "result": result}
        self._write_wire(
            response,
            direction="client_response",
            transcript_payload={
                "id": event["id"],
                "result": redacted_result,
            },
        )
        refresh_response_sha256 = sha256_bytes(
            canonical_json({
                "id": event["id"],
                "result": redacted_result,
            })
        )
        if (
            refresh_response_sha256
            in self._refresh_redacted_response_sha256
        ):
            raise AppServerTransportError(
                "external auth refresh response identity repeated"
            )
        self._refresh_redacted_response_sha256.append(
            refresh_response_sha256
        )
        self.credentials = refreshed
        self._credential_sentinel_scan()
        self._refresh_count += 1

    def credential_sentinels_for_host_scan(self) -> tuple[str, ...]:
        """Return live-only sentinels to the trusted host collection path."""

        return tuple(sorted(self._credential_sentinels))

    def _credential_sentinel_scan(self) -> Literal[True]:
        transcript_bytes = _bounded_regular_bytes(
            self.transcript.path,
            max_bytes=64 * 1024 * 1024,
            allow_empty=False,
        )
        retained = [
            transcript_bytes,
            bytes(self._stderr_complete),
        ]
        state_root = Path(self.binding.state_root)
        state_inventory = inventory_controller_state(
            state_root,
            sentinels=tuple(self._credential_sentinels),
        )
        if state_inventory.secret_occurrences or any(
            sentinel.encode("utf-8") in payload
            for sentinel in self._credential_sentinels
            for payload in retained
        ):
            raise AppServerTransportError(
                "credential entered retained controller state/evidence"
            )
        return True

    def _validate_token_usage_observations(self) -> None:
        previous: dict[str, float] = {}
        for observation in self._token_usage:
            if (
                observation.get("threadId") != self._thread_id
                or observation.get("turnId") != self._turn_id
            ):
                raise AppServerTransportError(
                    "token usage observation has stale turn identity"
                )

            def visit(prefix: str, value: object) -> None:
                if isinstance(value, dict):
                    for key, child in sorted(value.items()):
                        if key in {"threadId", "turnId"}:
                            continue
                        visit(f"{prefix}.{key}", child)
                elif (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                ):
                    numeric = float(value)
                    if numeric < 0 or numeric < previous.get(
                        prefix, numeric
                    ):
                        raise AppServerTransportError(
                            "token usage observations are nonmonotone"
                        )
                    previous[prefix] = numeric

            visit("usage", observation)

    def _drain_after_terminal(self) -> None:
        if (
            self.process is None
            or self.process.stdin is None
        ):
            raise AppServerTransportError(
                "terminal drain lacks the app-server process"
            )
        try:
            self.process.stdin.close()
        except OSError as exc:
            raise AppServerTransportError(
                "app-server stdin could not close for terminal drain"
            ) from exc
        self._allow_protocol_eof = True
        deadline = time.monotonic() + 30.0
        try:
            while True:
                try:
                    event = self._read_protocol_event(
                        max(0.001, deadline - time.monotonic())
                    )
                except _ProtocolEof:
                    break
                if (
                    "id" in event
                    or event.get("method")
                    not in {
                        "account/rateLimits/updated",
                        "account/updated",
                        "thread/status/changed",
                        "thread/tokenUsage/updated",
                    }
                ):
                    raise AppServerTransportError(
                        "unknown event followed terminal turn completion"
                    )
                self._post_turn_event_count += 1
        finally:
            self._allow_protocol_eof = False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AppServerTransportError(
                "app-server terminal pipe drain timed out"
            )
        try:
            exit_code = self.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            raise AppServerTransportError(
                "app-server did not exit after terminal EOF barrier"
            ) from exc
        if exit_code != 0:
            raise AppServerTransportError(
                "app-server terminal EOF barrier exited nonzero"
            )
        self._validate_token_usage_observations()

    def run_turn(self) -> TurnFinalEvidence:
        if (
            not self._turn_active
            or self._thread_id is None
            or self._turn_id is None
        ):
            raise AppServerTransportError("no active turn")
        deadline = (
            time.monotonic()
            + self.binding.hard_safety_seconds
        )
        turn_status: Literal["completed", "interrupted", "failed"] | None = None
        provider_outcome: Literal[
            "completed",
            "capacity",
            "rate_limit",
            "provider_failure",
            "containment_fault",
        ] = "completed"
        while turn_status is None:
            event = self._read_protocol_event(
                max(0.001, deadline - time.monotonic())
            )
            if "method" in event and "id" in event:
                self._handle_server_request(event)
                continue
            if "method" not in event:
                raise AppServerTransportError(
                    "unsolicited server response during active turn"
                )
            method = event["method"]
            params = event.get("params")
            if method == "turn/completed":
                if not isinstance(params, dict):
                    raise AppServerTransportError(
                        "turn/completed params are malformed"
                    )
                turn = params.get("turn")
                if (
                    not isinstance(turn, dict)
                    or turn.get("id") != self._turn_id
                    or turn.get("threadId", self._thread_id)
                    != self._thread_id
                ):
                    raise AppServerTransportError(
                        "terminal turn identity mismatch"
                    )
                status = turn.get("status")
                if status == "completed":
                    turn_status = "completed"
                elif status == "interrupted":
                    turn_status = "interrupted"
                else:
                    turn_status = "failed"
                    provider_outcome = "provider_failure"
        post_usage_window = self._request_provider_usage_window(
            "postflight"
        )
        self._drain_after_terminal()
        self._turn_active = False
        if self._preflight_provider_usage_window is None:
            raise AppServerTransportError(
                "turn lacks its authenticated preflight provider window"
            )
        usage_settlement = settle_provider_usage(
            self._preflight_provider_usage_window,
            post_usage_window,
            token_usage_observations=self._token_usage,
        )
        final_text = "".join(self._model_text_parts)
        assert self.transcript.head is not None
        return TurnFinalEvidence(
            schema=1,
            thread_id=self._thread_id,
            turn_id=self._turn_id,
            turn_status=turn_status,
            provider_outcome=provider_outcome,
            token_usage_observations=tuple(self._token_usage),
            pre_provider_usage_window=(
                self._preflight_provider_usage_window
            ),
            post_provider_usage_window=post_usage_window,
            provider_usage_settlement=usage_settlement,
            final_model_text_sha256=sha256_bytes(
                final_text.encode("utf-8")
            ),
            final_model_text=final_text,
            tool_call_count=self._tool_call_count,
            hard_safety_seconds=self.binding.hard_safety_seconds,
            max_auth_refreshes=self.binding.max_auth_refreshes,
            auth_refresh_count=self._refresh_count,
            redacted_auth_refresh_response_sha256=
                tuple(self._refresh_redacted_response_sha256),
            credential_sentinel_scan_passed=(
                self._credential_sentinel_scan()
            ),
            post_turn_event_count=self._post_turn_event_count,
            stdout_bytes=self._stdout_bytes_observed,
            stderr_bytes=self._stderr_bytes_observed,
            pipes_drained_to_eof=True,
            transcript_chain_sha256=self.transcript.head,
            transcript_event_count=self.transcript.sequence,
        )

    @staticmethod
    def _group_absent(process_group_id: int) -> bool:
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        return False

    def _drain_pipes_after_containment(self) -> None:
        assert self.process is not None
        for name, stream, buffer in (
            ("stdout", self.process.stdout, self._stdout_buffer),
            ("stderr", self.process.stderr, self._stderr_buffer),
        ):
            if stream is None:
                continue
            while True:
                try:
                    block = os.read(stream.fileno(), 65536)
                except BlockingIOError:
                    continue
                if not block:
                    break
                buffer.extend(block)
                if name == "stdout":
                    self._stdout_bytes_observed += len(block)
                else:
                    self._stderr_bytes_observed += len(block)
                if len(buffer) > MAX_BRIDGE_LINE_BYTES:
                    raise AppServerTransportError(
                        "post-containment pipe line exceeds its bound"
                    )
        while b"\n" in self._stderr_buffer:
            line, _, remainder = self._stderr_buffer.partition(b"\n")
            self._stderr_buffer[:] = remainder
            self._stderr_complete.extend(line + b"\n")
            self.transcript.append(
                direction="server_stderr",
                payload=line.decode("utf-8", errors="strict"),
            )
        while b"\n" in self._stdout_buffer:
            line, _, remainder = self._stdout_buffer.partition(b"\n")
            self._stdout_buffer[:] = remainder
            value = strict_json_loads(line)
            self.transcript.append(
                direction="server_post_containment",
                payload=value,
            )
        if self._stdout_buffer or self._stderr_buffer:
            raise AppServerTransportError(
                "app-server containment left a partial pipe record"
            )

    def contain(self) -> ControllerTeardownEvidence:
        if self.process is None or self._container_start is None:
            raise AppServerTransportError(
                "controller container was never started"
            )
        process = self.process
        if self._process_group_id is None:
            raise AppServerTransportError(
                "app-server process group identity is absent"
            )
        process_group_id = self._process_group_id
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        stopped = self.container_launcher.contain(
            binding=self.binding,
            started=self._container_start,
        )
        if (
            not isinstance(stopped, ControllerContainerStop)
            or stopped.controller_container_id
            != self._container_start.controller_container_id
            or stopped.egress_proxy_container_id
            != self._container_start.egress_proxy_container_id
            or any(
                value is not True
                for value in (
                    stopped.controller_inspect_absent,
                    stopped.controller_identity_query_empty,
                    stopped.controller_top_absent,
                    stopped.controller_no_descendants,
                    stopped.egress_proxy_inspect_absent,
                    stopped.egress_proxy_identity_query_empty,
                    stopped.egress_proxy_top_absent,
                    stopped.egress_proxy_no_descendants,
                )
            )
            or not SHA256_RE.fullmatch(
                stopped.absence_receipt_sha256
            )
        ):
            raise AppServerTransportError(
                "controller-container absence evidence is incomplete"
            )
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired as exc:
            # The controller cgroup is already authoritatively absent.  Do not
            # signal a host PID after a compare/use race; quarantine the
            # diagnostic Docker attach process for external guardian cleanup.
            raise AppServerTransportError(
                "Docker attach survived authoritative container removal"
            ) from exc
        exit_code = int(process.returncode)
        deadline = time.monotonic() + 5
        while not self._group_absent(process_group_id):
            if time.monotonic() >= deadline:
                raise AppServerTransportError(
                    "app-server process group survived containment"
                )
            time.sleep(0.01)
        self._drain_pipes_after_containment()
        _purge_ephemeral_directory(
            Path(self.binding.state_root) / "tmp"
        )
        self.bridge.close()
        self.transcript.close()
        head = self.transcript.head
        if head is None:
            raise AppServerTransportError(
                "app-server transcript is empty"
            )
        return ControllerTeardownEvidence(
            schema=1,
            pid=process.pid,
            process_group_id=process_group_id,
            exit_code=exit_code,
            process_absent=process.poll() is not None,
            process_group_absent=True,
            process_absent_receipt_sha256=(
                stopped.absence_receipt_sha256
            ),
            process_start_receipt_removed=True,
            ephemeral_tmp_purged=True,
            stderr_sha256=sha256_bytes(
                bytes(self._stderr_complete)
            ),
            stderr_bytes=len(self._stderr_complete),
            state_tree_sha256=_regular_tree_sha256(
                Path(self.binding.state_root)
            ),
            transcript_chain_sha256=head,
            process_identity_authority="controller_container_cgroup",
            controller_container_id=(
                stopped.controller_container_id
            ),
            egress_proxy_container_id=(
                stopped.egress_proxy_container_id
            ),
            controller_inspect_absent=True,
            controller_identity_query_empty=True,
            controller_top_absent=True,
            controller_no_descendants=True,
            egress_proxy_inspect_absent=True,
            egress_proxy_identity_query_empty=True,
            egress_proxy_top_absent=True,
            egress_proxy_no_descendants=True,
            controller_absence_receipt_sha256=(
                stopped.absence_receipt_sha256
            ),
        )


CONTIGUOUS_APP_SERVER_TRANSPORT_LAUNCH_READY = False
