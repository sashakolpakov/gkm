#!/usr/bin/env python3
"""Strict taint admission for contiguous-campaign evidence.

The general release auditor intentionally treats a few historical
introspection patterns as informational.  A live contiguous proposer has a
stronger information boundary: harness introspection, unknown transcript
events, host execution methods, malformed JSONL, and unreadable evidence are
all actionable failures.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Literal, Mapping, Sequence

from arc import audit_submission_taint as GeneralTaint
try:
    from arc.crack_lab import (
        arc_agi3_codex_app_server_transport as Transport,
    )
    from arc.crack_lab import arc_agi3_proposer_boundary as Boundary
except ModuleNotFoundError:  # pragma: no cover - direct-script fallback
    import arc_agi3_codex_app_server_transport as Transport
    import arc_agi3_proposer_boundary as Boundary


SCHEMA = 1
MAX_SCAN_BYTES = 64 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PINNED_APP_SERVER_VERSION = "0.145.0"
PINNED_MODEL = "gpt-5.6-sol"
PINNED_MODEL_PROVIDER = "openai"
APP_SERVER_HARD_SAFETY_SECONDS = (
    Transport.APP_SERVER_HARD_SAFETY_SECONDS
)
MAX_AUTH_REFRESHES = Transport.MAX_AUTH_REFRESHES
SUPPORTED_CAMPAIGN_EFFORTS = frozenset(
    {"medium", "high", "xhigh", "max"}
)
MAX_EMITTED_AT_MS = (1 << 63) - 1
JWT_LIKE_RE = re.compile(
    r"(?:^|[^A-Za-z0-9_-])"
    r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"
    r"(?:\.[A-Za-z0-9_-]{8,})?"
    r"(?:$|[^A-Za-z0-9_-])"
)
BEARER_SECRET_RE = re.compile(
    r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}",
    re.IGNORECASE,
)

# App-server notifications observed by the pinned protocol may carry model
# text, reasoning summaries, usage, or lifecycle state. Everything else must
# be reviewed and added explicitly rather than silently ignored. MCP startup
# is intentionally absent: the contiguous controller configures zero MCP
# servers, so even a protocol-valid MCP notification is a containment fault.
PREFLIGHT_SERVER_NOTIFICATION_METHODS = frozenset(
    {
        "account/login/completed",
        "account/rateLimits/updated",
        "account/updated",
    }
)
TURN_SERVER_NOTIFICATION_METHODS = frozenset(
    {
        "account/rateLimits/updated",
        "account/updated",
        "error",
        "item/agentMessage/delta",
        "item/completed",
        "item/reasoning/summaryPartAdded",
        "item/reasoning/summaryTextDelta",
        "item/reasoning/textDelta",
        "item/started",
        "thread/started",
        "thread/status/changed",
        "thread/tokenUsage/updated",
        "turn/completed",
        "turn/started",
    }
)

FORBIDDEN_APP_SERVER_METHOD_FRAGMENTS = (
    "command/exec",
    "config/",
    "externalAgentConfig/",
    "fuzzyFileSearch/",
    "memory/",
    "process/",
    "thread/backgroundTerminals/",
    "thread/shellCommand",
)
FORBIDDEN_EXACT_METHODS = frozenset(
    {
        "applyPatchApproval",
        "commandExecutionRequestApproval",
        "execCommandApproval",
        "fileChangeRequestApproval",
        "mcpServer/oauth/login",
        "permissions/requestApproval",
        "skills/config/write",
        "thread/approveGuardianDeniedAction",
        "tool/requestUserInput",
    }
)
FORBIDDEN_ITEM_TYPES = frozenset(
    {
        "commandExecution",
        "command_execution",
        "fileChange",
        "file_change",
        "imageGeneration",
        "process",
        "shell",
        "webSearch",
        "web_search",
    }
)
ALLOWED_TURN_ITEM_TYPES = frozenset(
    {
        "agentMessage",
        "dynamicToolCall",
        "reasoning",
        "userMessage",
    }
)
MODEL_TEXT_NOTIFICATION_METHODS = frozenset(
    {
        "item/agentMessage/delta",
        "item/completed",
        "item/reasoning/summaryPartAdded",
        "item/reasoning/summaryTextDelta",
        "item/reasoning/textDelta",
        "item/started",
    }
)
DELTA_NOTIFICATION_METHODS = frozenset(
    {
        "item/agentMessage/delta",
        "item/reasoning/summaryPartAdded",
        "item/reasoning/summaryTextDelta",
        "item/reasoning/textDelta",
    }
)
POST_TURN_NOTIFICATION_METHODS = frozenset(
    {
        "account/rateLimits/updated",
        "thread/status/changed",
        "thread/tokenUsage/updated",
    }
)
ALLOWED_DYNAMIC_TOOLS = frozenset(Transport.DYNAMIC_TOOL_NAMES)
SECRET_NAME_RE = re.compile(
    r"(?:API[_-]?KEY|ACCESS[_-]?TOKEN|AUTH(?:ORIZATION)?|"
    r"CHATGPT|CODEX_(?:TOKEN|AUTH)|OPENAI_API_KEY)",
    re.IGNORECASE,
)
CREDENTIAL_FIELD_RE = re.compile(
    r"^(?:"
    r"OPENAI_API_KEY|"
    r"access[_-]?token|"
    r"refresh[_-]?token|"
    r"id[_-]?token|"
    r"authorization|"
    r"chatgptAccountId|"
    r"account_id|"
    r"api[_-]?key"
    r")$",
    re.IGNORECASE,
)


class ContiguousTaintError(RuntimeError):
    """Evidence could not be scanned under the strict contract."""


CONTROLLER_CANARY_CATEGORIES = (
    "repository",
    "home",
    "environment",
    "auth_source",
    "controller_control_root",
    "sibling_lane",
)
CANARY_LOCATION_RE = re.compile(r"^[^\x00\r\n]{1,1024}$")


@dataclass(frozen=True)
class LiveCanary:
    """One independently generated containment value.

    ``value`` is live-only and must never be serialized.  Receipts retain the
    category, the public location/name where the value was planted, and a
    domain-separated commitment.
    """

    category: str
    location_name: str
    value: str
    provenance: Literal["secrets.token_hex_32"] = "secrets.token_hex_32"

    def commitment(self) -> dict[str, str]:
        return {
            "category": self.category,
            "location_name": self.location_name,
            "provenance": self.provenance,
            "commitment_sha256": hashlib.sha256(
                b"arc-agi3-controller-canary-v1\0"
                + self.category.encode("utf-8")
                + b"\0"
                + self.location_name.encode("utf-8")
                + b"\0"
                + self.provenance.encode("ascii")
                + b"\0"
                + self.value.encode("utf-8")
            ).hexdigest(),
        }


def _canary_commitment_rows(
    canaries: Sequence[LiveCanary],
) -> tuple[tuple[str, str, str, str], ...]:
    return tuple(
        (
            item.category,
            item.location_name,
            item.provenance,
            item.commitment()["commitment_sha256"],
        )
        for item in validate_live_canaries(
            tuple(canaries),
            require_complete=True,
        )
    )


def validate_live_canaries(
    canaries: Sequence[LiveCanary],
    *,
    require_complete: bool = True,
) -> tuple[LiveCanary, ...]:
    if not isinstance(canaries, tuple):
        raise ContiguousTaintError(
            "controller containment canaries must be an exact tuple"
        )
    normalized = tuple(canaries)
    if (
        any(not isinstance(item, LiveCanary) for item in normalized)
        or any(
            item.category not in CONTROLLER_CANARY_CATEGORIES
            or CANARY_LOCATION_RE.fullmatch(item.location_name) is None
            or not isinstance(item.value, str)
            or re.fullmatch(r"[0-9a-f]{64}", item.value) is None
            or item.provenance != "secrets.token_hex_32"
            for item in normalized
        )
        or len({item.category for item in normalized}) != len(normalized)
        or len({item.location_name for item in normalized}) != len(normalized)
        or len({item.value for item in normalized}) != len(normalized)
        or (
            require_complete
            and tuple(sorted(item.category for item in normalized))
            != tuple(sorted(CONTROLLER_CANARY_CATEGORIES))
        )
    ):
        raise ContiguousTaintError(
            "controller containment canary set is incomplete or malformed"
        )
    return tuple(sorted(normalized, key=lambda item: item.category))


def build_live_canary_reveal(
    canaries: tuple[LiveCanary, ...],
) -> dict[str, Any]:
    """Create the post-containment reveal used by an independent verifier.

    The returned object deliberately contains the noncredential marker values.
    It therefore MUST be written only after every controller/proposer process
    has stopped and only outside all retained attempt-evidence roots.  No
    attempt receipt may embed this object.
    """

    normalized = validate_live_canaries(canaries)
    return {
        "schema": 1,
        "kind": "contiguous_containment_canary_reveal",
        "canaries": [
            {
                **item.commitment(),
                "value": item.value,
            }
            for item in normalized
        ],
    }


def validate_live_canary_reveal(
    reveal: Mapping[str, Any],
    *,
    expected_commitments: Sequence[
        tuple[str, str, str, str]
    ],
) -> tuple[LiveCanary, ...]:
    """Validate a reveal against commitments from sealed attempt evidence."""

    if (
        not isinstance(reveal, Mapping)
        or set(reveal) != {"schema", "kind", "canaries"}
        or reveal.get("schema") != 1
        or reveal.get("kind")
        != "contiguous_containment_canary_reveal"
        or not isinstance(reveal.get("canaries"), list)
    ):
        raise ContiguousTaintError(
            "containment canary reveal is malformed"
        )
    reconstructed: list[LiveCanary] = []
    supplied_commitments: list[tuple[str, str, str, str]] = []
    for row in reveal["canaries"]:
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "category",
                "location_name",
                "provenance",
                "commitment_sha256",
                "value",
            }
            or SHA256_RE.fullmatch(
                str(row.get("commitment_sha256", ""))
            )
            is None
        ):
            raise ContiguousTaintError(
                "containment canary reveal row is malformed"
            )
        item = LiveCanary(
            category=row["category"],
            location_name=row["location_name"],
            provenance=row["provenance"],
            value=row["value"],
        )
        reconstructed.append(item)
        commitment = item.commitment()["commitment_sha256"]
        if not hmac.compare_digest(
            commitment,
            row["commitment_sha256"],
        ):
            raise ContiguousTaintError(
                "containment canary reveal commitment mismatch"
            )
        supplied_commitments.append(
            (
                item.category,
                item.location_name,
                item.provenance,
                commitment,
            )
        )
    normalized = validate_live_canaries(tuple(reconstructed))
    expected = tuple(expected_commitments)
    if (
        any(
            not isinstance(row, tuple)
            or len(row) != 4
            or any(not isinstance(value, str) for value in row)
            for row in expected
        )
        or tuple(sorted(supplied_commitments))
        != tuple(sorted(expected))
        or _canary_commitment_rows(normalized)
        != tuple(sorted(expected))
    ):
        raise ContiguousTaintError(
            "containment canary reveal does not match sealed commitments"
        )
    return normalized


@dataclass(frozen=True)
class ScanRecord:
    path: str
    sha256: str
    size: int
    evidence_kind: str
    hits: tuple[str, ...]


@dataclass(frozen=True)
class ControllerStateScan:
    """Hash-bound, complete scan of one quiescent controller state tree."""

    tree_sha256: str
    inventory_sha256: str
    file_count: int
    total_bytes: int
    records: tuple[ScanRecord, ...]
    hits: tuple[str, ...]
    canary_count: int
    canary_commitments: tuple[tuple[str, str, str, str], ...]
    canary_occurrences: int
    status: Literal["CLEAN", "TAINT"]

    def as_receipt(self) -> dict[str, Any]:
        return {
            "tree_sha256": self.tree_sha256,
            "inventory_sha256": self.inventory_sha256,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "records": [
                {
                    "path": record.path,
                    "sha256": record.sha256,
                    "size": record.size,
                    "evidence_kind": record.evidence_kind,
                    "hits": list(record.hits),
                }
                for record in self.records
            ],
            "hits": list(self.hits),
            "canary_count": self.canary_count,
            "canary_commitments": [
                {
                    "category": category,
                    "location_name": location_name,
                    "provenance": provenance,
                    "commitment_sha256": commitment,
                }
                for category, location_name, provenance, commitment
                in self.canary_commitments
            ],
            "canary_occurrences": self.canary_occurrences,
            "status": self.status,
        }


@dataclass(frozen=True)
class RetainedCanaryScan:
    """Complete canary-only scan of retained host/output evidence roots."""

    root_inventories: tuple[
        tuple[str, str, str, int, int], ...
    ]
    records: tuple[ScanRecord, ...]
    hits: tuple[str, ...]
    canary_commitments: tuple[tuple[str, str, str, str], ...]
    canary_occurrences: int
    status: Literal["CLEAN", "TAINT"]

    def as_receipt(self) -> dict[str, Any]:
        return {
            "root_inventories": [
                {
                    "label": label,
                    "tree_sha256": tree_sha256,
                    "inventory_sha256": inventory_sha256,
                    "file_count": file_count,
                    "total_bytes": total_bytes,
                }
                for (
                    label,
                    tree_sha256,
                    inventory_sha256,
                    file_count,
                    total_bytes,
                ) in self.root_inventories
            ],
            "records": [
                {
                    "path": record.path,
                    "sha256": record.sha256,
                    "size": record.size,
                    "evidence_kind": record.evidence_kind,
                    "hits": list(record.hits),
                }
                for record in self.records
            ],
            "hits": list(self.hits),
            "canary_commitments": [
                {
                    "category": category,
                    "location_name": location_name,
                    "provenance": provenance,
                    "commitment_sha256": commitment,
                }
                for category, location_name, provenance, commitment
                in self.canary_commitments
            ],
            "canary_occurrences": self.canary_occurrences,
            "status": self.status,
        }


STATE_ENV_ASSIGNMENT_RE = re.compile(
    r"(?:^|[\x00\r\n])(?:"
    r"HOME|USER|LOGNAME|SHELL|PWD|OLDPWD|PYTHONPATH|"
    r"CODEX_HOME|XDG_CONFIG_HOME|XDG_STATE_HOME|"
    r"OPENAI_API_KEY|ANTHROPIC_API_KEY"
    r")=",
    re.IGNORECASE,
)


def _plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class AppServerScanPolicy:
    """Attempt-bound facts needed to independently audit one raw turn.

    Secret sentinels are live-only inputs.  They must never be serialized into
    a receipt or retained beside the transcript.
    """

    state_root: str
    neutral_cwd: str
    model: str
    model_provider: str
    reasoning_effort: str
    thread_mode: Literal["new", "resume"]
    resume_thread_id: str | None
    prompt_sha256: str
    hard_safety_seconds: int
    max_auth_refreshes: int
    secret_sentinels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.state_root, str)
            or not Path(self.state_root).is_absolute()
            or not isinstance(self.neutral_cwd, str)
            or not Path(self.neutral_cwd).is_absolute()
            or self.model != PINNED_MODEL
            or self.model_provider != PINNED_MODEL_PROVIDER
            or self.reasoning_effort not in SUPPORTED_CAMPAIGN_EFFORTS
            or self.thread_mode not in {"new", "resume"}
            or (
                self.thread_mode == "new"
                and self.resume_thread_id is not None
            )
            or (
                self.thread_mode == "resume"
                and (
                    not isinstance(self.resume_thread_id, str)
                    or not self.resume_thread_id
                )
            )
            or not SHA256_RE.fullmatch(self.prompt_sha256)
            or not _plain_int(self.hard_safety_seconds)
            or self.hard_safety_seconds
            != APP_SERVER_HARD_SAFETY_SECONDS
            or not _plain_int(self.max_auth_refreshes)
            or self.max_auth_refreshes != MAX_AUTH_REFRESHES
            or not isinstance(self.secret_sentinels, tuple)
            or any(
                not isinstance(item, str)
                or not item
                or item == "REDACTED"
                for item in self.secret_sentinels
            )
        ):
            raise ContiguousTaintError(
                "app-server scan policy is malformed"
            )


def source_sha256() -> str:
    digest = hashlib.sha256(b"arc-agi3-contiguous-taint-controls-v2\0")
    for path in (Path(__file__), Path(Boundary.__file__)):
        raw = path.read_bytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _reject_duplicate_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContiguousTaintError(
                f"duplicate JSON object key: {key}"
            )
        result[key] = value
    return result


def _strict_json_loads(raw: str) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ContiguousTaintError(
                    f"non-finite JSON number: {value}"
                )
            ),
        )
    except ContiguousTaintError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ContiguousTaintError("malformed JSON") from exc


def _request_id(value: object) -> bool:
    return bool(
        (isinstance(value, str) and value)
        or (
            isinstance(value, int)
            and not isinstance(value, bool)
        )
    )


def _exact_mapping(
    value: object,
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> Mapping[str, Any]:
    if (
        not isinstance(value, dict)
        or not required.issubset(value)
        or not set(value).issubset(required | optional)
    ):
        raise ContiguousTaintError("app-server payload schema mismatch")
    return value


def _nested_identifier(
    value: object,
    *path: str,
) -> str | None:
    current = value
    for name in path:
        if not isinstance(current, dict):
            return None
        current = current.get(name)
    return current if isinstance(current, str) and current else None


def _scan_model_value(value: object) -> list[str]:
    hits: list[str] = []
    hits.extend(_item_hits(value))
    for text in _strings(value):
        hits.extend(_scan_text_full(text, execution_surface=True))
    return hits


def _validate_redacted_login(params: object) -> None:
    value = _exact_mapping(
        params,
        required=frozenset(
            {
                "type",
                "accessToken",
                "chatgptAccountId",
            }
        ),
        optional=frozenset({"chatgptPlanType"}),
    )
    if (
        value["type"] != "chatgptAuthTokens"
        or value["accessToken"] != "REDACTED"
        or value["chatgptAccountId"] != "REDACTED"
        or (
            "chatgptPlanType" in value
            and not isinstance(value["chatgptPlanType"], str)
        )
    ):
        raise ContiguousTaintError(
            "app-server login transcript is not exactly redacted"
        )


def _credential_value_hits(value: object) -> list[str]:
    """Find credential-shaped material independent of general taint rules."""

    hits: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if (
                isinstance(key, str)
                and CREDENTIAL_FIELD_RE.fullmatch(key)
                and child is not None
                and child != "REDACTED"
            ):
                hits.append("credential_value_exposure")
            hits.extend(_credential_value_hits(child))
    elif isinstance(value, list):
        for child in value:
            hits.extend(_credential_value_hits(child))
    elif isinstance(value, str):
        if JWT_LIKE_RE.search(value) or BEARER_SECRET_RE.search(value):
            hits.append("credential_value_exposure")
    return hits


def _expected_preflight_params(
    method: str,
    policy: AppServerScanPolicy,
) -> Mapping[str, Any] | None:
    """Return the exact retained request document for one pinned preflight."""

    expected: dict[str, Mapping[str, Any]] = {
        "initialize": Transport.INITIALIZE_PARAMS,
        "account/rateLimits/read": {},
        "account/read": {"refreshToken": False},
        "model/list": {
            "cursor": None,
            "includeHidden": True,
            "limit": 100,
        },
        "modelProvider/capabilities/read": {},
        "config/read": {
            "cwd": policy.neutral_cwd,
            "includeLayers": True,
        },
        "skills/list": {
            "cwds": [policy.neutral_cwd],
            "forceReload": True,
        },
        "hooks/list": {"cwds": [policy.neutral_cwd]},
        "plugin/list": {
            "cwds": [policy.neutral_cwd],
            "marketplaceKinds": [
                "local",
                "vertical",
                "workspace-directory",
            ],
        },
        "app/list": {
            "cursor": None,
            "forceRefetch": False,
            "limit": 100,
            "threadId": None,
        },
        "experimentalFeature/list": {
            "cursor": None,
            "limit": 100,
            "threadId": None,
        },
        "mcpServerStatus/list": {
            "cursor": None,
            "detail": "full",
            "limit": 100,
            "threadId": None,
        },
    }
    return expected.get(method)


def _list_result(
    result: object,
    *,
    paginated: bool,
) -> list[Any]:
    value = _exact_mapping(
        result,
        required=frozenset({"data"}),
        optional=(
            frozenset({"nextCursor"})
            if paginated
            else frozenset()
        ),
    )
    if (
        not isinstance(value["data"], list)
        or (
            paginated
            and value.get("nextCursor") is not None
        )
    ):
        raise ContiguousTaintError(
            "app-server inventory is malformed or incompletely paginated"
        )
    return value["data"]


def _validate_model_inventory(
    result: object,
    policy: AppServerScanPolicy,
) -> None:
    models = _list_result(result, paginated=True)
    matches = [
        row
        for row in models
        if isinstance(row, dict)
        and row.get("id") == policy.model
        and row.get("model") == policy.model
    ]
    if len(matches) != 1:
        raise ContiguousTaintError(
            "model/list lacks exactly one pinned model"
        )
    supported = matches[0].get("supportedReasoningEfforts")
    if not isinstance(supported, list):
        raise ContiguousTaintError(
            "pinned model lacks a reasoning-effort inventory"
        )
    efforts: set[str] = set()
    for row in supported:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("reasoningEffort"), str)
            or not isinstance(row.get("description"), str)
        ):
            raise ContiguousTaintError(
                "model reasoning-effort inventory is malformed"
            )
        efforts.add(row["reasoningEffort"])
    if not SUPPORTED_CAMPAIGN_EFFORTS.issubset(efforts):
        raise ContiguousTaintError(
            "pinned model lacks a campaign escalation effort"
        )


AUTHORITY_FEATURE_DENYLIST = frozenset(
    {
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
        "enable_fanout",
        "enable_mcp_apps",
        "executor_capability_discovery",
        "external_agent_memory_import",
        "goals",
        "guardian_approval",
        "hooks",
        "image_generation",
        "in_app_browser",
        "memories",
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
    }
)


def _validate_disabled_apps(value: object) -> bool:
    if value is None:
        return True
    if not isinstance(value, dict) or set(value) - {"_default"}:
        return False
    default = value.get("_default")
    if default is None:
        return True
    return bool(
        isinstance(default, dict)
        and default.get("enabled") is False
        and default.get("destructive_enabled") is False
        and default.get("open_world_enabled") is False
    )


def _validate_config_result(
    result: object,
    policy: AppServerScanPolicy,
) -> None:
    value = _exact_mapping(
        result,
        required=frozenset({"config", "origins", "layers"}),
    )
    config = value["config"]
    origins = value["origins"]
    layers = value["layers"]
    critical = {
        "model": policy.model,
        "model_provider": policy.model_provider,
        "model_reasoning_effort": policy.reasoning_effort,
        "approval_policy": "never",
        "sandbox_mode": "read-only",
        "web_search": "disabled",
    }
    if (
        not isinstance(config, dict)
        or any(config.get(name) != expected for name, expected in critical.items())
        or not _validate_disabled_apps(config.get("apps"))
        or config.get("tools") is not None
        or config.get("mcp_servers") not in (None, {})
        or config.get("plugins") not in (None, {})
        or config.get("marketplaces") not in (None, {})
        or not isinstance(config.get("features"), dict)
        or any(
            config["features"].get(name) is not False
            for name in AUTHORITY_FEATURE_DENYLIST
        )
        or not isinstance(origins, dict)
        or not set(critical).issubset(origins)
        or not {
            f"features.{name}" for name in AUTHORITY_FEATURE_DENYLIST
        }.issubset(origins)
        or not isinstance(layers, list)
        or not layers
    ):
        raise ContiguousTaintError(
            "config/read does not prove the frozen zero-authority projection"
        )
    expected_config = str(Path(policy.state_root) / "config.toml")
    admitted_layer = 0
    for layer in layers:
        if not isinstance(layer, dict):
            raise ContiguousTaintError(
                "config/read contains a malformed layer"
            )
        name = layer.get("name")
        file_name = (
            name.get("file") if isinstance(name, dict) else None
        )
        layer_config = layer.get("config")
        if file_name == expected_config:
            admitted_layer += 1
        elif layer_config not in ({}, None):
            raise ContiguousTaintError(
                "config/read discovered an unadmitted nonempty layer"
            )
    if admitted_layer != 1:
        raise ContiguousTaintError(
            "config/read did not bind exactly one lane config layer"
        )


def _validate_skills_result(
    result: object,
    policy: AppServerScanPolicy,
) -> None:
    entries = _list_result(result, paginated=False)
    if len(entries) != 1 or not isinstance(entries[0], dict):
        raise ContiguousTaintError(
            "skills/list did not return exactly one cwd inventory"
        )
    entry = entries[0]
    if (
        entry.get("cwd") != policy.neutral_cwd
        or entry.get("errors") != []
        or not isinstance(entry.get("skills"), list)
    ):
        raise ContiguousTaintError(
            "skills/list cwd or errors differ from the frozen inventory"
        )
    names: set[str] = set()
    state_skills = Path(policy.state_root) / "skills" / ".system"
    for skill in entry["skills"]:
        if (
            not isinstance(skill, dict)
            or not isinstance(skill.get("name"), str)
            or skill["name"] in names
            or skill.get("enabled") is not False
            or skill.get("scope") != "system"
            or not isinstance(skill.get("path"), str)
        ):
            raise ContiguousTaintError(
                "skills/list contains an unknown or enabled skill"
            )
        try:
            Path(skill["path"]).relative_to(state_skills)
        except ValueError as exc:
            raise ContiguousTaintError(
                "system skill path escaped the isolated state root"
            ) from exc
        names.add(skill["name"])
    if names != set(Transport.DISABLED_SYSTEM_SKILLS):
        raise ContiguousTaintError(
            "skills/list differs from the exact disabled inventory"
        )


def _validate_hooks_result(
    result: object,
    policy: AppServerScanPolicy,
) -> None:
    entries = _list_result(result, paginated=False)
    if (
        len(entries) != 1
        or not isinstance(entries[0], dict)
        or entries[0].get("cwd") != policy.neutral_cwd
        or entries[0].get("errors") != []
        or entries[0].get("hooks") != []
        or entries[0].get("warnings") != []
    ):
        raise ContiguousTaintError(
            "hooks/list returned an error or model-visible hook"
        )


def _validate_preflight_result(
    method: str,
    result: object,
    policy: AppServerScanPolicy,
) -> None:
    if method == "initialize":
        value = _exact_mapping(
            result,
            required=frozenset(
                {
                    "codexHome",
                    "platformFamily",
                    "platformOs",
                    "userAgent",
                }
            ),
        )
        if (
            value["codexHome"] != policy.state_root
            or value["platformFamily"] != "unix"
            or not isinstance(value["platformOs"], str)
            or not value["platformOs"]
            or not isinstance(value["userAgent"], str)
            or (
                f"gkm-arc-agi3-contiguous/{PINNED_APP_SERVER_VERSION}"
                not in value["userAgent"]
            )
        ):
            raise ContiguousTaintError(
                "initialize result differs from the attempt-bound runtime"
            )
    elif method == "account/login/start":
        if result != {"type": "chatgptAuthTokens"}:
            raise ContiguousTaintError(
                "login result is not external ChatGPT-token auth"
            )
    elif method == "account/read":
        value = _exact_mapping(
            result,
            required=frozenset({"account", "requiresOpenaiAuth"}),
        )
        account = _exact_mapping(
            value["account"],
            required=frozenset({"email", "planType", "type"}),
        )
        if (
            value["requiresOpenaiAuth"] is not True
            or account["type"] != "chatgpt"
            or not (
                account["email"] is None
                or isinstance(account["email"], str)
            )
            or not isinstance(account["planType"], str)
            or not account["planType"]
        ):
            raise ContiguousTaintError(
                "account/read did not prove external ChatGPT auth"
            )
    elif method == "model/list":
        _validate_model_inventory(result, policy)
    elif method == "modelProvider/capabilities/read":
        value = _exact_mapping(
            result,
            required=frozenset(
                {"imageGeneration", "namespaceTools", "webSearch"}
            ),
        )
        if (
            value["namespaceTools"] is not True
            or not isinstance(value["imageGeneration"], bool)
            or not isinstance(value["webSearch"], bool)
        ):
            raise ContiguousTaintError(
                "provider capability inventory is malformed"
            )
    elif method == "config/read":
        _validate_config_result(result, policy)
    elif method == "skills/list":
        _validate_skills_result(result, policy)
    elif method == "hooks/list":
        _validate_hooks_result(result, policy)
    elif method == "plugin/list":
        value = _exact_mapping(
            result,
            required=frozenset({"marketplaces"}),
            optional=frozenset(
                {"featuredPluginIds", "marketplaceLoadErrors"}
            ),
        )
        if (
            value["marketplaces"] != []
            or value.get("featuredPluginIds", []) != []
            or value.get("marketplaceLoadErrors", []) != []
        ):
            raise ContiguousTaintError(
                "plugin/list returned model-visible authority"
            )
    elif method == "app/list":
        if _list_result(result, paginated=True):
            raise ContiguousTaintError(
                "app/list returned model-visible authority"
            )
    elif method == "experimentalFeature/list":
        rows = _list_result(result, paginated=True)
        by_name: dict[str, bool] = {}
        for row in rows:
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("name"), str)
                or row["name"] in by_name
                or not isinstance(row.get("enabled"), bool)
            ):
                raise ContiguousTaintError(
                    "experimental feature inventory is malformed"
                )
            by_name[row["name"]] = row["enabled"]
        if (
            not AUTHORITY_FEATURE_DENYLIST.issubset(by_name)
            or any(by_name[name] for name in AUTHORITY_FEATURE_DENYLIST)
        ):
            raise ContiguousTaintError(
                "an authority-bearing experimental feature is enabled"
            )
    elif method == "mcpServerStatus/list":
        if _list_result(result, paginated=True):
            raise ContiguousTaintError(
                "mcpServerStatus/list returned model-visible authority"
            )


def _validate_thread_request(
    method: str,
    params: object,
    policy: AppServerScanPolicy,
) -> None:
    common = {
        "approvalPolicy": "never",
        "baseInstructions": Transport.BASE_INSTRUCTIONS,
        "cwd": policy.neutral_cwd,
        "developerInstructions": Transport.DEVELOPER_INSTRUCTIONS,
        "model": policy.model,
        "modelProvider": policy.model_provider,
        "runtimeWorkspaceRoots": [policy.neutral_cwd],
        "sandbox": "read-only",
    }
    if method == "thread/start":
        expected = {
            **common,
            "allowProviderModelFallback": False,
            "dynamicTools": list(Transport.DYNAMIC_TOOL_SPECS),
            "environments": [],
            "ephemeral": False,
            "experimentalRawEvents": False,
            "historyMode": "paginated",
            "selectedCapabilityRoots": [],
        }
    else:
        expected = {
            **common,
            "threadId": policy.resume_thread_id,
            "excludeTurns": False,
        }
    if (
        method != f"thread/{'start' if policy.thread_mode == 'new' else 'resume'}"
        or params != expected
    ):
        raise ContiguousTaintError(
            "thread request differs from the frozen security document"
        )


def _validate_turn_request(
    params: object,
    policy: AppServerScanPolicy,
    thread_id: str,
) -> None:
    if not isinstance(params, dict):
        raise ContiguousTaintError(
            "turn request parameters are not an object"
        )
    expected_keys = {
        "threadId",
        "input",
        "approvalPolicy",
        "cwd",
        "effort",
        "environments",
        "model",
        "runtimeWorkspaceRoots",
        "sandboxPolicy",
    }
    inputs = params.get("input")
    if (
        set(params) != expected_keys
        or params.get("threadId") != thread_id
        or params.get("approvalPolicy") != "never"
        or params.get("cwd") != policy.neutral_cwd
        or params.get("effort") != policy.reasoning_effort
        or params.get("environments") != []
        or params.get("model") != policy.model
        or params.get("runtimeWorkspaceRoots")
        != [policy.neutral_cwd]
        or params.get("sandboxPolicy")
        != {"type": "readOnly", "networkAccess": False}
        or not isinstance(inputs, list)
        or len(inputs) != 1
        or not isinstance(inputs[0], dict)
        or set(inputs[0]) != {"type", "text", "text_elements"}
        or inputs[0].get("type") != "text"
        or inputs[0].get("text_elements") != []
        or not isinstance(inputs[0].get("text"), str)
        or hashlib.sha256(
            inputs[0]["text"].encode("utf-8")
        ).hexdigest()
        != policy.prompt_sha256
    ):
        raise ContiguousTaintError(
            "turn request differs from the attempt-bound security document"
        )


class _AppServerLifecycle:
    """Exact single-controller lifecycle validator for one retained turn."""

    def __init__(self, policy: AppServerScanPolicy) -> None:
        self.policy = policy
        self.preflight_index = 0
        self.pending_client: dict[str | int, str] = {}
        self.pending_server: dict[str | int, tuple[str, str]] = {}
        self.client_request_ids: set[str | int] = set()
        self.server_request_ids: set[str | int] = set()
        self.dynamic_call_ids: set[str] = set()
        self.initialize_answered = False
        self.initialized = False
        self.login_notifications = {
            "account/login/completed": 0,
            "account/updated": 0,
        }
        self.login_notification_sequence: list[str] = []
        self.login_request_seen = False
        self.login_answered = False
        self.thread_method: str | None = None
        self.thread_id: str | None = None
        self.turn_id: str | None = None
        self.turn_request_seen = False
        self.turn_started = 0
        self.turn_completed = 0
        self.interrupt_requests = 0
        self.auth_refresh_requests = 0
        self.auth_refresh_responses = 0
        self.item_states: dict[str, Literal["started", "completed"]] = {}
        self.dynamic_items: dict[str, dict[str, Any]] = {}
        self.last_emitted_at_ms = -1
        self.last_method: str | None = None

    @property
    def preflight_complete(self) -> bool:
        return self.preflight_index == len(
            Transport.PREFLIGHT_REQUEST_SEQUENCE
        )

    def _require_no_pending_client(self) -> None:
        if self.pending_client:
            raise ContiguousTaintError(
                "app-server client requests overlap or lack responses"
            )

    def client_request(self, payload: object) -> list[str]:
        value = _exact_mapping(
            payload,
            required=frozenset({"id", "method", "params"}),
        )
        request_id = value["id"]
        method = value["method"]
        if (
            not _request_id(request_id)
            or request_id in self.client_request_ids
            or not isinstance(method, str)
            or not method
        ):
            raise ContiguousTaintError(
                "app-server client request identity is malformed or reused"
            )
        self._require_no_pending_client()
        if self.pending_server:
            raise ContiguousTaintError(
                "client request overlapped an unresolved server request"
            )

        if not self.preflight_complete:
            expected = Transport.PREFLIGHT_REQUEST_SEQUENCE[
                self.preflight_index
            ]
            if method != expected:
                raise ContiguousTaintError(
                    "app-server preflight request order mismatch"
                )
            if method != "initialize" and not self.initialized:
                raise ContiguousTaintError(
                    "app-server request preceded initialized notification"
                )
            if (
                method == "account/read"
                and self.login_notifications
                != {
                    "account/login/completed": 1,
                    "account/updated": 1,
                }
            ):
                raise ContiguousTaintError(
                    "app-server login notifications are incomplete"
                )
            if (
                method == "account/login/start"
            ):
                _validate_redacted_login(value["params"])
                self.login_request_seen = True
            else:
                expected_params = _expected_preflight_params(
                    method, self.policy
                )
                if (
                    expected_params is None
                    or value["params"] != expected_params
                ):
                    raise ContiguousTaintError(
                        "app-server preflight request parameters drifted"
                    )
            if not isinstance(value["params"], dict):
                raise ContiguousTaintError(
                    "app-server request parameters are not an object"
                )
            self.preflight_index += 1
        elif self.thread_method is None:
            if method not in {"thread/start", "thread/resume"}:
                raise ContiguousTaintError(
                    "unexpected app-server request after preflight"
                )
            _validate_thread_request(
                method, value["params"], self.policy
            )
            self.thread_method = method
        elif not self.turn_request_seen:
            if method != "turn/start" or self.thread_id is None:
                raise ContiguousTaintError(
                    "turn/start did not follow an admitted thread"
                )
            _validate_turn_request(
                value["params"], self.policy, self.thread_id
            )
            self.turn_request_seen = True
        else:
            if (
                method != "turn/interrupt"
                or self.turn_id is None
                or self.turn_completed
                or not isinstance(value["params"], dict)
                or value["params"].get("threadId") != self.thread_id
                or value["params"].get("turnId") != self.turn_id
            ):
                raise ContiguousTaintError(
                    "unexpected active-turn client request"
                )
            self.interrupt_requests += 1
            if self.interrupt_requests != 1:
                raise ContiguousTaintError(
                    "turn/interrupt cardinality mismatch"
                )

        self.client_request_ids.add(request_id)
        self.pending_client[request_id] = method
        self.last_method = method
        # The lifecycle above is the authoritative allowlist. In particular,
        # config/read is mandatory preflight even though config mutation and
        # active-turn config access remain forbidden.
        return (
            _scan_model_value(value["params"])
            if method in {"thread/start", "thread/resume", "turn/start"}
            else []
        )

    def client_notification(self, payload: object) -> list[str]:
        value = _exact_mapping(
            payload,
            required=frozenset({"method", "params"}),
        )
        if (
            value["method"] != "initialized"
            or not self.initialize_answered
            or self.initialized
            or not isinstance(value["params"], dict)
        ):
            raise ContiguousTaintError(
                "app-server initialized lifecycle mismatch"
            )
        self.initialized = True
        self.last_method = "initialized"
        return []

    def server_response(self, payload: object) -> list[str]:
        if not isinstance(payload, dict) or "id" not in payload:
            raise ContiguousTaintError(
                "app-server server response schema mismatch"
            )
        keys = set(payload)
        if keys not in ({"id", "result"}, {"id", "error"}):
            raise ContiguousTaintError(
                "app-server server response schema mismatch"
            )
        request_id = payload["id"]
        method = self.pending_client.pop(request_id, None)
        if method is None:
            raise ContiguousTaintError(
                "app-server response has no pending client request"
            )
        hits: list[str] = []
        if "error" in payload:
            hits.append("app_server_error_response")
            hits.extend(_credential_value_hits(payload["error"]))
            hits.extend(_scan_model_value(payload["error"]))
        else:
            result = payload["result"]
            hits.extend(_credential_value_hits(result))
            hits.extend(_item_hits(result))
            if method in Transport.PREFLIGHT_REQUEST_SEQUENCE:
                _validate_preflight_result(
                    method, result, self.policy
                )
            if method == "initialize":
                self.initialize_answered = True
            elif method == "account/login/start":
                self.login_answered = True
            elif method in {"thread/start", "thread/resume"}:
                thread_id = _nested_identifier(result, "thread", "id")
                if thread_id is None:
                    raise ContiguousTaintError(
                        "thread response lacks a thread identifier"
                    )
                self.thread_id = thread_id
            elif method == "turn/start":
                turn_id = _nested_identifier(result, "turn", "id")
                if turn_id is None:
                    raise ContiguousTaintError(
                        "turn response lacks a turn identifier"
                    )
                self.turn_id = turn_id
        self.last_method = method
        return hits

    def server_request(self, payload: object) -> list[str]:
        value = _exact_mapping(
            payload,
            required=frozenset({"id", "method", "params"}),
        )
        request_id = value["id"]
        method = value["method"]
        if method == "account/chatgptAuthTokens/refresh":
            if (
                not _request_id(request_id)
                or request_id in self.server_request_ids
                or self.turn_id is None
                or self.turn_started != 1
                or self.turn_completed
                or self.pending_server
                or self.auth_refresh_requests
                >= self.policy.max_auth_refreshes
            ):
                raise ContiguousTaintError(
                    "unexpected or over-budget ChatGPT token refresh"
                )
            params = _exact_mapping(
                value["params"],
                required=frozenset({"reason"}),
                optional=frozenset({"previousAccountId"}),
            )
            if (
                params["reason"] != "unauthorized"
                or (
                    "previousAccountId" in params
                    and params["previousAccountId"]
                    not in {None, "REDACTED"}
                )
            ):
                raise ContiguousTaintError(
                    "ChatGPT token refresh is not lineage-redacted"
            )
            self.server_request_ids.add(request_id)
            self.auth_refresh_requests += 1
            self.pending_server[request_id] = (
                "__auth_refresh__",
                method,
            )
            self.last_method = method
            return _credential_value_hits(params)
        if (
            not _request_id(request_id)
            or request_id in self.server_request_ids
            or method != "item/tool/call"
            or self.turn_id is None
            or self.turn_started != 1
            or self.turn_completed
            or self.pending_server
        ):
            raise ContiguousTaintError(
                "unexpected or overlapping app-server server request"
            )
        params = _exact_mapping(
            value["params"],
            required=frozenset(
                {
                    "arguments",
                    "callId",
                    "threadId",
                    "tool",
                    "turnId",
                }
            ),
            optional=frozenset({"namespace"}),
        )
        call_id = params["callId"]
        tool = params["tool"]
        dynamic_item = (
            self.dynamic_items.get(call_id)
            if isinstance(call_id, str)
            else None
        )
        if (
            not isinstance(call_id, str)
            or not call_id
            or call_id in self.dynamic_call_ids
            or not isinstance(tool, str)
            or tool not in ALLOWED_DYNAMIC_TOOLS
            or params["threadId"] != self.thread_id
            or params["turnId"] != self.turn_id
            or not isinstance(params["arguments"], dict)
            or (
                "namespace" in params
                and params["namespace"] not in {None, "contiguous_lane"}
            )
            or dynamic_item is None
            or dynamic_item["state"] != "started"
            or dynamic_item["tool"] != tool
            or dynamic_item["arguments"] != params["arguments"]
            or dynamic_item["namespace"] != params.get("namespace")
        ):
            raise ContiguousTaintError(
                "dynamic tool request binding or schema mismatch"
            )
        self.server_request_ids.add(request_id)
        self.dynamic_call_ids.add(call_id)
        self.pending_server[request_id] = (call_id, tool)
        self.last_method = method
        hits = _scan_model_value(params["arguments"])
        hits.extend(
            "filesystem_boundary:" + hit
            for hit in Boundary.dynamic_tool_boundary_hits(
                tool, params["arguments"]
            )
        )
        return hits

    def client_response(self, payload: object) -> list[str]:
        if not isinstance(payload, dict) or "id" not in payload:
            raise ContiguousTaintError(
                "app-server client response schema mismatch"
            )
        keys = set(payload)
        if keys not in ({"id", "result"}, {"id", "error"}):
            raise ContiguousTaintError(
                "app-server client response schema mismatch"
            )
        pending = self.pending_server.pop(payload["id"], None)
        if pending is None:
            raise ContiguousTaintError(
                "client response has no pending server request"
            )
        hits: list[str] = []
        call_id, method = pending
        hits.extend(_credential_value_hits(payload))
        if method == "account/chatgptAuthTokens/refresh":
            if (
                self.auth_refresh_responses
                >= self.auth_refresh_requests
            ):
                raise ContiguousTaintError(
                    "ChatGPT token refresh response cardinality is invalid"
                )
            if "error" in payload:
                hits.append("app_server_error_response")
                hits.extend(_credential_value_hits(payload["error"]))
                hits.extend(_scan_model_value(payload["error"]))
            else:
                result = _exact_mapping(
                    payload["result"],
                    required=frozenset(
                        {"accessToken", "chatgptAccountId"}
                    ),
                    optional=frozenset({"chatgptPlanType"}),
                )
                if (
                    result["accessToken"] != "REDACTED"
                    or result["chatgptAccountId"] != "REDACTED"
                    or (
                        "chatgptPlanType" in result
                        and result["chatgptPlanType"] is not None
                        and not isinstance(
                            result["chatgptPlanType"], str
                        )
                    )
                ):
                    raise ContiguousTaintError(
                        "ChatGPT token refresh response is not redacted"
                    )
            self.auth_refresh_responses += 1
            self.last_method = (
                "account/chatgptAuthTokens/refresh:response"
            )
            return hits
        if "error" in payload:
            if not isinstance(payload["error"], dict):
                raise ContiguousTaintError(
                    "dynamic tool error response is malformed"
                )
            hits.extend(_scan_model_value(payload["error"]))
        else:
            result = _exact_mapping(
                payload["result"],
                required=frozenset({"contentItems", "success"}),
            )
            if (
                not isinstance(result["success"], bool)
                or not isinstance(result["contentItems"], list)
            ):
                raise ContiguousTaintError(
                    "dynamic tool result schema mismatch"
                )
            for item in result["contentItems"]:
                content = _exact_mapping(
                    item,
                    required=frozenset({"text", "type"}),
                )
                if (
                    content["type"] != "inputText"
                    or not isinstance(content["text"], str)
                ):
                    raise ContiguousTaintError(
                        "dynamic tool returned undeclared media"
                    )
                hits.extend(
                    _scan_text_full(
                        content["text"],
                        execution_surface=True,
                    )
                )
        self.dynamic_items[call_id]["response_seen"] = True
        self.last_method = "item/tool/call:response"
        return hits

    def _validate_item_notification(
        self,
        method: str,
        params: Mapping[str, Any],
    ) -> None:
        if method in {"item/started", "item/completed"}:
            item = params.get("item")
            item_id = (
                item.get("id") if isinstance(item, dict) else None
            )
            item_type = (
                item.get("type") if isinstance(item, dict) else None
            )
            if (
                not isinstance(item_id, str)
                or not item_id
                or item_type not in ALLOWED_TURN_ITEM_TYPES
            ):
                raise ContiguousTaintError(
                    "app-server emitted an inadmissible item"
                )
            if method == "item/started":
                if item_id in self.item_states:
                    raise ContiguousTaintError(
                        "app-server item started more than once"
                    )
                self.item_states[item_id] = "started"
                if item_type == "dynamicToolCall":
                    item_mapping = _exact_mapping(
                        item,
                        required=frozenset(
                            {
                                "arguments",
                                "id",
                                "status",
                                "tool",
                                "type",
                            }
                        ),
                        optional=frozenset(
                            {
                                "contentItems",
                                "durationMs",
                                "namespace",
                                "success",
                            }
                        ),
                    )
                    if (
                        not isinstance(item_mapping["arguments"], dict)
                        or item_mapping["tool"]
                        not in ALLOWED_DYNAMIC_TOOLS
                        or item_mapping.get("namespace")
                        not in {None, "contiguous_lane"}
                    ):
                        raise ContiguousTaintError(
                            "dynamic item schema mismatch"
                        )
                    self.dynamic_items[item_id] = {
                        "state": "started",
                        "tool": item_mapping["tool"],
                        "arguments": item_mapping["arguments"],
                        "namespace": item_mapping.get("namespace"),
                        "response_seen": False,
                    }
            else:
                if self.item_states.get(item_id) != "started":
                    raise ContiguousTaintError(
                        "app-server item completion is unpaired"
                    )
                if item_type == "dynamicToolCall":
                    dynamic_item = self.dynamic_items.get(item_id)
                    if (
                        dynamic_item is None
                        or not dynamic_item["response_seen"]
                        or item.get("tool") != dynamic_item["tool"]
                        or item.get("arguments")
                        != dynamic_item["arguments"]
                        or item.get("namespace")
                        != dynamic_item["namespace"]
                    ):
                        raise ContiguousTaintError(
                            "dynamic item completion is unpaired"
                        )
                    dynamic_item["state"] = "completed"
                self.item_states[item_id] = "completed"
        elif method in DELTA_NOTIFICATION_METHODS:
            item_id = params.get("itemId")
            if (
                not isinstance(item_id, str)
                or self.item_states.get(item_id) != "started"
            ):
                raise ContiguousTaintError(
                    "app-server item delta is unpaired"
                )

    def server_notification(self, payload: object) -> list[str]:
        value = _exact_mapping(
            payload,
            required=frozenset({"method", "params"}),
            optional=frozenset({"emittedAtMs"}),
        )
        method = value["method"]
        params = value["params"]
        if not isinstance(method, str) or not isinstance(params, dict):
            raise ContiguousTaintError(
                "app-server notification schema mismatch"
            )
        if "emittedAtMs" in value:
            emitted = value["emittedAtMs"]
            if (
                not isinstance(emitted, int)
                or isinstance(emitted, bool)
                or emitted < 0
                or emitted > MAX_EMITTED_AT_MS
                or emitted < self.last_emitted_at_ms
            ):
                raise ContiguousTaintError(
                    "notification emittedAtMs is invalid or nonmonotone"
                )
            self.last_emitted_at_ms = emitted
        if self.turn_completed and method not in POST_TURN_NOTIFICATION_METHODS:
            raise ContiguousTaintError(
                "app-server emitted active content after turn completion"
            )

        if self.thread_method is None:
            if method not in PREFLIGHT_SERVER_NOTIFICATION_METHODS:
                raise ContiguousTaintError(
                    "unknown app-server preflight notification"
                )
            if method in self.login_notifications:
                if (
                    not self.login_request_seen
                    or not self.login_answered
                ):
                    raise ContiguousTaintError(
                        "app-server login notification preceded login"
                    )
                self.login_notifications[method] += 1
                if self.login_notifications[method] != 1:
                    raise ContiguousTaintError(
                        "app-server login notification cardinality mismatch"
                    )
                self.login_notification_sequence.append(method)
                if self.login_notification_sequence not in (
                    ["account/login/completed"],
                    [
                        "account/login/completed",
                        "account/updated",
                    ],
                ):
                    raise ContiguousTaintError(
                        "app-server login notification order mismatch"
                    )
                if method == "account/login/completed":
                    completed = _exact_mapping(
                        params,
                        required=frozenset(
                            {"error", "loginId", "success"}
                        ),
                    )
                    if (
                        completed["error"] is not None
                        or completed["loginId"] is not None
                        or completed["success"] is not True
                    ):
                        raise ContiguousTaintError(
                            "external ChatGPT login did not complete"
                        )
                elif method == "account/updated":
                    updated = _exact_mapping(
                        params,
                        required=frozenset(
                            {"authMode", "planType"}
                        ),
                    )
                    if (
                        updated["authMode"]
                        != "chatgptAuthTokens"
                        or not isinstance(updated["planType"], str)
                        or not updated["planType"]
                    ):
                        raise ContiguousTaintError(
                            "account update changed authentication mode"
                        )
        else:
            if method not in TURN_SERVER_NOTIFICATION_METHODS:
                raise ContiguousTaintError(
                    "unknown app-server turn notification"
                )
            if method == "thread/started":
                notified = _nested_identifier(params, "thread", "id")
                if notified != self.thread_id:
                    raise ContiguousTaintError(
                        "thread notification references another thread"
                    )
            elif method == "turn/started":
                notified_thread = params.get("threadId")
                notified_turn = _nested_identifier(params, "turn", "id")
                if (
                    notified_thread != self.thread_id
                    or notified_turn != self.turn_id
                    or self.turn_started
                ):
                    raise ContiguousTaintError(
                        "turn/start notification binding mismatch"
                    )
                self.turn_started = 1
            elif method == "turn/completed":
                notified_thread = params.get("threadId")
                notified_turn = _nested_identifier(params, "turn", "id")
                if (
                    notified_thread != self.thread_id
                    or notified_turn != self.turn_id
                    or self.turn_started != 1
                    or self.turn_completed
                    or self.pending_server
                ):
                    raise ContiguousTaintError(
                        "turn/completed notification binding mismatch"
                    )
                self.turn_completed = 1
            self._validate_item_notification(method, params)

        self.last_method = method
        hits = _credential_value_hits(params)
        hits.extend(
            _scan_model_value(params)
            if method in MODEL_TEXT_NOTIFICATION_METHODS
            else _item_hits(params)
        )
        return hits

    def finish(self) -> None:
        if self.pending_client or self.pending_server:
            raise ContiguousTaintError(
                "app-server transcript ended with unresolved requests"
            )
        if (
            not self.preflight_complete
            or not self.initialize_answered
            or not self.initialized
            or self.login_notifications
            != {
                "account/login/completed": 1,
                "account/updated": 1,
            }
            or self.login_notification_sequence
            != [
                "account/login/completed",
                "account/updated",
            ]
            or self.thread_method not in {"thread/start", "thread/resume"}
            or self.thread_id is None
            or not self.turn_request_seen
            or self.turn_id is None
            or self.turn_started != 1
            or self.turn_completed != 1
            or self.auth_refresh_requests
            != self.auth_refresh_responses
            or any(
                state != "completed"
                for state in self.item_states.values()
            )
        ):
            raise ContiguousTaintError(
                "app-server transcript lifecycle is incomplete"
            )


def _regular_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ContiguousTaintError(
            f"taint evidence is unreadable: {path}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size > MAX_SCAN_BYTES
        ):
            raise ContiguousTaintError(
                f"taint evidence is aliased, nonregular, or oversized: {path}"
            )
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                raise ContiguousTaintError(
                    f"taint evidence truncated while reading: {path}"
                )
            chunks.append(block)
            remaining -= len(block)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                yield key
            yield from _strings(item)


def _method_hits(method: object) -> list[str]:
    if not isinstance(method, str) or not method:
        return ["malformed_app_server_method"]
    if (
        method in FORBIDDEN_EXACT_METHODS
        or any(
            method == fragment or method.startswith(fragment)
            for fragment in FORBIDDEN_APP_SERVER_METHOD_FRAGMENTS
        )
    ):
        return ["forbidden_app_server_method"]
    return []


def _item_hits(value: Any) -> list[str]:
    hits: list[str] = []
    if isinstance(value, dict):
        item_type = value.get("type")
        if isinstance(item_type, str) and item_type in FORBIDDEN_ITEM_TYPES:
            hits.append("forbidden_app_server_item")
        for key, child in value.items():
            if key in {"environment", "env", "environmentVariables"}:
                names: Iterable[str]
                if isinstance(child, dict):
                    names = child.keys()
                elif isinstance(child, list):
                    names = (
                        str(item).split("=", 1)[0] for item in child
                    )
                else:
                    names = ()
                if any(SECRET_NAME_RE.search(name) for name in names):
                    hits.append("credential_environment_exposure")
            hits.extend(_item_hits(child))
    elif isinstance(value, list):
        for child in value:
            hits.extend(_item_hits(child))
    return hits


def _scan_text_full(text: str, *, execution_surface: bool) -> list[str]:
    hits = GeneralTaint.scan_text(
        text,
        strip_inline_code=False,
        execution_surface=execution_surface,
    )
    # In the contiguous boundary, every general hit is actionable, including
    # harness introspection.  Do not subtract GeneralTaint.INFORMATIONAL_HITS.
    return list(hits)


def scan_app_server_jsonl(
    path: Path,
    *,
    policy: AppServerScanPolicy,
) -> ScanRecord:
    if not isinstance(policy, AppServerScanPolicy):
        raise ContiguousTaintError(
            "attempt-bound app-server scan policy is required"
        )
    raw = _regular_bytes(path)
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise ContiguousTaintError(
            "app-server transcript is not UTF-8"
        ) from exc
    if not raw.endswith(b"\n") or b"\r" in raw:
        raise ContiguousTaintError(
            "app-server transcript lacks exact LF-terminated byte coverage"
        )
    hits: list[str] = [
        "credential_sentinel_exposure"
        for sentinel in policy.secret_sentinels
        if sentinel.encode("utf-8") in raw
    ]
    expected_sequence = 1
    previous_digest: str | None = None
    lifecycle = _AppServerLifecycle(policy)
    for line_number, line in enumerate(text.split("\n")[:-1], 1):
        if not line:
            raise ContiguousTaintError(
                f"empty app-server JSONL record at line {line_number}"
            )
        try:
            record = _strict_json_loads(line)
        except ContiguousTaintError as exc:
            raise ContiguousTaintError(
                f"malformed app-server JSONL at line {line_number}"
            ) from exc
        if (
            not isinstance(record, dict)
            or set(record)
            != {
                "schema",
                "sequence",
                "previous_digest",
                "direction",
                "payload",
                "digest",
            }
            or record.get("schema") != SCHEMA
            or record.get("sequence") != expected_sequence
            or record.get("previous_digest") != previous_digest
            or record.get("direction")
            not in {"client_request", "client_notification",
                    "client_response", "server_response", "server_request",
                    "server_notification", "server_stderr"}
            or "payload" not in record
            or not isinstance(record.get("digest"), str)
        ):
            raise ContiguousTaintError(
                f"app-server chain schema mismatch at line {line_number}"
            )
        body = dict(record)
        digest = body.pop("digest")
        if (
            not SHA256_RE.fullmatch(digest)
            or hashlib.sha256(_canonical_json(body)).hexdigest() != digest
        ):
            raise ContiguousTaintError(
                f"app-server chain digest mismatch at line {line_number}"
            )
        payload = record["payload"]
        direction = record["direction"]
        if direction == "server_stderr":
            if not isinstance(payload, str):
                raise ContiguousTaintError(
                    "app-server stderr record is not text"
                )
            hits.append("app_server_stderr")
            hits.extend(
                _scan_text_full(payload, execution_surface=True)
            )
        else:
            if not isinstance(payload, dict):
                raise ContiguousTaintError(
                    f"app-server payload is not an object at line {line_number}"
                )
            method = payload.get("method")
            if direction == "client_request":
                hits.extend(lifecycle.client_request(payload))
            elif direction == "client_notification":
                hits.extend(lifecycle.client_notification(payload))
            elif direction == "client_response":
                hits.extend(lifecycle.client_response(payload))
            elif direction == "server_response":
                hits.extend(lifecycle.server_response(payload))
            elif direction == "server_request":
                hits.extend(lifecycle.server_request(payload))
            elif direction == "server_notification":
                hits.extend(lifecycle.server_notification(payload))
            else:  # pragma: no cover - direction was closed above.
                raise ContiguousTaintError(
                    f"unhandled app-server direction at line {line_number}"
                )
        previous_digest = digest
        expected_sequence += 1
    if expected_sequence == 1:
        raise ContiguousTaintError("app-server transcript is empty")
    lifecycle.finish()
    return ScanRecord(
        path=str(path),
        sha256=hashlib.sha256(raw).hexdigest(),
        size=len(raw),
        evidence_kind="app_server_jsonl",
        hits=tuple(sorted(set(hits))),
    )


def scan_jsonl(path: Path, *, evidence_kind: str) -> ScanRecord:
    raw = _regular_bytes(path)
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise ContiguousTaintError(
            f"{evidence_kind} is not UTF-8"
        ) from exc
    hits: list[str] = []
    records = 0
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line:
            raise ContiguousTaintError(
                f"empty {evidence_kind} JSONL record at line {line_number}"
            )
        try:
            value = _strict_json_loads(line)
        except ContiguousTaintError as exc:
            raise ContiguousTaintError(
                f"malformed {evidence_kind} JSONL at line {line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise ContiguousTaintError(
                f"{evidence_kind} JSONL record is not an object"
            )
        hits.extend(_item_hits(value))
        for item in _strings(value):
            hits.extend(_scan_text_full(item, execution_surface=True))
        records += 1
    if records == 0:
        raise ContiguousTaintError(f"{evidence_kind} transcript is empty")
    return ScanRecord(
        path=str(path),
        sha256=hashlib.sha256(raw).hexdigest(),
        size=len(raw),
        evidence_kind=evidence_kind,
        hits=tuple(sorted(set(hits))),
    )


def scan_regular_file(path: Path, *, evidence_kind: str) -> ScanRecord:
    raw = _regular_bytes(path)
    text = raw.decode("utf-8", errors="strict")
    hits = _scan_text_full(text, execution_surface=True)
    suffix = Path(path).suffix.lower()
    if suffix in Boundary.SOURCE_SUFFIXES:
        if suffix in {".py", ".pyw"}:
            boundary = Boundary.scan_python_source(
                text,
                logical_path=Path(path).name,
                arena_module_root=None,
            )
        else:
            boundary = Boundary.scan_shell_command(
                text,
                logical_path=Path(path).name,
                line=1,
                arena_module_root=None,
            )
        hits.extend(
            "filesystem_boundary:" + finding.code
            for finding in boundary
        )
    return ScanRecord(
        path=str(path),
        sha256=hashlib.sha256(raw).hexdigest(),
        size=len(raw),
        evidence_kind=evidence_kind,
        hits=tuple(sorted(set(hits))),
    )


def _read_inventory_file(
    root_descriptor: int,
    relative_text: str,
    expected_sha256: str,
    expected_bytes: int,
) -> bytes:
    relative = PurePosixPath(relative_text)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ContiguousTaintError(
            "retained inventory path is unsafe"
        )
    directory_descriptor = os.dup(root_descriptor)
    try:
        for component in relative.parts[:-1]:
            child = os.open(
                component,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_descriptor,
            )
            os.close(directory_descriptor)
            directory_descriptor = child
        descriptor = os.open(
            relative.parts[-1],
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_descriptor,
        )
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size != expected_bytes
                or before.st_size
                > Transport.MAX_APP_SERVER_STATE_FILE_BYTES
            ):
                raise ContiguousTaintError(
                    "retained file identity differs from inventory"
                )
            chunks: list[bytes] = []
            remaining = expected_bytes
            digest = hashlib.sha256()
            while remaining:
                block = os.read(
                    descriptor, min(1024 * 1024, remaining)
                )
                if not block:
                    raise ContiguousTaintError(
                        "retained file truncated during scan"
                    )
                chunks.append(block)
                digest.update(block)
                remaining -= len(block)
            if os.read(descriptor, 1):
                raise ContiguousTaintError(
                    "retained file grew during scan"
                )
            after = os.fstat(descriptor)
            if (
                (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                )
                != (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                )
                or digest.hexdigest() != expected_sha256
            ):
                raise ContiguousTaintError(
                    "retained file changed after inventory"
                )
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise ContiguousTaintError(
            "retained descriptor traversal failed"
        ) from exc
    finally:
        os.close(directory_descriptor)


def scan_controller_state(
    root: Path,
    *,
    inventory: Transport.ControllerStateInventory,
    canaries: tuple[LiveCanary, ...],
) -> ControllerStateScan:
    """Reopen and scan every exact inventoried state byte.

    The controller has already stopped when this runs.  The inventory is still
    treated as untrusted: every path is traversed relative to a descriptor,
    every file's identity/size/hash is rechecked, and a second complete
    inventory must agree before a clean result can be returned.  Canary values
    are live-only inputs and are represented in the receipt solely by count.
    """

    selected = Path(root)
    live_canaries = validate_live_canaries(canaries)
    encoded_canaries = tuple(
        item.value.encode("utf-8")
        for item in live_canaries
    )
    inventory_fields = (
        "tree_sha256",
        "inventory_sha256",
        "file_count",
        "total_bytes",
        "files",
        "secret_occurrences",
    )
    if (
        not selected.is_absolute()
        or any(not hasattr(inventory, name) for name in inventory_fields)
        or not SHA256_RE.fullmatch(
            str(getattr(inventory, "tree_sha256", ""))
        )
        or not SHA256_RE.fullmatch(
            str(getattr(inventory, "inventory_sha256", ""))
        )
        or not _plain_int(getattr(inventory, "file_count", None))
        or getattr(inventory, "file_count", -1) < 0
        or not _plain_int(getattr(inventory, "total_bytes", None))
        or getattr(inventory, "total_bytes", -1) < 0
        or not _plain_int(
            getattr(inventory, "secret_occurrences", None)
        )
        or getattr(inventory, "secret_occurrences", -1) < 0
        or not isinstance(getattr(inventory, "files", None), tuple)
        or any(
            not isinstance(row, tuple)
            or len(row) != 3
            or not isinstance(row[0], str)
            or not row[0]
            or not SHA256_RE.fullmatch(str(row[1]))
            or not _plain_int(row[2])
            or row[2] < 0
            for row in getattr(inventory, "files", ())
        )
    ):
        raise ContiguousTaintError(
            "controller-state scan binding is malformed"
        )
    try:
        root_descriptor = os.open(
            selected,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise ContiguousTaintError(
            "controller-state root is unreadable"
        ) from exc
    records: list[ScanRecord] = []
    all_hits: set[str] = set()
    canary_occurrences = 0

    try:
        root_metadata = os.fstat(root_descriptor)
        if not stat.S_ISDIR(root_metadata.st_mode):
            raise ContiguousTaintError(
                "controller-state root is not a directory"
            )
        for relative_text, expected_sha256, expected_bytes in inventory.files:
            raw = _read_inventory_file(
                root_descriptor,
                relative_text, expected_sha256, expected_bytes
            )
            text = raw.decode("utf-8", errors="ignore")
            hits = set(
                _scan_text_full(text, execution_surface=True)
            )
            if JWT_LIKE_RE.search(text) or BEARER_SECRET_RE.search(text):
                hits.add("credential_shaped_state")
            if STATE_ENV_ASSIGNMENT_RE.search(text):
                hits.add("host_environment_marker")
            occurrences = sum(
                raw.count(value) for value in encoded_canaries
            )
            if occurrences:
                canary_occurrences += occurrences
                hits.add("controller_state_canary_exposure")
            record = ScanRecord(
                path=relative_text,
                sha256=expected_sha256,
                size=expected_bytes,
                evidence_kind="controller_state",
                hits=tuple(sorted(hits)),
            )
            records.append(record)
            all_hits.update(hits)
    finally:
        os.close(root_descriptor)
    try:
        final_inventory = Transport.inventory_controller_state(selected)
    except Transport.AppServerTransportError as exc:
        raise ContiguousTaintError(
            "controller-state changed after its taint scan"
        ) from exc
    if (
        final_inventory.tree_sha256 != inventory.tree_sha256
        or final_inventory.inventory_sha256
        != inventory.inventory_sha256
        or final_inventory.files != inventory.files
        or final_inventory.file_count != inventory.file_count
        or final_inventory.total_bytes != inventory.total_bytes
    ):
        raise ContiguousTaintError(
            "controller-state inventory changed during taint scan"
        )
    normalized_hits = tuple(sorted(all_hits))
    return ControllerStateScan(
        tree_sha256=inventory.tree_sha256,
        inventory_sha256=inventory.inventory_sha256,
        file_count=inventory.file_count,
        total_bytes=inventory.total_bytes,
        records=tuple(records),
        hits=normalized_hits,
        canary_count=len(live_canaries),
        canary_commitments=tuple(
            (
                item.category,
                item.location_name,
                item.provenance,
                item.commitment()["commitment_sha256"],
            )
            for item in live_canaries
        ),
        canary_occurrences=canary_occurrences,
        status=(
            "TAINT"
            if normalized_hits or canary_occurrences
            else "CLEAN"
        ),
    )


def scan_retained_canary_roots(
    roots: Mapping[str, Path],
    *,
    canaries: tuple[LiveCanary, ...],
) -> RetainedCanaryScan:
    """Scan every regular byte under disjoint retained evidence roots."""

    live_canaries = validate_live_canaries(canaries)
    if (
        not isinstance(roots, Mapping)
        or not roots
        or any(
            not isinstance(label, str)
            or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", label) is None
            or not isinstance(path, Path)
            or not path.is_absolute()
            for label, path in roots.items()
        )
    ):
        raise ContiguousTaintError(
            "retained canary roots are malformed"
        )
    selected_roots = tuple(
        sorted((label, Path(path)) for label, path in roots.items())
    )
    resolved = tuple(
        path.resolve(strict=True) for _label, path in selected_roots
    )
    if len(set(resolved)) != len(resolved) or any(
        first == second
        or first in second.parents
        or second in first.parents
        for index, first in enumerate(resolved)
        for second in resolved[index + 1:]
    ):
        raise ContiguousTaintError(
            "retained canary roots overlap"
        )
    encoded = tuple(
        (item.category, item.value.encode("utf-8"))
        for item in live_canaries
    )
    root_rows: list[tuple[str, str, str, int, int]] = []
    records: list[ScanRecord] = []
    all_hits: set[str] = set()
    occurrences = 0
    for label, root in selected_roots:
        try:
            inventory = Transport.inventory_controller_state(root)
            descriptor = os.open(
                root,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
        except (OSError, Transport.AppServerTransportError) as exc:
            raise ContiguousTaintError(
                f"retained {label} root is not descriptor-safe"
            ) from exc
        try:
            for relative, digest, byte_count in inventory.files:
                raw = _read_inventory_file(
                    descriptor, relative, digest, byte_count
                )
                categories = tuple(
                    sorted(
                        category
                        for category, needle in encoded
                        if needle in raw
                    )
                )
                file_occurrences = sum(
                    raw.count(needle) for _category, needle in encoded
                )
                occurrences += file_occurrences
                hits = tuple(
                    f"containment_canary_exposure:{category}"
                    for category in categories
                )
                all_hits.update(hits)
                records.append(
                    ScanRecord(
                        path=f"{label}/{relative}",
                        sha256=digest,
                        size=byte_count,
                        evidence_kind="retained_canary_surface",
                        hits=hits,
                    )
                )
        finally:
            os.close(descriptor)
        final_inventory = Transport.inventory_controller_state(root)
        if final_inventory != inventory:
            raise ContiguousTaintError(
                f"retained {label} root changed during canary scan"
            )
        root_rows.append(
            (
                label,
                inventory.tree_sha256,
                inventory.inventory_sha256,
                inventory.file_count,
                inventory.total_bytes,
            )
        )
    normalized_hits = tuple(sorted(all_hits))
    return RetainedCanaryScan(
        root_inventories=tuple(root_rows),
        records=tuple(records),
        hits=normalized_hits,
        canary_commitments=tuple(
            (
                item.category,
                item.location_name,
                item.provenance,
                item.commitment()["commitment_sha256"],
            )
            for item in live_canaries
        ),
        canary_occurrences=occurrences,
        status=(
            "TAINT" if normalized_hits or occurrences else "CLEAN"
        ),
    )


def scan_canaries_in_file(
    path: Path,
    *,
    canaries: tuple[LiveCanary, ...],
    evidence_kind: str,
) -> ScanRecord:
    live_canaries = validate_live_canaries(canaries)
    if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", evidence_kind) is None:
        raise ContiguousTaintError(
            "canary file evidence kind is malformed"
        )
    raw = _regular_bytes(Path(path))
    categories = tuple(
        sorted(
            item.category
            for item in live_canaries
            if item.value.encode("utf-8") in raw
        )
    )
    return ScanRecord(
        path=str(path),
        sha256=hashlib.sha256(raw).hexdigest(),
        size=len(raw),
        evidence_kind=evidence_kind,
        hits=tuple(
            f"containment_canary_exposure:{category}"
            for category in categories
        ),
    )


def scan_evidence(
    path: Path,
    *,
    evidence_kind: Literal[
        "app_server_jsonl",
        "backend_jsonl",
        "container_stdout",
        "container_stderr",
        "candidate_output",
    ],
    app_server_policy: AppServerScanPolicy | None = None,
) -> ScanRecord:
    selected = Path(path)
    if evidence_kind == "app_server_jsonl":
        if app_server_policy is None:
            raise ContiguousTaintError(
                "app-server evidence lacks an attempt-bound scan policy"
            )
        return scan_app_server_jsonl(
            selected, policy=app_server_policy
        )
    if evidence_kind == "backend_jsonl":
        return scan_jsonl(selected, evidence_kind=evidence_kind)
    return scan_regular_file(selected, evidence_kind=evidence_kind)
