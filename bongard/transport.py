"""Hardened, non-interactive Codex transport for Bongard vision turns.

The scientific runners deliberately do not give Codex a repository checkout or
an editable experiment workspace.  Each turn receives only copied PNGs, the
prompt over stdin, and a strict JSON output schema.  Codex runs ephemerally
in a read-only sandbox with approvals, shell, search, apps, hooks, memories,
goals, remote plugins, and sub-agents disabled.  The caller applies a validated
proposal to its own workspace transactionally.

The ephemeral ``CODEX_HOME`` contains only invocation authentication when
needed and the exact signed cloud-policy cache when one is available.  It does
not inherit user configuration, memories, sessions, skills, or plugins.

``codex exec --json`` currently records the requested model only indirectly:
the exact ``--model`` flag is part of the invocation, while the documented
JSONL completion event need not repeat the served model.  Receipts therefore
distinguish provider-reported identity from explicit-CLI-request identity
instead of claiming stronger evidence than the CLI emits.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import signal
import shutil
import stat
import subprocess
import tempfile
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


CODEX_RECEIPT_SCHEMA = "bongard.codex-cli-proposer-receipt/v3"
CODEX_ISOLATION_POLICY = (
    "ephemeral-image-only-view-read-only-no-tools-signed-policy-cache-"
    "no-user-config-rules/v2")
STRUCTURED_INPUT_DIGEST_SCHEMA = "bongard.codex-structured-input/v1"
NAMED_IMAGE_INPUT_DIGEST_SCHEMA = "bongard.codex-named-image-input/v1"
DEFAULT_CODEX_MODEL = "gpt-5.6-sol"
DEFAULT_REASONING_EFFORT = "medium"
REASONING_EFFORTS = frozenset({
    "minimal", "low", "medium", "high", "xhigh", "max", "ultra",
})

MAX_PANEL_PNG_BYTES = 4_000_000
MAX_TASK_UTF8_BYTES = 4_000_000
MAX_SCHEMA_UTF8_BYTES = 1_000_000
MAX_STDOUT_BYTES = 4_000_000
MAX_STDERR_BYTES = 1_000_000
MAX_JSONL_EVENTS = 20_000
MAX_STRUCTURED_OUTPUT_CANONICAL_BYTES = MAX_STDOUT_BYTES
MAX_AUTH_FILE_BYTES = 1_000_000
MAX_CLOUD_CONFIG_BUNDLE_CACHE_BYTES = 1_000_000
MAX_CODEX_LAUNCHER_BYTES = 20_000_000
MAX_NAMED_IMAGES = 32
PROCESS_GROUP_GRACE_SECONDS = 2
SEMANTIC_INK_THRESHOLD = 128
_PANEL_SCHEMA = "bongard.panel-canonical/v1"
_PACKED_BINARY_ENCODING = "numpy-packbits-little-base64/v1"
_MAX_PANEL_ELEMENTS = 16_777_216

WORKSPACE_ROOT = os.path.realpath(
    os.path.join(os.path.dirname(__file__), ".."))

_MODEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_IMAGE_NAME_RE = re.compile(r"[a-z][a-z0-9_-]{0,63}\.png\Z")
_STRICT_SCHEMA_FORBIDDEN_KEYWORDS = frozenset({
    "oneOf",
    "uniqueItems",
    "minItems",
    "maxItems",
    "minimum",
    "maximum",
    "minLength",
    "maxLength",
    "const",
    "not",
})
_PANEL_NAMES = tuple(
    [f"pos_{index}.png" for index in range(6)]
    + [f"neg_{index}.png" for index in range(6)])
_CLOUD_CONFIG_BUNDLE_CACHE = "cloud-config-bundle-cache.json"
_DISABLED_FEATURES = (
    "shell_tool",
    "unified_exec",
    "apps",
    "multi_agent",
    "hooks",
    "goals",
    "memories",
    "remote_plugin",
    "skill_mcp_dependency_install",
    "plugins",
    "plugin_sharing",
    "skill_search",
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "computer_use",
    "image_generation",
    "in_app_browser",
    "code_mode_host",
    "auth_elicitation",
    "tool_call_mcp_elicitation",
    "tool_suggest",
    "workspace_dependencies",
    "network_proxy",
    "standalone_web_search",
)
_ALLOWED_ITEM_TYPES = frozenset({"reasoning", "agent_message"})
_RECEIPT_KEYS = frozenset({
    "schema",
    "source",
    "requested_model",
    "reported_model",
    "model_identity_evidence",
    "requested_reasoning_effort",
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "thread_id",
    "codex_cli_version",
    "codex_launcher_digest",
    "cloud_config_bundle_cache_binding",
    "task_digest",
    "current_source_digest",
    "current_log_digest",
    "prompt_digest",
    "input_digest_schema",
    "input_digest",
    "output_schema_digest",
    "panel_view_digest",
    "panel_set_digest",
    "structured_output_digest",
    "proposed_source_digest",
    "proposed_log_digest",
    "event_stream_digest",
    "event_types",
    "item_types",
    "isolation_policy",
    "outcome",
    "receipt_digest",
})


class CodexProposerFailure(RuntimeError):
    """Codex failed before producing an admissible scientific proposal."""


@dataclass(frozen=True)
class CodexReceipt:
    schema: str
    source: str
    requested_model: str
    reported_model: str
    model_identity_evidence: str
    requested_reasoning_effort: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_output_tokens: int
    thread_id: str
    codex_cli_version: str
    codex_launcher_digest: str
    cloud_config_bundle_cache_binding: str
    task_digest: str
    current_source_digest: str
    current_log_digest: str
    prompt_digest: str
    input_digest_schema: str
    input_digest: str
    output_schema_digest: str
    panel_view_digest: str
    panel_set_digest: str
    structured_output_digest: str
    proposed_source_digest: str
    proposed_log_digest: str
    event_stream_digest: str
    event_types: tuple[str, ...]
    item_types: tuple[str, ...]
    isolation_policy: str
    outcome: str
    receipt_digest: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["event_types"] = list(self.event_types)
        value["item_types"] = list(self.item_types)
        return value


@dataclass(frozen=True)
class CodexStructuredResult:
    payload: dict[str, Any]
    receipt: CodexReceipt


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise CodexProposerFailure(
            f"value is not canonical finite UTF-8 JSON: {exc}") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _bytes_digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _strict_json(text: str, description: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")),
        )
    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
        raise CodexProposerFailure(
            f"Codex {description} is malformed JSON: {exc}") from exc


def _bounded_utf8(value: Any, name: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise CodexProposerFailure(f"Codex {name} must be a string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError as exc:
        raise CodexProposerFailure(
            f"Codex {name} is not valid UTF-8 text") from exc
    if len(encoded) > maximum:
        raise CodexProposerFailure(
            f"Codex {name} exceeds {maximum} UTF-8 bytes")
    if "\x00" in value:
        raise CodexProposerFailure(f"Codex {name} contains a NUL byte")
    return value


def _validate_model(model: str) -> str:
    if not isinstance(model, str) or _MODEL_RE.fullmatch(model) is None:
        raise CodexProposerFailure("Codex model identifier is invalid")
    return model


def _validate_reasoning_effort(reasoning_effort: str) -> str:
    if not isinstance(reasoning_effort, str) \
            or reasoning_effort not in REASONING_EFFORTS:
        raise CodexProposerFailure(
            "Codex reasoning effort is not an allowlisted exact value")
    return reasoning_effort


def _path_is_within(path: str, root: str) -> bool:
    try:
        return os.path.commonpath((path, root)) == root
    except ValueError:
        return False


def _safe_temp_parent() -> str:
    """Resolve the temp root and fail before creation if it is in the repo."""
    configured = next((
        os.environ[key] for key in ("TMPDIR", "TEMP", "TMP")
        if isinstance(os.environ.get(key), str) and os.environ[key]
    ), None)
    candidate = configured if configured is not None else tempfile.gettempdir()
    candidate = os.path.realpath(os.path.abspath(candidate))
    try:
        info = os.stat(candidate)
    except OSError as exc:
        raise CodexProposerFailure(
            "Codex temporary parent is unavailable") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise CodexProposerFailure("Codex temporary parent is not a directory")
    if _path_is_within(candidate, WORKSPACE_ROOT):
        raise CodexProposerFailure(
            "Codex temporary parent must be outside the workspace")
    return candidate


def _minimal_environment(
        *, codex_home: str | None = None,
        temp_parent: str | None = None) -> dict[str, str]:
    """Keep CLI authentication/routing necessities, not ambient API secrets.

    ``CODEX_API_KEY`` is the CLI's documented invocation-scoped automation
    credential and is retained when present.  The model has no shell or other
    tool with which to inspect this process environment.
    """
    allowed = {
        "PATH", "HOME", "CODEX_HOME", "TMPDIR", "TMP", "TEMP",
        "LANG", "LC_ALL", "LC_CTYPE", "TZ", "SSL_CERT_FILE",
        "SSL_CERT_DIR", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy",
        "CODEX_API_KEY",
    }
    env = {key: value for key, value in os.environ.items()
           if key in allowed and isinstance(value, str)}
    env["TERM"] = "dumb"
    env["NO_COLOR"] = "1"
    if codex_home is not None:
        env["CODEX_HOME"] = codex_home
    if temp_parent is not None:
        env["TMPDIR"] = temp_parent
        env["TMP"] = temp_parent
        env["TEMP"] = temp_parent
    return env


def _require_outside_bongard(path: str, description: str) -> None:
    resolved = os.path.realpath(path)
    try:
        inside = os.path.commonpath((WORKSPACE_ROOT, resolved)) == WORKSPACE_ROOT
    except ValueError as exc:
        raise CodexProposerFailure(
            f"cannot establish {description} isolation") from exc
    if inside:
        raise CodexProposerFailure(
            f"{description} must be outside the repository working tree")


@dataclass(frozen=True)
class _StagedCloudPolicyCache:
    path: str
    binding: str
    identity: tuple[int, int, int, int, int, int, int] | None


@dataclass(frozen=True)
class CloudPolicyCacheSnapshot:
    """One exact signed cache preimage frozen for a multi-call episode.

    ``None`` records that no cache existed at snapshot time.  The bytes are
    immutable and revalidated before every staging operation; callers cannot
    substitute a digest without also supplying the exact signed envelope.
    """

    data: bytes | None

    def __post_init__(self) -> None:
        if self.data is not None:
            if not isinstance(self.data, bytes):
                raise CodexProposerFailure(
                    "cloud policy cache snapshot must contain exact bytes"
                )
            _validate_cloud_policy_cache(self.data)

    @property
    def binding(self) -> str:
        if self.data is None:
            return "absent"
        return "sha256:" + _bytes_digest(self.data)


def _file_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _source_codex_home() -> str:
    configured = os.environ.get("CODEX_HOME")
    if configured:
        return configured
    user_home = os.environ.get("HOME")
    return os.path.join(user_home, ".codex") if user_home else ""


def _read_optional_cloud_policy_cache(source: str) -> bytes | None:
    """Read one bounded cache file by descriptor, never through a symlink."""

    try:
        before = os.lstat(source)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise CodexProposerFailure(
            "cannot inspect Codex cloud policy cache") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or not 0 < before.st_size <= MAX_CLOUD_CONFIG_BUNDLE_CACHE_BYTES
    ):
        raise CodexProposerFailure(
            "Codex cloud policy cache is not a bounded, singly-linked file"
        )
    if not hasattr(os, "O_NOFOLLOW"):
        raise CodexProposerFailure(
            "platform cannot safely stage the Codex cloud policy cache"
        )
    try:
        descriptor = os.open(
            source,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
        )
    except OSError as exc:
        raise CodexProposerFailure("cannot open Codex cloud policy cache") from exc
    try:
        opened = os.fstat(descriptor)
        identity = _file_identity(opened)
        if identity != _file_identity(before):
            raise CodexProposerFailure(
                "Codex cloud policy cache changed while being opened"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            block = os.read(
                descriptor,
                min(
                    65_536,
                    MAX_CLOUD_CONFIG_BUNDLE_CACHE_BYTES + 1 - total,
                ),
            )
            if not block:
                break
            chunks.append(block)
            total += len(block)
            if total > MAX_CLOUD_CONFIG_BUNDLE_CACHE_BYTES:
                raise CodexProposerFailure(
                    "Codex cloud policy cache became oversized"
                )
        if total != opened.st_size or _file_identity(os.fstat(descriptor)) != identity:
            raise CodexProposerFailure(
                "Codex cloud policy cache changed while being read"
            )
    finally:
        os.close(descriptor)
    try:
        after = os.lstat(source)
    except OSError as exc:
        raise CodexProposerFailure(
            "Codex cloud policy cache path changed while being read"
        ) from exc
    if _file_identity(after) != identity:
        raise CodexProposerFailure(
            "Codex cloud policy cache path changed while being read"
        )
    return b"".join(chunks)


def _validate_cloud_policy_cache(data: bytes) -> None:
    """Validate only the signed envelope; the Codex CLI validates its payload."""

    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise CodexProposerFailure(
            "Codex cloud policy cache is not UTF-8 JSON"
        ) from exc
    decoded = _strict_json(text, "cloud policy cache")
    if (
        not isinstance(decoded, dict)
        or set(decoded) != {"signed_payload", "signature"}
        or not isinstance(decoded["signed_payload"], dict)
        or not isinstance(decoded["signature"], str)
        or not decoded["signature"]
    ):
        raise CodexProposerFailure(
            "Codex cloud policy cache signed envelope is malformed"
        )


def snapshot_cloud_policy_cache() -> CloudPolicyCacheSnapshot:
    """Freeze the exact currently available signed cache for one episode."""

    source_home = _source_codex_home()
    if not source_home:
        return CloudPolicyCacheSnapshot(None)
    source = os.path.join(source_home, _CLOUD_CONFIG_BUNDLE_CACHE)
    data = _read_optional_cloud_policy_cache(source)
    return CloudPolicyCacheSnapshot(data)


def _stage_cloud_policy_cache(
    codex_home: str,
    snapshot: CloudPolicyCacheSnapshot | None = None,
) -> _StagedCloudPolicyCache:
    """Copy one exact signed workspace-policy snapshot into an ephemeral home.

    A standalone transport snapshots the live source at call time.  A
    multi-call episode passes one prior snapshot so unrelated refreshes of the
    global cache cannot change its policy input halfway through the episode.
    """

    target = os.path.join(codex_home, _CLOUD_CONFIG_BUNDLE_CACHE)
    frozen = snapshot if snapshot is not None else snapshot_cloud_policy_cache()
    if not isinstance(frozen, CloudPolicyCacheSnapshot):
        raise CodexProposerFailure("cloud policy cache snapshot type is invalid")
    # Revalidate at every use.  This is cheap and prevents a deserialized or
    # otherwise forged snapshot object from bypassing the envelope boundary.
    data = frozen.data
    if data is None:
        return _StagedCloudPolicyCache(target, "absent", None)
    _validate_cloud_policy_cache(data)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(target, flags, 0o600)
    except OSError as exc:
        raise CodexProposerFailure(
            "cannot create staged Codex cloud policy cache"
        ) from exc
    try:
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise CodexProposerFailure(
                    "could not completely stage Codex cloud policy cache"
                )
            offset += written
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size != len(data)
            or stat.S_IMODE(opened.st_mode) != 0o600
        ):
            raise CodexProposerFailure(
                "staged Codex cloud policy cache has unsafe metadata"
            )
    finally:
        os.close(descriptor)
    try:
        staged = os.lstat(target)
    except OSError as exc:
        raise CodexProposerFailure(
            "staged Codex cloud policy cache disappeared"
        ) from exc
    identity = _file_identity(staged)
    if identity != _file_identity(opened):
        raise CodexProposerFailure(
            "staged Codex cloud policy cache changed after copy"
        )
    return _StagedCloudPolicyCache(
        target,
        frozen.binding,
        identity,
    )


def _recheck_staged_cloud_policy_cache(stage: _StagedCloudPolicyCache) -> None:
    """Recheck the exact staged path immediately around process launch."""

    if stage.binding == "absent":
        try:
            os.lstat(stage.path)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise CodexProposerFailure(
                "cannot recheck absent staged Codex cloud policy cache"
            ) from exc
        raise CodexProposerFailure(
            "Codex cloud policy cache appeared after absence was committed"
        )
    if stage.identity is None:
        raise CodexProposerFailure("Codex cloud policy cache binding is incomplete")
    try:
        before = os.lstat(stage.path)
    except OSError as exc:
        raise CodexProposerFailure(
            "staged Codex cloud policy cache disappeared before launch"
        ) from exc
    if _file_identity(before) != stage.identity:
        raise CodexProposerFailure(
            "staged Codex cloud policy cache metadata changed before launch"
        )
    data = _read_stable_view_file(
        stage.path,
        maximum=MAX_CLOUD_CONFIG_BUNDLE_CACHE_BYTES,
        description="staged cloud policy cache",
    )
    try:
        after = os.lstat(stage.path)
    except OSError as exc:
        raise CodexProposerFailure(
            "staged Codex cloud policy cache disappeared during recheck"
        ) from exc
    if _file_identity(after) != stage.identity:
        raise CodexProposerFailure(
            "staged Codex cloud policy cache metadata changed during recheck"
        )
    if "sha256:" + _bytes_digest(data) != stage.binding:
        raise CodexProposerFailure(
            "staged Codex cloud policy cache bytes changed before launch"
        )


def _stage_codex_auth(auth_dir: str) -> None:
    """Copy only the normal CLI auth file into an otherwise empty CODEX_HOME."""
    os.chmod(auth_dir, 0o700)
    if os.environ.get("CODEX_API_KEY"):
        return
    source_home = _source_codex_home()
    if not source_home:
        return
    source = os.path.join(source_home, "auth.json")
    if not os.path.exists(source):
        # OS-keychain installations need no file copy.
        return
    try:
        before = os.lstat(source)
    except OSError as exc:
        raise CodexProposerFailure("cannot inspect Codex CLI auth file") from exc
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 \
            or not 0 < before.st_size <= MAX_AUTH_FILE_BYTES:
        raise CodexProposerFailure("Codex CLI auth file is not a bounded file")
    descriptor = os.open(
        source,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        opened = os.fstat(descriptor)
        identity = (
            opened.st_dev, opened.st_ino, opened.st_size,
            opened.st_mtime_ns, opened.st_ctime_ns,
        )
        if opened.st_nlink != 1 or identity != (
                before.st_dev, before.st_ino, before.st_size,
                before.st_mtime_ns, before.st_ctime_ns):
            raise CodexProposerFailure("Codex CLI auth file changed during read")
        data = os.read(descriptor, MAX_AUTH_FILE_BYTES + 1)
        after = os.fstat(descriptor)
        if len(data) != opened.st_size or len(data) > MAX_AUTH_FILE_BYTES \
                or after.st_nlink != 1 or (
                    after.st_dev, after.st_ino, after.st_size,
                    after.st_mtime_ns, after.st_ctime_ns) != identity:
            raise CodexProposerFailure("Codex CLI auth file changed during read")
    finally:
        os.close(descriptor)
    try:
        decoded = json.loads(data.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CodexProposerFailure("Codex CLI auth file is malformed") from exc
    if not isinstance(decoded, dict):
        raise CodexProposerFailure("Codex CLI auth file is malformed")
    target = os.path.join(auth_dir, "auth.json")
    output = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(output, data[offset:])
        os.fsync(output)
    finally:
        os.close(output)


def _read_regular_png(path: str) -> bytes:
    if not isinstance(path, str) or not os.path.isabs(path):
        raise CodexProposerFailure("panel PNG paths must be absolute")
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise CodexProposerFailure(
            f"cannot inspect proposer panel {os.path.basename(path)!r}: {exc}") \
            from exc
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 \
            or not 0 < before.st_size <= MAX_PANEL_PNG_BYTES:
        raise CodexProposerFailure(
            f"proposer panel {os.path.basename(path)!r} is not a bounded, "
            "singly-linked regular file")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CodexProposerFailure(
            f"cannot open proposer panel {os.path.basename(path)!r}: {exc}") \
            from exc
    try:
        info = os.fstat(descriptor)
        identity = (
            info.st_dev, info.st_ino, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns,
        )
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 \
                or identity != (
                    before.st_dev, before.st_ino, before.st_size,
                    before.st_mtime_ns, before.st_ctime_ns):
            raise CodexProposerFailure(
                f"proposer panel {os.path.basename(path)!r} is not a bounded "
                "regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, min(1_048_576,
                                             MAX_PANEL_PNG_BYTES + 1 - total))
            if not block:
                break
            chunks.append(block)
            total += len(block)
            if total > MAX_PANEL_PNG_BYTES:
                raise CodexProposerFailure(
                    f"proposer panel {os.path.basename(path)!r} is oversized")
        after = os.fstat(descriptor)
        if after.st_nlink != 1 or (
                after.st_dev, after.st_ino, after.st_size,
                after.st_mtime_ns, after.st_ctime_ns) != identity \
                or total != info.st_size:
            raise CodexProposerFailure(
                f"proposer panel {os.path.basename(path)!r} changed while read")
    finally:
        os.close(descriptor)
    try:
        path_after = os.lstat(path)
    except OSError as exc:
        raise CodexProposerFailure(
            f"proposer panel {os.path.basename(path)!r} changed after read") \
            from exc
    if path_after.st_nlink != 1 or (
            path_after.st_dev, path_after.st_ino, path_after.st_size,
            path_after.st_mtime_ns, path_after.st_ctime_ns) != identity:
        raise CodexProposerFailure(
            f"proposer panel {os.path.basename(path)!r} path changed while read")
    data = b"".join(chunks)
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise CodexProposerFailure(
            f"proposer panel {os.path.basename(path)!r} is not a PNG")
    return data


def _read_stable_view_file(
        path: str, *, maximum: int, description: str) -> bytes:
    """Read one private-view file without following or racing path changes."""
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise CodexProposerFailure(
            f"Codex {description} disappeared after launch") from exc
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 \
            or not 0 < before.st_size <= maximum:
        raise CodexProposerFailure(
            f"Codex {description} is not a bounded regular file")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
    except OSError as exc:
        raise CodexProposerFailure(
            f"cannot re-open Codex {description}") from exc
    try:
        opened = os.fstat(descriptor)
        identity = (
            opened.st_dev, opened.st_ino, opened.st_size,
            opened.st_mtime_ns, opened.st_ctime_ns,
        )
        if opened.st_nlink != 1 or identity != (
                before.st_dev, before.st_ino, before.st_size,
                before.st_mtime_ns, before.st_ctime_ns):
            raise CodexProposerFailure(
                f"Codex {description} changed while being re-opened")
        chunks: list[bytes] = []
        total = 0
        while True:
            block = os.read(
                descriptor, min(1_048_576, maximum + 1 - total))
            if not block:
                break
            chunks.append(block)
            total += len(block)
            if total > maximum:
                raise CodexProposerFailure(
                    f"Codex {description} became oversized")
        after = os.fstat(descriptor)
        if total != opened.st_size or after.st_nlink != 1 or (
                after.st_dev, after.st_ino, after.st_size,
                after.st_mtime_ns, after.st_ctime_ns) != identity:
            raise CodexProposerFailure(
                f"Codex {description} changed while being read")
    finally:
        os.close(descriptor)
    try:
        path_after = os.lstat(path)
    except OSError as exc:
        raise CodexProposerFailure(
            f"Codex {description} path changed after launch") from exc
    if path_after.st_nlink != 1 or (
            path_after.st_dev, path_after.st_ino, path_after.st_size,
            path_after.st_mtime_ns, path_after.st_ctime_ns) != identity:
        raise CodexProposerFailure(
            f"Codex {description} path changed after launch")
    return b"".join(chunks)


def _ordered_panel_snapshot(
        panel_png_paths: Sequence[str]) -> tuple[tuple[str, bytes], ...]:
    if isinstance(panel_png_paths, (str, bytes)) \
            or len(panel_png_paths) != len(_PANEL_NAMES):
        raise CodexProposerFailure("Codex proposer requires exactly 12 PNGs")
    image_paths = tuple(panel_png_paths)
    if tuple(os.path.basename(path) for path in image_paths) != _PANEL_NAMES:
        raise CodexProposerFailure(
            "Codex panel paths must have canonical ordered filenames")
    return tuple(
        (name, _read_regular_png(path))
        for name, path in zip(_PANEL_NAMES, image_paths)
    )


def _named_image_snapshot(
        image_png_paths: Sequence[str], image_names: Sequence[str],
        ) -> tuple[tuple[str, bytes], ...]:
    """Read a bounded ordered image view whose names carry no class labels.

    The original Bongard proposer deliberately requires ``pos_*``/``neg_*``
    filenames because it sees the labelled support set.  A prose-conditioned
    soft scorer has a different information boundary: it must see an opaque
    image (or neutral batch), never a side assignment.  Keep that boundary in
    the transport rather than relying on a prompt to make label-bearing names
    harmless.
    """
    if isinstance(image_png_paths, (str, bytes)) \
            or isinstance(image_names, (str, bytes)):
        raise CodexProposerFailure("named image inputs must be sequences")
    paths = tuple(image_png_paths)
    names = tuple(image_names)
    if not 1 <= len(paths) <= MAX_NAMED_IMAGES or len(paths) != len(names):
        raise CodexProposerFailure(
            f"named Codex input requires 1..{MAX_NAMED_IMAGES} images and "
            "one neutral name per image")
    if len(set(names)) != len(names) or any(
            not isinstance(name, str) or _IMAGE_NAME_RE.fullmatch(name) is None
            for name in names):
        raise CodexProposerFailure(
            "named Codex image names must be unique lowercase neutral PNG names")
    if any(name.startswith(("pos_", "neg_", "positive", "negative"))
           for name in names):
        raise CodexProposerFailure(
            "blind Codex image names must not encode a Bongard side")
    return tuple(
        (name, _read_regular_png(path))
        for name, path in zip(names, paths)
    )


def _panel_identities_from_snapshot(
        snapshot: Sequence[tuple[str, bytes]]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "byte_count": len(data),
            "content_digest": _bytes_digest(data),
        }
        for name, data in snapshot
    ]


def _named_image_set_digest_from_snapshot(
        snapshot: Sequence[tuple[str, bytes]]) -> str:
    return "sha256:" + _digest({
        "schema": NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
        "images": _panel_identities_from_snapshot(snapshot),
    })


def named_image_view_digest(
        image_png_paths: Sequence[str], image_names: Sequence[str]) -> str:
    """Digest the exact ordered neutral image presentation."""
    return _digest(_panel_identities_from_snapshot(
        _named_image_snapshot(image_png_paths, image_names)))


def named_image_set_digest(
        image_png_paths: Sequence[str], image_names: Sequence[str]) -> str:
    """Content-address a neutral scorer image set without inventing labels."""
    return _named_image_set_digest_from_snapshot(
        _named_image_snapshot(image_png_paths, image_names))


def ordered_panel_view_digest(panel_png_paths: Sequence[str]) -> str:
    """Digest canonical ordered panel names, byte counts, and exact bytes."""
    snapshot = _ordered_panel_snapshot(panel_png_paths)
    return _digest(_panel_identities_from_snapshot(snapshot))


def _semantic_panel_set_digest_from_snapshot(
        snapshot: Sequence[tuple[str, bytes]]) -> str:
    try:
        import numpy as np
        from PIL import Image
    except (ImportError, OSError) as exc:
        raise CodexProposerFailure(
            "cannot load canonical panel digest dependencies") from exc
    manifest: list[dict[str, Any]] = []
    try:
        for expected_name, (name, data) in zip(_PANEL_NAMES, snapshot):
            if name != expected_name:
                raise CodexProposerFailure(
                    "Codex semantic panel order is non-canonical")
            side, raw_index = name[:-4].split("_")
            with Image.open(io.BytesIO(data)) as encoded:
                if encoded.format != "PNG" \
                        or getattr(encoded, "n_frames", 1) != 1:
                    raise CodexProposerFailure(
                        f"Codex panel {name!r} is not a canonical PNG")
                width, height = encoded.size
                if width <= 0 or height <= 0 \
                        or width * height > _MAX_PANEL_ELEMENTS:
                    raise CodexProposerFailure(
                        f"Codex panel {name!r} has an invalid decoded size")
                encoded.load()
                if encoded.mode == "L":
                    presentation = np.array(
                        encoded, dtype=np.uint8, copy=True)
                elif encoded.mode == "RGB":
                    rgb = np.array(encoded, dtype=np.uint8, copy=True)
                    if rgb.ndim != 3 or rgb.shape[2] != 3 \
                            or not bool((rgb[..., 0] == rgb[..., 1]).all()) \
                            or not bool((rgb[..., 1] == rgb[..., 2]).all()):
                        raise CodexProposerFailure(
                            f"Codex panel {name!r} has non-grayscale RGB pixels")
                    presentation = rgb[..., 0]
                else:
                    raise CodexProposerFailure(
                        f"Codex panel {name!r} is not grayscale L/RGB")
            if presentation.ndim != 2:
                raise CodexProposerFailure(
                    f"Codex panel {name!r} is not a grayscale raster")
            # The released ShapeBongard_V2 PNGs are lossless grayscale RGB
            # with antialiased edge pixels.  Codex receives those exact source
            # bytes; this threshold is only the deterministic derived ink
            # witness used by the historical semantic panel-set digest.
            panel = np.ascontiguousarray(
                (presentation < SEMANTIC_INK_THRESHOLD).astype(
                    np.uint8, copy=False))
            shape = [int(panel.shape[0]), int(panel.shape[1])]
            dtype = panel.dtype.str
            content_hasher = hashlib.sha256()
            content_hasher.update(_canonical_json_bytes({
                "schema": _PANEL_SCHEMA,
                "shape": shape,
                "dtype": dtype,
            }))
            content_hasher.update(b"\0")
            content_hasher.update(panel.tobytes(order="C"))
            manifest.append({
                "side": side,
                "index": int(raw_index),
                "shape": shape,
                "dtype": dtype,
                "encoding": _PACKED_BINARY_ENCODING,
                "content_digest": "sha256:" + content_hasher.hexdigest(),
            })
        if len(manifest) != len(_PANEL_NAMES):
            raise CodexProposerFailure(
                "Codex semantic panel set is incomplete")
        return "sha256:" + _digest({
            "schema": _PANEL_SCHEMA,
            "panels": manifest,
        })
    except CodexProposerFailure:
        raise
    except Exception as exc:
        raise CodexProposerFailure(
            "cannot decode canonical Codex panel set") from exc


def semantic_panel_set_digest(panel_png_paths: Sequence[str]) -> str:
    """Digest the thresholded ink while preserving exact source-byte receipts."""
    return _semantic_panel_set_digest_from_snapshot(
        _ordered_panel_snapshot(panel_png_paths))


def _copy_panel_view(panel_png_paths: Sequence[str], view_dir: str) \
        -> tuple[tuple[str, ...], str]:
    if isinstance(panel_png_paths, (str, bytes)) \
            or len(panel_png_paths) != len(_PANEL_NAMES):
        raise CodexProposerFailure("Codex proposer requires exactly 12 PNGs")
    by_name: dict[str, str] = {}
    for path in panel_png_paths:
        name = os.path.basename(path) if isinstance(path, str) else ""
        if name in by_name or name not in _PANEL_NAMES:
            raise CodexProposerFailure(
                "Codex proposer panel filenames are incomplete or duplicated")
        by_name[name] = path
    if set(by_name) != set(_PANEL_NAMES):
        raise CodexProposerFailure(
            "Codex proposer panel filenames differ from the canonical set")
    copied: list[str] = []
    for name in _PANEL_NAMES:
        data = _read_regular_png(by_name[name])
        target = os.path.join(view_dir, name)
        descriptor = os.open(
            target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            offset = 0
            while offset < len(data):
                offset += os.write(descriptor, data[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        copied.append(target)
    copied_paths = tuple(copied)
    return copied_paths, ordered_panel_view_digest(copied_paths)


def _copy_named_image_view(
        image_png_paths: Sequence[str], image_names: Sequence[str],
        view_dir: str) -> tuple[tuple[str, ...], str, str]:
    snapshot = _named_image_snapshot(image_png_paths, image_names)
    copied: list[str] = []
    for name, data in snapshot:
        target = os.path.join(view_dir, name)
        descriptor = os.open(
            target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            offset = 0
            while offset < len(data):
                offset += os.write(descriptor, data[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        copied.append(target)
    copied_paths = tuple(copied)
    copied_snapshot = _named_image_snapshot(copied_paths, image_names)
    identities = _panel_identities_from_snapshot(copied_snapshot)
    return (
        copied_paths,
        _digest(identities),
        _named_image_set_digest_from_snapshot(copied_snapshot),
    )


def _recheck_private_view(
        view_dir: str, image_paths: Sequence[str],
        expected_panel_digest: str, schema_path: str,
        expected_schema_digest: str) -> None:
    expected_names = set(_PANEL_NAMES) | {"output_schema.json"}
    try:
        actual_names = set(os.listdir(view_dir))
    except OSError as exc:
        raise CodexProposerFailure(
            "Codex private view disappeared after launch") from exc
    if actual_names != expected_names:
        raise CodexProposerFailure(
            "Codex private view contents changed during execution")
    if ordered_panel_view_digest(image_paths) != expected_panel_digest:
        raise CodexProposerFailure(
            "Codex panel view bytes changed during execution")
    schema_bytes = _read_stable_view_file(
        schema_path, maximum=MAX_SCHEMA_UTF8_BYTES,
        description="output schema")
    if _bytes_digest(schema_bytes) != expected_schema_digest:
        raise CodexProposerFailure(
            "Codex output schema bytes changed during execution")


def _recheck_named_private_view(
        view_dir: str, image_paths: Sequence[str], image_names: Sequence[str],
        expected_view_digest: str, expected_set_digest: str,
        schema_path: str, expected_schema_digest: str) -> None:
    expected_names = set(image_names) | {"output_schema.json"}
    try:
        actual_names = set(os.listdir(view_dir))
    except OSError as exc:
        raise CodexProposerFailure(
            "Codex named-image private view disappeared after launch") from exc
    if actual_names != expected_names:
        raise CodexProposerFailure(
            "Codex named-image private view contents changed during execution")
    snapshot = _named_image_snapshot(image_paths, image_names)
    identities = _panel_identities_from_snapshot(snapshot)
    if _digest(identities) != expected_view_digest \
            or _named_image_set_digest_from_snapshot(snapshot) != \
            expected_set_digest:
        raise CodexProposerFailure(
            "Codex named-image view bytes changed during execution")
    schema_bytes = _read_stable_view_file(
        schema_path, maximum=MAX_SCHEMA_UTF8_BYTES,
        description="output schema")
    if _bytes_digest(schema_bytes) != expected_schema_digest:
        raise CodexProposerFailure(
            "Codex output schema bytes changed during execution")


def _codex_launcher_identity(executable: str) -> tuple[str, tuple[Any, ...]]:
    resolved = shutil.which(executable, path=os.environ.get("PATH"))
    if not resolved:
        raise CodexProposerFailure("Codex CLI executable is not on PATH")
    resolved = os.path.realpath(resolved)
    try:
        before = os.lstat(resolved)
    except OSError as exc:
        raise CodexProposerFailure("cannot inspect Codex CLI executable") from exc
    if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= \
            MAX_CODEX_LAUNCHER_BYTES:
        raise CodexProposerFailure("Codex CLI launcher is not a bounded file")
    descriptor = os.open(
        resolved,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        opened = os.fstat(descriptor)
        identity = (
            opened.st_dev, opened.st_ino, opened.st_size,
            opened.st_mtime_ns, opened.st_ctime_ns,
        )
        if identity != (
                before.st_dev, before.st_ino, before.st_size,
                before.st_mtime_ns, before.st_ctime_ns):
            raise CodexProposerFailure("Codex CLI launcher changed during read")
        hasher = hashlib.sha256()
        total = 0
        while True:
            block = os.read(
                descriptor,
                min(1_048_576, MAX_CODEX_LAUNCHER_BYTES + 1 - total),
            )
            if not block:
                break
            hasher.update(block)
            total += len(block)
            if total > MAX_CODEX_LAUNCHER_BYTES:
                raise CodexProposerFailure("Codex CLI launcher is oversized")
        after = os.fstat(descriptor)
        if total != opened.st_size or (
                after.st_dev, after.st_ino, after.st_size,
                after.st_mtime_ns, after.st_ctime_ns) != identity:
            raise CodexProposerFailure("Codex CLI launcher changed during read")
    finally:
        os.close(descriptor)
    return resolved, (*identity, hasher.hexdigest())


def _codex_cli_version(
        executable: str, *, temp_parent: str | None = None) -> str:
    try:
        with tempfile.TemporaryFile() as stdout_file, \
                tempfile.TemporaryFile() as stderr_file:
            proc = subprocess.run(
                [executable, "--version"],
                stdout=stdout_file,
                stderr=stderr_file,
                timeout=10,
                env=_minimal_environment(temp_parent=temp_parent),
                check=False,
            )
            stdout_file.seek(0, os.SEEK_END)
            stdout_size = stdout_file.tell()
            stderr_file.seek(0, os.SEEK_END)
            stderr_size = stderr_file.tell()
            if stdout_size > 10_000 or stderr_size > 100_000:
                raise CodexProposerFailure(
                    "Codex CLI version output is oversized")
            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout = stdout_file.read(10_001)
            stderr = stderr_file.read(100_001)
    except (OSError, subprocess.SubprocessError) as exc:
        raise CodexProposerFailure(
            f"cannot fingerprint Codex CLI: {exc}") from exc
    try:
        version = (stdout or stderr).decode("utf-8", errors="strict").strip()
    except UnicodeError as exc:
        raise CodexProposerFailure("Codex CLI version is not UTF-8") from exc
    if proc.returncode != 0 or not version:
        raise CodexProposerFailure("cannot fingerprint Codex CLI")
    # macOS sandbox warnings belong to stderr; prefer the clean stdout line.
    return version.splitlines()[-1]


def codex_cli_version(executable: str = "codex") -> str:
    """Return the exact installed CLI version used in policy fingerprints."""
    resolved, _identity = _codex_launcher_identity(executable)
    return _codex_cli_version(resolved, temp_parent=_safe_temp_parent())


def codex_cli_fingerprint(executable: str = "codex") -> dict[str, str]:
    """Fingerprint the resolved launcher bytes and its reported version."""
    resolved, identity = _codex_launcher_identity(executable)
    temp_parent = _safe_temp_parent()
    return {
        "version": _codex_cli_version(
            resolved, temp_parent=temp_parent),
        "launcher_digest": identity[-1],
    }


def codex_cli_authenticated_fingerprint(
        executable: str = "codex", *,
        expected_launcher_digest: str) -> dict[str, str]:
    """Authenticate launcher bytes before executing even ``--version``.

    The ordinary fingerprint helper is appropriate for inventory.  A causal
    benchmark boundary already has an external byte commitment, so it must
    compare the read-only launcher hash before allowing those bytes to run.
    The second identity read closes changes made by the version invocation.
    """

    if not isinstance(expected_launcher_digest, str) or re.fullmatch(
            r"[0-9a-f]{64}", expected_launcher_digest) is None:
        raise CodexProposerFailure(
            "expected Codex launcher digest must be 64 lowercase hex digits")
    resolved, identity = _codex_launcher_identity(executable)
    if identity[-1] != expected_launcher_digest:
        raise CodexProposerFailure(
            "Codex launcher bytes differ from the external commitment")
    temp_parent = _safe_temp_parent()
    version = _codex_cli_version(resolved, temp_parent=temp_parent)
    resolved_after, identity_after = _codex_launcher_identity(resolved)
    if resolved_after != resolved or identity_after != identity:
        raise CodexProposerFailure(
            "Codex launcher changed during authenticated fingerprinting")
    return {
        "version": version,
        "launcher_digest": identity[-1],
    }


def _codex_command(
        *, executable: str, view_dir: str, image_paths: Sequence[str],
        schema_path: str, model: str, reasoning_effort: str) -> list[str]:
    command = [
        executable,
        "--ask-for-approval", "never",
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--strict-config",
        "--skip-git-repo-check",
        "--sandbox", "read-only",
        "--model", model,
        "--config", f'model_reasoning_effort="{reasoning_effort}"',
        "--config", 'web_search="disabled"',
        "--config", "agents.enabled=false",
    ]
    for feature in _DISABLED_FEATURES:
        command.extend(("--disable", feature))
    for path in image_paths:
        command.extend(("--image", path))
    command.extend((
        "--json",
        "--color", "never",
        "--output-schema", schema_path,
        "--cd", view_dir,
        "-",
    ))
    return command


def validate_codex_strict_output_schema(schema: Mapping[str, Any]) -> None:
    """Reject schema features outside the frozen Responses strict subset."""

    if not isinstance(schema, Mapping):
        raise CodexProposerFailure("strict output schema must be an object")
    stack: list[object] = [schema]
    while stack:
        node = stack.pop()
        if isinstance(node, Mapping):
            forbidden = _STRICT_SCHEMA_FORBIDDEN_KEYWORDS.intersection(node)
            if forbidden:
                raise CodexProposerFailure(
                    "strict output schema uses unsupported keywords: "
                    + ", ".join(sorted(forbidden))
                )
            if node.get("type") == "object":
                properties = node.get("properties")
                required = node.get("required")
                if (
                    not isinstance(properties, Mapping)
                    or node.get("additionalProperties") is not False
                    or not isinstance(required, list)
                    or set(required) != set(properties)
                ):
                    raise CodexProposerFailure(
                        "strict object schemas must require every declared field "
                        "and forbid additional properties"
                    )
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)


def _wait_for_process(process: subprocess.Popen, timeout: float) -> bool:
    try:
        process.wait(timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        return False
    except OSError:
        return process.poll() is not None


def _terminate_process_group(process: subprocess.Popen) -> None:
    """Terminate, then kill, the isolated session created for one turn."""
    group_signalled = False
    if os.name == "posix" and hasattr(os, "killpg"):
        try:
            os.killpg(process.pid, signal.SIGTERM)
            group_signalled = True
        except ProcessLookupError:
            # The group is already gone; still reap the leader below.
            group_signalled = True
        except OSError:
            group_signalled = False
    if not group_signalled:
        try:
            process.terminate()
        except (OSError, ProcessLookupError):
            pass

    _wait_for_process(process, PROCESS_GROUP_GRACE_SECONDS)

    # Even if the group leader exited on TERM, descendants may remain.  A
    # second group signal is safe because start_new_session gives this turn a
    # unique process group.
    killed_group = False
    if os.name == "posix" and hasattr(os, "killpg"):
        try:
            os.killpg(process.pid, signal.SIGKILL)
            killed_group = True
        except ProcessLookupError:
            killed_group = True
        except OSError:
            killed_group = False
    if not killed_group and process.poll() is None:
        try:
            process.kill()
        except (OSError, ProcessLookupError):
            pass
    _wait_for_process(process, PROCESS_GROUP_GRACE_SECONDS)


def _run_codex_process(
        command: Sequence[str], *, task_bytes: bytes, view_dir: str,
        environment: Mapping[str, str], minutes: int) \
        -> tuple[int, bytes, bytes]:
    """Launch one bounded process with timeout cleanup for all descendants."""
    try:
        with tempfile.TemporaryFile(dir=view_dir) as stdout_file, \
                tempfile.TemporaryFile(dir=view_dir) as stderr_file:
            process = subprocess.Popen(
                list(command),
                stdin=subprocess.PIPE,
                stdout=stdout_file,
                stderr=stderr_file,
                cwd=view_dir,
                env=dict(environment),
                start_new_session=True,
            )
            try:
                process.communicate(
                    input=task_bytes, timeout=minutes * 60)
            except subprocess.TimeoutExpired as exc:
                _terminate_process_group(process)
                raise CodexProposerFailure(
                    f"Codex timed out after {minutes} min") from exc
            except OSError as exc:
                _terminate_process_group(process)
                raise CodexProposerFailure(
                    f"Codex process communication failed: {exc}") from exc
            except BaseException:
                _terminate_process_group(process)
                raise

            stdout_file.seek(0, os.SEEK_END)
            stdout_size = stdout_file.tell()
            stderr_file.seek(0, os.SEEK_END)
            stderr_size = stderr_file.tell()
            if stdout_size > MAX_STDOUT_BYTES:
                raise CodexProposerFailure(
                    "Codex JSONL event stream is oversized")
            if stderr_size > MAX_STDERR_BYTES:
                raise CodexProposerFailure(
                    "Codex diagnostic output is oversized")
            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout = stdout_file.read(MAX_STDOUT_BYTES + 1)
            stderr = stderr_file.read(MAX_STDERR_BYTES + 1)
            returncode = process.returncode
            if isinstance(returncode, bool) or not isinstance(returncode, int):
                raise CodexProposerFailure(
                    "Codex process ended without an exit status")
            return returncode, stdout, stderr
    except CodexProposerFailure:
        raise
    except OSError as exc:
        raise CodexProposerFailure(f"Codex could not be launched: {exc}") \
            from exc


def _reported_models(event: Mapping[str, Any]) -> set[str]:
    models: set[str] = set()
    if "model" in event:
        direct = event["model"]
        if not isinstance(direct, str) or not direct:
            raise CodexProposerFailure(
                "Codex JSONL contains malformed model identity evidence")
        models.add(direct)
    for key in ("response", "item", "usage"):
        nested = event.get(key)
        if isinstance(nested, Mapping) and "model" in nested:
            model = nested["model"]
            if not isinstance(model, str) or not model:
                raise CodexProposerFailure(
                    "Codex JSONL contains malformed model identity evidence")
            models.add(model)
    return models


def _raw_utf8_digest(value: str) -> str:
    try:
        return _bytes_digest(value.encode("utf-8", errors="strict"))
    except UnicodeError as exc:
        raise CodexProposerFailure(
            "cannot digest non-UTF-8 Codex text") from exc


def _structured_payload_digest(payload: Mapping[str, Any]) -> str:
    encoded = _canonical_json_bytes(dict(payload))
    if len(encoded) > MAX_STRUCTURED_OUTPUT_CANONICAL_BYTES:
        raise CodexProposerFailure(
            "Codex structured output canonical form is oversized")
    return _bytes_digest(encoded)


def _causal_input_metadata(
        executed_prompt: str,
        image_paths: Sequence[str],
        output_schema_digest: str,
        expected_panel_view_digest: str) -> dict[str, str]:
    snapshot = _ordered_panel_snapshot(image_paths)
    identities = _panel_identities_from_snapshot(snapshot)
    observed_panel_view_digest = _digest(identities)
    if observed_panel_view_digest != expected_panel_view_digest:
        raise CodexProposerFailure(
            "Codex panel view changed before input binding")
    panel_set = _semantic_panel_set_digest_from_snapshot(snapshot)
    prompt_digest = _raw_utf8_digest(executed_prompt)
    envelope: dict[str, Any] = {
        "schema": STRUCTURED_INPUT_DIGEST_SCHEMA,
        "task": executed_prompt,
        "ordered_panel_identities": identities,
        "panel_view_digest": observed_panel_view_digest,
        "panel_set_digest": panel_set,
        "prompt_digest": prompt_digest,
        "output_schema_digest": output_schema_digest,
    }
    return {
        "task_digest": prompt_digest,
        "current_source_digest": "",
        "current_log_digest": "",
        "prompt_digest": prompt_digest,
        "input_digest_schema": STRUCTURED_INPUT_DIGEST_SCHEMA,
        "input_digest": _digest(envelope),
        "panel_view_digest": observed_panel_view_digest,
        "panel_set_digest": panel_set,
    }


def _causal_named_image_input_metadata(
        executed_prompt: str, image_paths: Sequence[str],
        image_names: Sequence[str], output_schema_digest: str,
        expected_view_digest: str, expected_set_digest: str,
        ) -> dict[str, str]:
    snapshot = _named_image_snapshot(image_paths, image_names)
    identities = _panel_identities_from_snapshot(snapshot)
    observed_view_digest = _digest(identities)
    observed_set_digest = _named_image_set_digest_from_snapshot(snapshot)
    if observed_view_digest != expected_view_digest \
            or observed_set_digest != expected_set_digest:
        raise CodexProposerFailure(
            "Codex named-image view changed before input binding")
    prompt_digest = _raw_utf8_digest(executed_prompt)
    envelope = {
        "schema": NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
        "task": executed_prompt,
        "ordered_image_identities": identities,
        "image_view_digest": observed_view_digest,
        "image_set_digest": observed_set_digest,
        "prompt_digest": prompt_digest,
        "output_schema_digest": output_schema_digest,
    }
    return {
        "task_digest": prompt_digest,
        "current_source_digest": "",
        "current_log_digest": "",
        "prompt_digest": prompt_digest,
        "input_digest_schema": NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
        "input_digest": _digest(envelope),
        # Keep the receipt's stable field names.  Under the named-image schema
        # these bind a neutral image view/set, not Bongard label semantics.
        "panel_view_digest": observed_view_digest,
        "panel_set_digest": observed_set_digest,
    }


def _parse_jsonl(
        stdout: bytes, *, requested_model: str, reasoning_effort: str,
        cli_version: str, cli_launcher_digest: str,
        cloud_config_bundle_cache_binding: str,
        output_schema_digest: str,
        causal_input: Mapping[str, str]) \
        -> tuple[dict[str, Any], CodexReceipt]:
    if not stdout or len(stdout) > MAX_STDOUT_BYTES:
        raise CodexProposerFailure(
            "Codex returned an empty or oversized JSONL event stream")
    try:
        text = stdout.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise CodexProposerFailure("Codex JSONL is not UTF-8") from exc
    lines = text.splitlines()
    if not lines or len(lines) > MAX_JSONL_EVENTS or any(not line for line in lines):
        raise CodexProposerFailure("Codex JSONL line framing is invalid")
    events: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        event = _strict_json(line, f"JSONL event {index}")
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            raise CodexProposerFailure(
                f"Codex JSONL event {index} lacks an event type")
        events.append(event)

    event_types = tuple(event["type"] for event in events)
    if event_types[0] != "thread.started" \
            or event_types[-1] != "turn.completed" \
            or event_types.count("thread.started") != 1 \
            or event_types.count("turn.started") != 1 \
            or event_types.count("turn.completed") != 1:
        raise CodexProposerFailure("Codex JSONL lifecycle is incomplete or reordered")
    if any(kind in {"error", "turn.failed"} for kind in event_types):
        raise CodexProposerFailure("Codex reported a failed turn")
    allowed_events = {
        "thread.started", "turn.started", "turn.completed",
        "item.started", "item.updated", "item.completed",
    }
    if any(kind not in allowed_events for kind in event_types):
        raise CodexProposerFailure("Codex emitted an unsupported event type")

    thread_id = events[0].get("thread_id")
    try:
        parsed_uuid = uuid.UUID(thread_id) if isinstance(thread_id, str) else None
    except (ValueError, AttributeError) as exc:
        raise CodexProposerFailure("Codex thread ID is malformed") from exc
    if parsed_uuid is None or str(parsed_uuid) != thread_id:
        raise CodexProposerFailure("Codex thread ID is not canonical UUID text")

    messages: list[str] = []
    reported_models: set[str] = set()
    item_types: list[str] = []
    seen_turn_started = False
    last_item: tuple[str, str] | None = None
    for index, event in enumerate(events):
        kind = event["type"]
        reported_models.update(_reported_models(event))
        if kind == "turn.started":
            seen_turn_started = True
        if kind.startswith("item."):
            item = event.get("item")
            if not seen_turn_started or not isinstance(item, Mapping) \
                    or item.get("type") not in _ALLOWED_ITEM_TYPES:
                raise CodexProposerFailure(
                    f"Codex emitted a forbidden or malformed tool item at {index}")
            item_types.append(item["type"])
            last_item = (kind, item["type"])
            if item.get("type") == "agent_message" \
                    and kind != "item.completed":
                raise CodexProposerFailure(
                    "Codex agent messages must be completed final items")
            if item.get("type") == "agent_message" \
                    and kind == "item.completed":
                message = item.get("text")
                if not isinstance(message, str):
                    raise CodexProposerFailure(
                        "Codex completed agent message has no text")
                messages.append(message)
    if not messages:
        raise CodexProposerFailure(
            "Codex must emit a completed final agent message")
    if last_item != ("item.completed", "agent_message"):
        raise CodexProposerFailure(
            "Codex final item must be the completed structured message")
    if reported_models and reported_models != {requested_model}:
        raise CodexProposerFailure(
            "Codex JSONL reported a model different from the requested model")

    completion = events[-1]
    usage = completion.get("usage")
    usage_keys = (
        "input_tokens", "cached_input_tokens", "output_tokens",
        "reasoning_output_tokens",
    )
    if not isinstance(usage, Mapping):
        raise CodexProposerFailure("Codex completion lacks token usage")
    counters: dict[str, int] = {}
    for key in usage_keys:
        value = usage.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CodexProposerFailure(
                f"Codex completion has invalid {key}")
        counters[key] = value
    if counters["input_tokens"] + counters["output_tokens"] <= 0 \
            or counters["cached_input_tokens"] > counters["input_tokens"]:
        raise CodexProposerFailure("Codex completion has inconsistent token usage")

    # Current Codex CLI versions may emit one or more completed progress
    # messages before the schema-constrained final message.  They remain
    # tool-free and are bound by event_stream_digest; only the final item is
    # interpreted as the structured result.
    payload = _strict_json(messages[-1], "structured final message")
    if not isinstance(payload, dict):
        raise CodexProposerFailure(
            "Codex structured final message must be an object")
    structured_output_digest = _structured_payload_digest(payload)

    body: dict[str, Any] = {
        "schema": CODEX_RECEIPT_SCHEMA,
        "source": "codex-cli",
        "requested_model": requested_model,
        "reported_model": requested_model if reported_models else "",
        "model_identity_evidence": (
            "jsonl-reported-model"
            if reported_models else "explicit-cli-model-flag;jsonl-omits-model"),
        "requested_reasoning_effort": reasoning_effort,
        **counters,
        "thread_id": thread_id,
        "codex_cli_version": cli_version,
        "codex_launcher_digest": cli_launcher_digest,
        "cloud_config_bundle_cache_binding": (
            cloud_config_bundle_cache_binding
        ),
        "task_digest": causal_input.get("task_digest"),
        "current_source_digest": causal_input.get("current_source_digest"),
        "current_log_digest": causal_input.get("current_log_digest"),
        "prompt_digest": causal_input.get("prompt_digest"),
        "input_digest_schema": causal_input.get("input_digest_schema"),
        "input_digest": causal_input.get("input_digest"),
        "output_schema_digest": output_schema_digest,
        "panel_view_digest": causal_input.get("panel_view_digest"),
        "panel_set_digest": causal_input.get("panel_set_digest"),
        "structured_output_digest": structured_output_digest,
        "proposed_source_digest": "",
        "proposed_log_digest": "",
        "event_stream_digest": _bytes_digest(stdout),
        "event_types": list(event_types),
        "item_types": item_types,
        "isolation_policy": CODEX_ISOLATION_POLICY,
        "outcome": "success",
    }
    body["receipt_digest"] = _digest(body)
    validate_codex_receipt(body)
    return payload, CodexReceipt(
        **{
            **body,
            "event_types": tuple(body["event_types"]),
            "item_types": tuple(body["item_types"]),
        })


def validate_codex_receipt(receipt: Mapping[str, Any]) -> None:
    if not isinstance(receipt, Mapping) or set(receipt) != _RECEIPT_KEYS:
        raise CodexProposerFailure("Codex receipt fields are invalid")
    if receipt["schema"] != CODEX_RECEIPT_SCHEMA \
            or receipt["source"] != "codex-cli" \
            or receipt["isolation_policy"] != CODEX_ISOLATION_POLICY \
            or receipt["outcome"] != "success":
        raise CodexProposerFailure("Codex receipt policy identity differs")
    _validate_model(receipt["requested_model"])
    _validate_reasoning_effort(receipt["requested_reasoning_effort"])
    reported = receipt["reported_model"]
    evidence = receipt["model_identity_evidence"]
    if evidence == "jsonl-reported-model":
        if reported != receipt["requested_model"]:
            raise CodexProposerFailure("Codex reported model differs")
    elif evidence == "explicit-cli-model-flag;jsonl-omits-model":
        if reported != "":
            raise CodexProposerFailure(
                "Codex unreported-model receipt claims a reported model")
    else:
        raise CodexProposerFailure("Codex model identity evidence is unknown")
    for key in (
            "input_tokens", "cached_input_tokens", "output_tokens",
            "reasoning_output_tokens"):
        value = receipt[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CodexProposerFailure(f"Codex receipt {key} is invalid")
    if receipt["input_tokens"] + receipt["output_tokens"] <= 0 \
            or receipt["cached_input_tokens"] > receipt["input_tokens"]:
        raise CodexProposerFailure("Codex receipt token usage is inconsistent")
    try:
        parsed_uuid = uuid.UUID(receipt["thread_id"])
    except (ValueError, AttributeError, TypeError) as exc:
        raise CodexProposerFailure("Codex receipt thread ID is malformed") from exc
    if str(parsed_uuid) != receipt["thread_id"]:
        raise CodexProposerFailure("Codex receipt thread ID is non-canonical")
    if not isinstance(receipt["codex_cli_version"], str) \
            or not receipt["codex_cli_version"]:
        raise CodexProposerFailure("Codex receipt CLI version is missing")
    policy_cache_binding = receipt["cloud_config_bundle_cache_binding"]
    if policy_cache_binding != "absent" and (
        not isinstance(policy_cache_binding, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", policy_cache_binding) is None
    ):
        raise CodexProposerFailure(
            "Codex receipt cloud policy cache binding is invalid"
        )
    input_digest_schema = receipt["input_digest_schema"]
    if not isinstance(input_digest_schema, str) \
            or input_digest_schema not in {
            STRUCTURED_INPUT_DIGEST_SCHEMA, NAMED_IMAGE_INPUT_DIGEST_SCHEMA}:
        raise CodexProposerFailure(
            "Codex receipt input digest schema is unknown")
    for key in (
            "codex_launcher_digest", "task_digest", "prompt_digest",
            "input_digest", "output_schema_digest", "panel_view_digest",
            "structured_output_digest", "event_stream_digest",
            "receipt_digest"):
        value = receipt[key]
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) \
                is None:
            raise CodexProposerFailure(f"Codex receipt {key} is not SHA-256")
    panel_set = receipt["panel_set_digest"]
    if not isinstance(panel_set, str) \
            or re.fullmatch(r"sha256:[0-9a-f]{64}", panel_set) is None:
        raise CodexProposerFailure(
            "Codex receipt panel_set_digest is not semantic SHA-256")
    chain_keys = (
        "current_source_digest", "current_log_digest",
        "proposed_source_digest", "proposed_log_digest",
    )
    if any(receipt[key] != "" for key in chain_keys):
        raise CodexProposerFailure(
            "Codex transport receipt claims legacy predicate digest fields")
    if receipt["task_digest"] != receipt["prompt_digest"]:
        raise CodexProposerFailure(
            "Codex task and prompt digests differ")
    event_types = receipt["event_types"]
    item_types = receipt["item_types"]
    if not isinstance(event_types, list) or not event_types \
            or any(not isinstance(item, str) or not item for item in event_types) \
            or event_types[0] != "thread.started" \
            or event_types[-1] != "turn.completed" \
            or event_types.count("thread.started") != 1 \
            or event_types.count("turn.started") != 1 \
            or event_types.count("turn.completed") != 1 \
            or any(item not in {
                "thread.started", "turn.started", "turn.completed",
                "item.started", "item.updated", "item.completed",
            } for item in event_types):
        raise CodexProposerFailure("Codex receipt event summary is malformed")
    if not isinstance(item_types, list) \
            or any(item not in _ALLOWED_ITEM_TYPES for item in item_types) \
            or len(item_types) != sum(
                item.startswith("item.") for item in event_types):
        raise CodexProposerFailure("Codex receipt item summary is malformed")
    item_iter = iter(item_types)
    completed_messages = 0
    turn_started = False
    last_item: tuple[str, str] | None = None
    for event_type in event_types:
        if event_type == "turn.started":
            turn_started = True
        elif event_type.startswith("item."):
            item_type = next(item_iter)
            last_item = (event_type, item_type)
            if not turn_started:
                raise CodexProposerFailure(
                    "Codex receipt item precedes its turn")
            if item_type == "agent_message" \
                    and event_type != "item.completed":
                raise CodexProposerFailure(
                    "Codex receipt has a partial agent message")
            if event_type == "item.completed" \
                    and item_type == "agent_message":
                completed_messages += 1
    if completed_messages < 1:
        raise CodexProposerFailure(
            "Codex receipt must summarize a completed agent message")
    if last_item != ("item.completed", "agent_message"):
        raise CodexProposerFailure(
            "Codex receipt final item is not its completed message")
    body = {key: value for key, value in receipt.items()
            if key != "receipt_digest"}
    if receipt["receipt_digest"] != _digest(body):
        raise CodexProposerFailure("Codex receipt digest does not reproduce")


def run_codex_structured(
        task: str,
        panel_png_paths: Sequence[str],
        output_schema: Mapping[str, Any],
        model: str = DEFAULT_CODEX_MODEL,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        minutes: int = 15,
        verbose: bool = False,
        executable: str = "codex",
        cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None = None,
        expected_launcher_digest: str | None = None,
        ) -> CodexStructuredResult:
    """Run one schema-constrained, tool-free Codex turn."""
    task = _bounded_utf8(task, "task", MAX_TASK_UTF8_BYTES)
    model = _validate_model(model)
    reasoning_effort = _validate_reasoning_effort(reasoning_effort)
    if isinstance(minutes, bool) or not isinstance(minutes, int) \
            or not 1 <= minutes <= 120:
        raise CodexProposerFailure("Codex timeout minutes must be in [1, 120]")
    if not isinstance(output_schema, Mapping):
        raise CodexProposerFailure("Codex output schema must be a mapping")
    schema_bytes = _canonical_json_bytes(dict(output_schema))
    if not schema_bytes or len(schema_bytes) > MAX_SCHEMA_UTF8_BYTES:
        raise CodexProposerFailure("Codex output schema is empty or oversized")
    schema_digest = _bytes_digest(schema_bytes)
    temp_parent = _safe_temp_parent()
    resolved_executable, launcher_identity = _codex_launcher_identity(executable)
    launcher_digest = launcher_identity[-1]
    if expected_launcher_digest is not None:
        if not isinstance(expected_launcher_digest, str) or re.fullmatch(
                r"[0-9a-f]{64}", expected_launcher_digest) is None:
            raise CodexProposerFailure(
                "expected Codex launcher digest must be 64 lowercase hex digits")
        if launcher_digest != expected_launcher_digest:
            raise CodexProposerFailure(
                "Codex launcher bytes differ from the external commitment")
    cli_version = _codex_cli_version(
        resolved_executable, temp_parent=temp_parent)

    with tempfile.TemporaryDirectory(
            prefix="bongard-codex-auth-", dir=temp_parent) as auth_dir, \
            tempfile.TemporaryDirectory(
                prefix="bongard-codex-", dir=temp_parent) as view_dir:
        _require_outside_bongard(auth_dir, "Codex auth home")
        _require_outside_bongard(view_dir, "Codex proposer view")
        _stage_codex_auth(auth_dir)
        policy_cache = _stage_cloud_policy_cache(
            auth_dir, cloud_policy_cache_snapshot
        )
        os.chmod(view_dir, 0o700)
        image_paths, panel_view_digest = _copy_panel_view(
            panel_png_paths, view_dir)
        schema_path = os.path.join(view_dir, "output_schema.json")
        descriptor = os.open(
            schema_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            offset = 0
            while offset < len(schema_bytes):
                offset += os.write(descriptor, schema_bytes[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        causal_input = _causal_input_metadata(
            task, image_paths, schema_digest, panel_view_digest)
        command = _codex_command(
            executable=resolved_executable,
            view_dir=view_dir,
            image_paths=image_paths,
            schema_path=schema_path,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        if verbose:
            print(
                f"invoking isolated {model} Codex proposer "
                f"({reasoning_effort}, up to {minutes} min)",
                flush=True,
            )
        _recheck_staged_cloud_policy_cache(policy_cache)
        returncode, stdout, stderr = _run_codex_process(
            command,
            task_bytes=task.encode("utf-8"),
            view_dir=view_dir,
            environment=_minimal_environment(
                codex_home=auth_dir, temp_parent=temp_parent),
            minutes=minutes,
        )
        _recheck_staged_cloud_policy_cache(policy_cache)
        _recheck_private_view(
            view_dir, image_paths, panel_view_digest,
            schema_path, schema_digest)
        if returncode != 0:
            try:
                stderr_detail = stderr.decode(
                    "utf-8", errors="strict").strip()[-800:]
            except UnicodeError:
                stderr_detail = "non-UTF-8 diagnostic output"
            try:
                stdout_detail = stdout.decode(
                    "utf-8", errors="strict").strip()[-1600:]
            except UnicodeError:
                stdout_detail = "non-UTF-8 JSONL diagnostic output"
            detail = " | ".join(
                part for part in (stderr_detail, stdout_detail) if part)
            raise CodexProposerFailure(
                f"Codex exited {returncode}: {detail or 'no diagnostic'}")
        resolved_after, launcher_identity_after = _codex_launcher_identity(
            resolved_executable)
        if resolved_after != resolved_executable \
                or launcher_identity_after != launcher_identity:
            raise CodexProposerFailure(
                "Codex CLI launcher changed during proposer execution")
        cli_version_after = _codex_cli_version(
            resolved_after, temp_parent=temp_parent)
        if cli_version_after != cli_version:
            raise CodexProposerFailure(
                "Codex CLI version changed during proposer execution")
        payload, receipt = _parse_jsonl(
            stdout,
            requested_model=model,
            reasoning_effort=reasoning_effort,
            cli_version=cli_version,
            cli_launcher_digest=launcher_digest,
            cloud_config_bundle_cache_binding=policy_cache.binding,
            output_schema_digest=schema_digest,
            causal_input=causal_input,
        )
        return CodexStructuredResult(payload=payload, receipt=receipt)


def run_codex_named_images_structured(
        task: str,
        image_png_paths: Sequence[str],
        image_names: Sequence[str],
        output_schema: Mapping[str, Any],
        model: str = DEFAULT_CODEX_MODEL,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        minutes: int = 15,
        verbose: bool = False,
        executable: str = "codex",
        cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None = None,
        expected_launcher_digest: str | None = None,
        ) -> CodexStructuredResult:
    """Run a schema-only turn over neutral, caller-declared image names.

    This is the transport for blind prose-conditioned scoring.  Unlike the
    labelled support proposer, it refuses ``pos_*``/``neg_*`` presentation
    names and binds the neutral names and exact bytes into a distinct receipt
    schema.  The model still has no tools, repository, network, or writable
    experiment state.
    """
    task = _bounded_utf8(task, "task", MAX_TASK_UTF8_BYTES)
    model = _validate_model(model)
    reasoning_effort = _validate_reasoning_effort(reasoning_effort)
    if isinstance(minutes, bool) or not isinstance(minutes, int) \
            or not 1 <= minutes <= 120:
        raise CodexProposerFailure("Codex timeout minutes must be in [1, 120]")
    if not isinstance(output_schema, Mapping):
        raise CodexProposerFailure("Codex output schema must be a mapping")
    validate_codex_strict_output_schema(output_schema)
    # Validate before creating any private state.
    _named_image_snapshot(image_png_paths, image_names)
    schema_bytes = _canonical_json_bytes(dict(output_schema))
    if not schema_bytes or len(schema_bytes) > MAX_SCHEMA_UTF8_BYTES:
        raise CodexProposerFailure("Codex output schema is empty or oversized")
    schema_digest = _bytes_digest(schema_bytes)
    temp_parent = _safe_temp_parent()
    resolved_executable, launcher_identity = _codex_launcher_identity(executable)
    launcher_digest = launcher_identity[-1]
    if expected_launcher_digest is not None:
        if not isinstance(expected_launcher_digest, str) or re.fullmatch(
                r"[0-9a-f]{64}", expected_launcher_digest) is None:
            raise CodexProposerFailure(
                "expected Codex launcher digest must be 64 lowercase hex digits")
        if launcher_digest != expected_launcher_digest:
            raise CodexProposerFailure(
                "Codex launcher bytes differ from the external commitment")
    cli_version = _codex_cli_version(
        resolved_executable, temp_parent=temp_parent)

    with tempfile.TemporaryDirectory(
            prefix="bongard-codex-auth-", dir=temp_parent) as auth_dir, \
            tempfile.TemporaryDirectory(
                prefix="bongard-codex-blind-", dir=temp_parent) as view_dir:
        _require_outside_bongard(auth_dir, "Codex auth home")
        _require_outside_bongard(view_dir, "Codex named-image view")
        _stage_codex_auth(auth_dir)
        policy_cache = _stage_cloud_policy_cache(
            auth_dir, cloud_policy_cache_snapshot
        )
        os.chmod(view_dir, 0o700)
        staged_paths, view_digest, set_digest = _copy_named_image_view(
            image_png_paths, image_names, view_dir)
        schema_path = os.path.join(view_dir, "output_schema.json")
        descriptor = os.open(
            schema_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            offset = 0
            while offset < len(schema_bytes):
                offset += os.write(descriptor, schema_bytes[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        causal_input = _causal_named_image_input_metadata(
            task, staged_paths, image_names, schema_digest,
            view_digest, set_digest)
        command = _codex_command(
            executable=resolved_executable,
            view_dir=view_dir,
            image_paths=staged_paths,
            schema_path=schema_path,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        if verbose:
            print(
                f"invoking isolated {model} Codex blind image scorer "
                f"({reasoning_effort}, up to {minutes} min)",
                flush=True,
            )
        _recheck_staged_cloud_policy_cache(policy_cache)
        returncode, stdout, stderr = _run_codex_process(
            command,
            task_bytes=task.encode("utf-8"),
            view_dir=view_dir,
            environment=_minimal_environment(
                codex_home=auth_dir, temp_parent=temp_parent),
            minutes=minutes,
        )
        _recheck_staged_cloud_policy_cache(policy_cache)
        _recheck_named_private_view(
            view_dir, staged_paths, image_names, view_digest, set_digest,
            schema_path, schema_digest)
        if returncode != 0:
            try:
                stderr_detail = stderr.decode(
                    "utf-8", errors="strict").strip()[-800:]
            except UnicodeError:
                stderr_detail = "non-UTF-8 diagnostic output"
            try:
                stdout_detail = stdout.decode(
                    "utf-8", errors="strict").strip()[-1600:]
            except UnicodeError:
                stdout_detail = "non-UTF-8 JSONL diagnostic output"
            detail = " | ".join(
                part for part in (stderr_detail, stdout_detail) if part)
            raise CodexProposerFailure(
                f"Codex exited {returncode}: {detail or 'no diagnostic'}")
        resolved_after, launcher_identity_after = _codex_launcher_identity(
            resolved_executable)
        if resolved_after != resolved_executable \
                or launcher_identity_after != launcher_identity:
            raise CodexProposerFailure(
                "Codex CLI launcher changed during blind scoring")
        cli_version_after = _codex_cli_version(
            resolved_after, temp_parent=temp_parent)
        if cli_version_after != cli_version:
            raise CodexProposerFailure(
                "Codex CLI version changed during blind scoring")
        payload, receipt = _parse_jsonl(
            stdout,
            requested_model=model,
            reasoning_effort=reasoning_effort,
            cli_version=cli_version,
            cli_launcher_digest=launcher_digest,
            cloud_config_bundle_cache_binding=policy_cache.binding,
            output_schema_digest=schema_digest,
            causal_input=causal_input,
        )
        return CodexStructuredResult(payload=payload, receipt=receipt)


__all__ = [
    "CODEX_ISOLATION_POLICY",
    "CODEX_RECEIPT_SCHEMA",
    "DEFAULT_CODEX_MODEL",
    "DEFAULT_REASONING_EFFORT",
    "NAMED_IMAGE_INPUT_DIGEST_SCHEMA",
    "STRUCTURED_INPUT_DIGEST_SCHEMA",
    "CodexProposerFailure",
    "CodexReceipt",
    "CodexStructuredResult",
    "CloudPolicyCacheSnapshot",
    "codex_cli_authenticated_fingerprint",
    "codex_cli_fingerprint",
    "codex_cli_version",
    "named_image_set_digest",
    "named_image_view_digest",
    "ordered_panel_view_digest",
    "run_codex_named_images_structured",
    "run_codex_structured",
    "semantic_panel_set_digest",
    "snapshot_cloud_policy_cache",
    "validate_codex_strict_output_schema",
    "validate_codex_receipt",
]
