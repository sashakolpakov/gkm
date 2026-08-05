"""Enforced leg-library orchestration (the R-LEGS design made structural).

A prompt REQUEST to "grow a leg library" was ignored -- the proposer grew a monolithic
solver (see FINDINGS R-LEGS). So the discipline is enforced by the harness here:

  * Files in the workspace are split so logic can only accumulate in a SHARED library:
      legs.py     -- reusable, named skills (perception, navigation, transport, ...)
      players.py  -- per-level players `play_level_K(env)` that ONLY compose legs
      solve.py    -- `solve(env)` dispatching by env.levels_completed to the players
  * Per level K the loop runs: PROPOSE (compose legs + minimal new) -> VERIFY on the
    real game (does solve.py reach level K, replay-validated?) -> DEBRIEF (refactor any
    repeated code into shared legs; log the recurring composition).
  * Admission/scoring uses MARGINAL free energy  F = R + lambda * C_marginal , where
    C_marginal is the NEW structure introduced this level (LOC added to legs.py +
    players.py). A REUSED leg adds zero, so parsimony directly rewards transfer: later
    levels should show near-zero marginal C. This is F=R+lambda*C with C = novelty.

The proposer and verifier are INJECTABLE (`propose_fn`, `verify_fn`) so the control
loop + marginal-C accounting can be unit-tested offline; the defaults call the real
Claude Code agent (with tools) and run the real game. Requires credits only for the
default proposer.
"""
from __future__ import annotations
import importlib.machinery
import importlib.util
import ast
import fcntl
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
import types
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

# The boundary policy must be loaded before any local helper that could import
# the raw Arena.  The compatibility runner authenticates and privately executes
# the exact source bytes below; ambient import state is never authority.
import arc_agi3_proposer_boundary as APB

_ARENA_MODULE_ROOT = Path(__file__).resolve().parent
COMPATIBILITY_ARENA_CLOSURE_AUTHORITY = False


def _compatibility_arena_host_shadow_reason(
    arena_module_root: Path | str,
) -> Optional[str]:
    """Inventory raw-import alternatives without consulting importer state.

    Direct proposer probes retain a narrow compatibility permission to import
    ``gkm_arena`` and call ``run_program``.  They are not the production
    authority (the contiguous runner uses its hash-bound Arena RPC), but the
    host compatibility root must still contain no package or native-extension
    alternative that a normal import could select.  ``PathFinder``,
    ``sys.modules``, loader metadata, and bytecode caches are deliberately not
    consulted here.

    A top-level ``gkm_arena.pyc`` and ordinary ``__pycache__`` entries are not
    alternatives to the authenticated private source execution used by the
    host harness, so they are preserved and ignored.
    """

    root = Path(arena_module_root)
    try:
        before = os.lstat(root)
    except OSError as exc:
        return f"arena_host_root_unavailable: {type(exc).__name__}"
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        return "arena_host_root_unsafe: expected a physical directory"
    if before.st_nlink < 1:
        return "arena_host_root_unsafe: invalid directory link count"
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(root, flags)
    except OSError as exc:
        return f"arena_host_root_unavailable: {type(exc).__name__}"
    try:
        opened = os.fstat(descriptor)
        if APB._stat_identity(opened) != APB._stat_identity(before):
            return "arena_host_root_raced: directory changed before inventory"
        names = tuple(os.listdir(descriptor))
        after_fd = os.fstat(descriptor)
    except OSError as exc:
        return f"arena_host_root_unavailable: {type(exc).__name__}"
    finally:
        os.close(descriptor)
    try:
        after_path = os.lstat(root)
    except OSError as exc:
        return f"arena_host_root_raced: {type(exc).__name__}"
    if (
        APB._stat_identity(after_fd) != APB._stat_identity(before)
        or APB._stat_identity(after_path) != APB._stat_identity(before)
    ):
        return "arena_host_root_raced: directory changed during inventory"

    forbidden = {"gkm_arena"}
    forbidden.update(
        "gkm_arena" + suffix
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    )
    shadows = sorted(forbidden.intersection(names))
    if shadows:
        return "arena_host_shadow: " + ", ".join(shadows)
    return None


def _read_authenticated_arena_source(
    arena_module_root: Path | str,
) -> tuple[bytes, str]:
    """Read and hash the exact physical Arena source under the APB contract."""

    raw, finding, _identity = APB._read_regular_nofollow(
        Path(arena_module_root) / "gkm_arena.py",
        logical_path="gkm_arena.py",
        kind="arena_module",
        max_bytes=APB.MAX_SOURCE_BYTES,
    )
    if finding is not None or raw is None:
        detail = finding.describe() if finding is not None else "unavailable"
        raise RuntimeError(f"raw-arena module identity is unsafe: {detail}")
    return raw, hashlib.sha256(raw).hexdigest()


def _load_authenticated_arena(
    arena_module_root: Path | str,
) -> tuple[types.ModuleType, str]:
    """Execute authenticated source bytes in an unregistered private module."""

    root = Path(arena_module_root)
    reason = _compatibility_arena_host_shadow_reason(root)
    if reason:
        raise RuntimeError(reason)
    raw, digest = _read_authenticated_arena_source(root)
    source_path = os.fspath(root / "gkm_arena.py")
    private_name = f"_gkm_authenticated_arena_{digest[:16]}"
    module = types.ModuleType(private_name)
    module.__file__ = source_path
    module.__package__ = ""
    module.__loader__ = None
    module.__spec__ = importlib.machinery.ModuleSpec(
        private_name, loader=None, origin=source_path
    )
    code = compile(raw, source_path, "exec", dont_inherit=True)
    exec(code, module.__dict__)
    required = (
        "Arena",
        "run_program",
        "validate",
        "_compile",
        "free_energy",
        "PRECONCEPTIONS",
        "API",
        "DEFAULT_STEP_CAP",
        "FRAME_SIDE",
        "PUBLIC_ACTION_PROTOCOL_VIOLATION_MARKER",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(
            "authenticated raw-arena module lacks required surface: "
            + ", ".join(missing)
        )
    reason = _compatibility_arena_host_shadow_reason(root)
    if reason:
        raise RuntimeError(reason)
    raw_after, digest_after = _read_authenticated_arena_source(root)
    if raw_after != raw or digest_after != digest:
        raise RuntimeError(
            "arena_module_control_drift: raw-arena source changed during "
            "authenticated private load"
        )
    return module, digest


def _compatibility_arena_control_reason() -> Optional[str]:
    """Reopen the physical controls and fail closed on any post-load drift."""

    reason = _compatibility_arena_host_shadow_reason(_ARENA_MODULE_ROOT)
    if reason:
        return reason
    try:
        _raw, current_digest = _read_authenticated_arena_source(
            _ARENA_MODULE_ROOT
        )
    except (OSError, RuntimeError) as exc:
        return f"arena_module_control_drift: {exc}"
    if current_digest != _LOADED_ARENA_MODULE_SHA256:
        return (
            "arena_module_control_drift: raw-arena source changed after "
            "authenticated private load"
        )
    return None


A, _LOADED_ARENA_MODULE_SHA256 = _load_authenticated_arena(
    _ARENA_MODULE_ROOT
)

import codex_campaign_status as CCS
import codex_usage_guard as CUG
import claude_usage_guard as CLG
from gkm_solve_agent import discovered_context

# Working-directory root for per-game leg workspaces. Defaults to a repo-relative
# ``runs/scratch`` dir; override with the ``GKM_SCRATCH`` environment variable.
from pathlib import Path as _Path
SCRATCH = os.environ.get(
    "GKM_SCRATCH",
    str(_Path(__file__).resolve().parent / "runs" / "scratch"),
)
ENVIRONMENT_FILES = _Path(__file__).resolve().parents[2] / "environment_files"


def authoritative_level_target(
    game: str, environments: Path | str = ENVIRONMENT_FILES
) -> int:
    """Read one game's public authoritative level count from toolkit metadata."""
    root = Path(environments)
    counts = set()
    paths = sorted((root / game).glob("*/metadata.json"))
    if not paths:
        raise ValueError(
            f"game is absent from authoritative toolkit metadata: {game!r}"
        )
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid authoritative metadata: {path}") from exc
        actions = payload.get("baseline_actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError(
                f"missing authoritative baseline_actions: {path}"
            )
        counts.add(len(actions))
    if len(counts) != 1:
        raise ValueError(
            f"conflicting authoritative level counts for {game}: "
            f"{sorted(counts)}"
        )
    return counts.pop()


def _loc(code: str) -> int:
    """Description length proxy: non-blank, non-comment lines."""
    return sum(1 for ln in (code or "").splitlines()
               if ln.strip() and not ln.strip().startswith("#"))


def _literal_cost(code: str) -> int:
    """Extra description length for large literals hidden on one line.

    LOC alone makes `execute_path(env, [60 actions...])` cost the same as
    `solve_masked(env)`. Count literal list/tuple elements so replay plans and
    other hard-coded tables carry MDL cost even when formatted on one line.
    """
    import ast
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return 0
    cost = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            cost += len(node.elts)
        elif isinstance(node, ast.Dict):
            cost += len(node.keys)
    return cost


def description_complexity(code: str) -> int:
    """Coarse code-description proxy used for marginal-C accounting."""
    return _loc(code) + _literal_cost(code)


def marginal_complexity(legs_before: str, legs_after: str,
                        players_before: str, players_after: str) -> int:
    """Return positive net description growth in the library and player files.

    Unchanged code contributes zero. Additions and deletions within the same file are
    netted before the positive part is taken, so this historical metric is not gross
    diff size and must not be interpreted as charging every newly written structure.
    """
    return (max(0, description_complexity(legs_after) - description_complexity(legs_before))
            + max(0, description_complexity(players_after) - description_complexity(players_before)))


def should_run_debrief(policy: str, *, auto_solved: bool,
                       pre_debrief_marginal_C: int,
                       threshold: int = 150) -> bool:
    """Decide whether a separate paid refactor turn is worth admitting.

    ``adaptive`` never debriefs a literal one-call auto-solve and only pays for
    a refactor after a substantial acquisition.  This leaves the successful
    pre-debrief solver as the promotion candidate and avoids spending a weekly
    point merely to narrate reuse that is already literal in source.
    """
    if policy not in {"always", "adaptive", "never"}:
        raise ValueError("debrief policy must be always, adaptive, or never")
    if threshold < 0:
        raise ValueError("debrief threshold must be nonnegative")
    if policy == "always":
        return True
    if policy == "never" or auto_solved:
        return False
    return pre_debrief_marginal_C >= threshold


def free_energy(levels: int, marginal_C_total: int, lam: float = 0.02) -> float:
    """F = R + lambda*C with R = -levels_reached and C = total marginal novelty."""
    return -float(levels) + lam * float(marginal_C_total)


CHECKPOINT_FILE = "checkpoint.json"
"""Filename for per-level marginal-C checkpoint (enables cross-run resume)."""


def _valid_replay_action(action: object) -> bool:
    """Return whether ``action`` is accepted by the local public Arena bridge."""
    if isinstance(action, int) and not isinstance(action, bool):
        return 1 <= action <= 7 and action != 6
    return (
        isinstance(action, (list, tuple))
        and len(action) == 3
        and action[0] == 6
        and not isinstance(action[0], bool)
        and all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in action[1:]
        )
        and all(0 <= value < A.FRAME_SIDE for value in action[1:])
    )


AUTO_SOLVE_LOG = "auto_solve_attempts.json"
"""Per-level record of failed auto-solve attempts, keyed by (level, legs-hash), so a
relaunch does not re-pay a long BFS that already failed against the same legs."""
AUTO_SOLVE_MAX_CANDIDATES = 6
AUTO_SOLVE_CANDIDATE_SECONDS = 10

ARC_GAME_SOURCE_NAMES = tuple(
    f"{game}.py" for game in (
        "ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59",
        "lf52", "lp85", "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26",
        "sc25", "sk48", "sp80", "su15", "tn36", "tr87", "tu93", "vc33",
        "wa30",
    )
)

SOURCE_TAINT_MARKERS = (
    "environment_files/",
    "/environment_files/",
    "agent_solutions/",
    "/agent_solutions/",
    "source reveals",
    "actual game source",
)

ARC_GAME_SOURCE_RE = re.compile(
    r"(?<![a-z0-9_])(?:"
    + "|".join(re.escape(name) for name in ARC_GAME_SOURCE_NAMES)
    + r")(?![a-z0-9_])"
)

PRIVATE_RUNTIME_RE = re.compile(
    r"\.\s*_(?:game|env|fd|budget)\b"
    r"|\benv\s*\.\s*__dict__\b"
    r"|\bvars\s*\(\s*env\b"
    r"|object\.__getattribute__"
    r"|\b(?:getattr|hasattr)\s*\([^,\n]+,\s*['\"]_(?:game|env|fd|budget)\b"
)

EXTERNAL_NETWORK_RE = re.compile(
    r"(?:^|[\n;&|])\s*(?:sudo\s+)?(?:curl|wget|lynx|links|nc|ncat|netcat|telnet|ssh|scp|rsync)(?!\s*=)\s+"
    r"|\b(?:web[_ -]?search|browser\.open|search_query|open_url)\b"
    r"|\b(?:requests|httpx|aiohttp|urllib\.request|http\.client)\s*\."
    r"|\bsocket\.(?:create_connection|socket|getaddrinfo|gethostbyname)\b"
    r"|https?://(?!localhost(?::\d+)?(?:/|\b)|127\.0\.0\.1(?::\d+)?(?:/|\b)|"
    r"\[?::1\]?(?::\d+)?(?:/|\b))",
    re.IGNORECASE,
)
"""Strings that make a proposer workspace inadmissible.

The arena may execute the hidden game implementation internally, but the
proposer must not inspect source files or solution history outside its admitted
same-lineage WIP. If any agent-authored workspace file records such an access,
the harness refuses to verify or promote the attempt.
"""

PROMOTED_FILES = ("legs.py", "players.py", "solve.py", "legs_log.md", CHECKPOINT_FILE,
                  AUTO_SOLVE_LOG)
"""Files that define a verified leg-library state and should survive scratch loss."""

SNAPSHOT_SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git"}
SNAPSHOT_SKIP_FILES: set[str] = set()
BLOCKED_ATTEMPTS_LOG = "blocked_attempts.log"
MAX_TAINT_SCAN_BYTES = 50_000_000
MAX_RECOVERY_PATH_CANDIDATES = 24
MAX_RECOVERY_PREFIX_CANDIDATES = 6
MAX_RECOVERY_GLUE_ATTEMPTS = 72


class WorkspaceTainted(RuntimeError):
    """The proposer workspace contains evidence of forbidden source/history use."""


def _codex_log_execution_surface(text: str) -> Optional[str]:
    """Return agent-authored actions from a Codex JSONL transcript.

    Command output is evidence of what an allowed observation returned, not of
    what private operation the agent requested. In particular, an exception from
    the public ``env.clone()`` API can expose private harness field names in a
    traceback. Commands remain immutable in the same JSONL, and agent-authored
    workspace files are scanned separately.
    """
    values = []
    parsed = 0
    diagnostics = []
    passive_event_types = {
        # Transport diagnostics are emitted by the Codex CLI itself.  They are
        # neither proposer-authored commands nor observations supplied to the
        # proposer.  In particular, reconnect/fallback records may name the
        # provider HTTPS endpoint; treating that URL as attempted web access
        # falsely quarantines an otherwise clean long turn.
        "error",
        "thread.started",
        "turn.started",
        "turn.completed",
        "turn.failed",
    }
    passive_item_types = {"agent_message", "reasoning", "error"}
    for raw in text.splitlines():
        if not raw.strip():
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            # Codex CLI diagnostics are outside structured item events. Keep
            # scanning their own text as agent-visible/action-adjacent evidence,
            # but never let their count force us to scan JSON-escaped tool
            # output as though it were an authored command.
            diagnostics.append(raw)
            continue
        parsed += 1
        if not isinstance(event, dict):
            # A parseable but unrecognized record must not make the whole file
            # look like a trusted Codex transcript while hiding its text.
            values.append(raw)
            continue
        item = event.get("item")
        if not isinstance(item, dict):
            if event.get("type") not in passive_event_types:
                values.append(raw)
            continue
        item_type = item.get("type")
        if item_type == "command_execution" and isinstance(item.get("command"), str):
            values.append(item["command"])
        elif item_type in {"web_search", "file_change"}:
            # Web-search requests and file-change paths are agent-authored. File
            # contents themselves are covered by the workspace walk.
            values.append(json.dumps(item, sort_keys=True))
        elif item_type not in passive_item_types:
            # Future or malformed item schemas are ambiguous. Scan their exact
            # record fail-closed instead of silently dropping a possible
            # agent-authored command.
            values.append(raw)
    # `codex exec --json` can emit many plain-text diagnostics over a long or
    # concatenated turn (model refresh and failed apply-patch messages are
    # common). Command output remains JSON-escaped inside structured events.
    # Scan the diagnostics themselves, without falling back to tool output.
    if parsed:
        return "\n".join(values + diagnostics)
    return None


def _codex_protocol_self_report(text: str) -> bool:
    """Detect an explicit agent admission that its own turn is invalid.

    The descriptor-level arena marker is the primary authority.  This narrow
    fallback covers older/current transcripts in which a probe caught the
    public-action exception and printed only ``type(error)`` while the agent
    still explicitly reported that it had attempted an out-of-frame action.
    Hypothetical policy discussion is intentionally not enough: the message
    must say the turn/generation *is* invalidated, admit an attempted action,
    and identify the 64x64 coordinate violation.
    """

    for raw in text.splitlines():
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        item = event.get("item")
        if (
            not isinstance(item, dict)
            or item.get("type") != "agent_message"
            or not isinstance(item.get("text"), str)
        ):
            continue
        message = item["text"].lower()
        invalidated = re.search(
            r"\b(?:turn|generation)\s+(?:is\s+|was\s+)?invalidated\b",
            message,
        )
        attempted = re.search(
            r"\b(?:i|probe|solver|turn|generation)\b"
            r"[^.\n]{0,120}\b(?:attempted|issued|sent|called)\b"
            r"|\battempted\s+action\b",
            message,
        )
        coordinate_fault = (
            "out-of-frame" in message
            or "out of frame" in message
            or (
                "outside" in message
                and ("0..63" in message or "64x64" in message)
            )
            or (
                "coordinate" in message
                and ("0..63" in message or "64x64" in message)
            )
        )
        if invalidated and attempted and coordinate_fault:
            return True
    return False


def _has_forbidden_host_process_command(text: str) -> bool:
    """Classify each host-process command, never a transcript as one blob.

    A narrow own-worker ``pgrep``/``ps | rg`` is informational.  It must not
    mask a separate broad query or process-control command elsewhere in the
    same turn.
    """
    return APB.has_forbidden_host_process_command(text)


def _file_taint_reason(path: str, display_name: str) -> Optional[str]:
    try:
        size = os.path.getsize(path)
        if size > MAX_TAINT_SCAN_BYTES:
            return (
                f"oversized unscanned evidence in {display_name} "
                f"({size} > {MAX_TAINT_SCAN_BYTES} bytes)"
            )
        raw = Path(path).read_bytes()
    except OSError:
        return None
    # Binary probe artifacts are observations, not an execution surface.  Their
    # container metadata commonly embeds specification/vendor URLs (for example
    # PNG XMP names Matplotlib and W3C), which is not evidence of network access.
    # Agent-authored commands remain covered by the immutable JSONL transcript.
    # Scan only strict UTF-8 text; never silently mine printable fragments from
    # an otherwise binary container.
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if A.PUBLIC_ACTION_PROTOCOL_VIOLATION_MARKER in text:
        return f"public action protocol violation in {display_name}"
    if (
        (
            os.path.basename(path) == "proposer_last.log"
            or path.endswith(".jsonl")
        )
        and _codex_protocol_self_report(text)
    ):
        return (
            "self-reported public action protocol violation in "
            f"{display_name}"
        )
    execution_surface = (
        _codex_log_execution_surface(text)
        if os.path.basename(path) == "proposer_last.log" or path.endswith(".jsonl")
        else None
    )
    if execution_surface is not None:
        text = execution_surface
    elif os.path.basename(path) == "proposer_last.log":
        if execution_surface is None:
            # Legacy prose logs may quote a blocked command as Markdown inline
            # code. The blocked-attempt ledger is the execution record.
            text = re.sub(r"`[^`\n]*`", "", text)
    text = text.lower()
    if PRIVATE_RUNTIME_RE.search(text):
        return f"private game/runtime introspection in {display_name}"
    if EXTERNAL_NETWORK_RE.search(text):
        return f"external web/network access in {display_name}"
    if (
        execution_surface is not None
        and _has_forbidden_host_process_command(text)
    ):
        return f"host process introspection in {display_name}"
    for marker in SOURCE_TAINT_MARKERS:
        if marker in text:
            return f"{marker} in {display_name}"
    source_match = ARC_GAME_SOURCE_RE.search(text)
    if source_match:
        return f"{source_match.group(0)} in {display_name}"
    return None


def _workspace_marker_taint_reason(ws: str) -> Optional[str]:
    for root, dirs, files in os.walk(ws):
        dirs[:] = [d for d in dirs if d not in SNAPSHOT_SKIP_DIRS]
        for name in files:
            if name == BLOCKED_ATTEMPTS_LOG:
                continue
            path = os.path.join(root, name)
            reason = _file_taint_reason(path, os.path.relpath(path, ws))
            if reason:
                return reason
    return None


def _workspace_boundary_reason(ws: str) -> Optional[str]:
    """Enforce the machine-readable clean-room filesystem capability."""
    arena_control_reason = _compatibility_arena_control_reason()
    if arena_control_reason:
        return arena_control_reason
    trusted = _trusted_host_scaffold_hashes(ws)
    findings = APB.scan_workspace(
        Path(ws),
        arena_module_root=Path(__file__).resolve().parent,
        trusted_host_scaffolds=trusted,
    )
    return APB.first_reason(
        _filter_trusted_scaffold_root_literal(ws, findings, trusted=trusted)
    )


def _trusted_host_scaffold_hashes(ws: str) -> Dict[str, frozenset[str]]:
    """Bind each supported compatibility scaffold to exact host-owned bytes.

    A live turn may span a host-policy upgrade.  Retain old scaffold authority
    only by rendering a frozen, versioned host template for the basename-bound
    game and hashing those exact bytes.  This is deliberately not a semantic
    or source-pattern exception: any proposer edit, including one byte, loses
    the authority.
    """
    base = os.path.basename(os.path.abspath(ws))
    games = sorted(name.removesuffix(".py") for name in ARC_GAME_SOURCE_NAMES)
    game = next(
        (
            candidate
            for candidate in games
            if base == f"gkm_legs_ws_{candidate}"
            or base.startswith(f"gkm_legs_ws_{candidate}_")
        ),
        None,
    )
    if game is None:
        match = re.fullmatch(
            r"gkm_legs_ws_([A-Za-z0-9]+)(?:_.*)?", base
        )
        if match is not None:
            # Offline/injected harness tests use synthetic game ids.  Trust
            # only bytes that exactly reproduce the host TESTER template for
            # that basename-derived id; no proposer-authored variation gains
            # this exception.
            game = match.group(1)
    testers = globals().get("_TRUSTED_HOST_TESTER_TEMPLATES")
    if not (
        isinstance(testers, tuple)
        and testers
        and all(isinstance(tester, str) for tester in testers)
    ):
        return {}
    candidate_games = [game] if game is not None else games
    hashes = frozenset(
        hashlib.sha256(tester.format(
            labdir=os.path.dirname(os.path.abspath(__file__)),
            game=candidate,
        ).encode("utf-8")).hexdigest()
        for tester in testers
        for candidate in candidate_games
    )
    return {"gkm_try.py": hashes}


def _filter_trusted_scaffold_root_literal(
    ws: str | Path,
    findings,
    *,
    trusted: Optional[Dict[str, frozenset[str]]] = None,
    sealed_payloads: Optional[Dict[str, bytes]] = None,
):
    """Allow the exact host tester's root literal, never mutated variants.

    APB already grants a digest-bound host scaffold its ``gkm_legs`` import and
    verifier calls.  Its exact module-root insertion is also host-owned, but it
    is intentionally *not* a general proposer exception; keep that final
    exception confined to the compatibility harness.
    """

    selected = tuple(findings)
    if not any(
        item.path == "gkm_try.py" and item.code == "absolute_path"
        for item in selected
    ):
        return selected
    authority = trusted or _trusted_host_scaffold_hashes(os.fspath(ws))
    payload = (sealed_payloads or {}).get("gkm_try.py")
    if payload is None:
        payload, read_finding, _identity = APB._read_regular_nofollow(
            Path(ws) / "gkm_try.py",
            logical_path="gkm_try.py",
            kind="source",
            max_bytes=APB.MAX_SOURCE_BYTES,
        )
        if read_finding is not None or payload is None:
            return selected
    digest = hashlib.sha256(payload).hexdigest()
    if not APB._trusted_digest(authority, "gkm_try.py", digest):
        return selected
    return tuple(
        item
        for item in selected
        if not (
            item.path == "gkm_try.py" and item.code == "absolute_path"
        )
    )


def _filesystem_boundary_policy_binding() -> dict:
    """Return the exact prospective policy identity sealed into evidence."""
    return {
        "filesystem_boundary_policy_schema": APB.POLICY_SCHEMA,
        "filesystem_boundary_policy_sha256": APB.policy_sha256(),
        "compatibility_arena_module_sha256": _LOADED_ARENA_MODULE_SHA256,
        "compatibility_boundary_authority": "behavioral_defense_in_depth",
    }


def _workspace_taint_reason(ws: str) -> Optional[str]:
    """Return source/environment taint or a filesystem-boundary violation."""
    # The boundary gate uses lstat/O_NOFOLLOW and must run before the marker
    # scanner opens any path.  A workspace symlink therefore cannot redirect
    # the older content scanner outside the attempt root.
    return _workspace_boundary_reason(ws) or _workspace_marker_taint_reason(ws)


def promoted_artifact_taint_reason(art: str) -> Optional[str]:
    """Scan canonical promoted evidence without reclassifying forensic WIP."""
    for name in PROMOTED_FILES:
        path = os.path.join(art, name)
        if os.path.lexists(path):
            try:
                payload = _boundary_checked_payload(path, name)
            except (OSError, WorkspaceTainted) as exc:
                return f"unsafe promoted source {name}: {exc}"
        else:
            payload = None
        reason = _file_taint_reason(path, name)
        if reason:
            return reason
    return None


def assert_workspace_not_tainted(ws: str) -> None:
    reason = _workspace_or_protected_taint_reason(ws)
    if reason:
        raise WorkspaceTainted(
            f"forbidden source/history access tainted proposer workspace: {reason}"
        )


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protected_codex_transcript_dir(ws: str) -> str:
    """Return the host-owned transcript directory paired with one workspace."""
    workspace = os.path.abspath(ws)
    return os.path.join(
        os.path.dirname(workspace),
        ".proposer_transcripts",
        os.path.basename(workspace),
    )


def _protected_transcript_taint_reason(ws: str) -> Optional[str]:
    """Scan host-owned Codex transcripts paired with one proposer workspace.

    The live transcript is deliberately outside the proposer-writable
    workspace.  Persistence and promotion gates must nevertheless reopen it:
    otherwise a caught public-action violation can leave apparently clean probe
    files and an unsafe resumable WIP snapshot.
    """
    transcript_root = _protected_codex_transcript_dir(ws)
    protected_paths = (
        glob.glob(os.path.join(transcript_root, "codex_turn_*.jsonl"))
        + glob.glob(
            os.path.join(transcript_root, "codex_turn_*.stderr.log")
        )
    )
    for path in sorted(protected_paths):
        reason = _file_taint_reason(
            path,
            os.path.join(
                ".proposer_transcripts", os.path.basename(path)
            ),
        )
        if reason:
            return reason
        if path.endswith(".jsonl"):
            boundary = APB.first_reason(
                APB.scan_codex_transcript(
                    Path(path),
                    workspace_root=Path(ws),
                    arena_module_root=Path(__file__).resolve().parent,
                )
            )
            if boundary:
                return boundary
    return None


def _workspace_or_protected_taint_reason(ws: str) -> Optional[str]:
    """Return the first taint reason across writable and host-owned evidence."""
    return (
        _workspace_taint_reason(ws)
        or _protected_transcript_taint_reason(ws)
    )


def _read_single_link_regular(path: str) -> bytes:
    """Read an unchanged, single-linked regular file without following links.

    Merely checking the pathname before ``open`` leaves a rename/unlink race:
    the process can retain an open descriptor after archive cleanup has removed
    the directory entry.  Evidence is admissible only when the pathname still
    resolves to the same single-linked inode after the complete read.
    """
    before = os.lstat(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or before.st_nlink != 1
            or opened.st_nlink != 1
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise WorkspaceTainted(
                f"refusing aliased/non-regular evidence source: {path}"
            )
        chunks = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        payload = b"".join(chunks)
        final_descriptor = os.fstat(descriptor)
        final_path = os.lstat(path)
        stable_identity = (
            (opened.st_dev, opened.st_ino)
            == (final_descriptor.st_dev, final_descriptor.st_ino)
            == (final_path.st_dev, final_path.st_ino)
        )
        stable_content = (
            opened.st_size == final_descriptor.st_size == len(payload)
            and opened.st_mtime_ns == final_descriptor.st_mtime_ns
            and opened.st_ctime_ns == final_descriptor.st_ctime_ns
            and opened.st_mode == final_descriptor.st_mode
        )
        if (
            not stable_identity
            or not stable_content
            or final_descriptor.st_nlink != 1
            or final_path.st_nlink != 1
        ):
            raise WorkspaceTainted(
                f"evidence source changed or was unlinked while read: {path}"
            )
        return payload
    finally:
        os.close(descriptor)


def _write_promotion_evidence(
    game: str,
    ws: str,
    art: str,
    rep: Report,
    *,
    authorized_turn: Optional[dict] = None,
) -> None:
    """Freeze a machine-verifiable provenance record for a new promoted level."""
    evidence_root = os.path.join(art, "promotion_evidence")
    evidence_dir = os.path.join(evidence_root, f"level_{rep.reached:02d}")
    os.makedirs(evidence_dir, exist_ok=True)

    protected_root = _protected_codex_transcript_dir(ws)
    protected_sources = []
    protected_diagnostics = []
    authority_kind = None
    if (
        isinstance(authorized_turn, dict)
        and authorized_turn.get("authority_kind") == "host_auto_solve"
    ):
        authority_kind = "host_auto_solve"
        transcript_src = None
    elif authorized_turn is not None:
        if not isinstance(authorized_turn, dict):
            raise ProposerEvidenceUnavailable(
                "promotion authority record is not an object"
            )
        transcript_name = authorized_turn.get("transcript")
        diagnostics_name = authorized_turn.get("diagnostics")
        if (
            not isinstance(transcript_name, str)
            or Path(transcript_name).name != transcript_name
            or not transcript_name.endswith(".jsonl")
        ):
            raise ProposerEvidenceUnavailable(
                "winning turn does not identify one protected transcript"
            )
        transcript_src = os.path.join(protected_root, transcript_name)
        if not os.path.isfile(transcript_src):
            raise ProposerEvidenceUnavailable(
                "winning protected transcript is unavailable"
            )
        protected_sources = [transcript_src]
        authority_kind = "codex_turn"
        if diagnostics_name is not None:
            if (
                not isinstance(diagnostics_name, str)
                or Path(diagnostics_name).name != diagnostics_name
                or not diagnostics_name.endswith(".stderr.log")
            ):
                raise ProposerEvidenceUnavailable(
                    "winning turn diagnostics binding is malformed"
                )
            diagnostics_src = os.path.join(protected_root, diagnostics_name)
            if not os.path.isfile(diagnostics_src):
                raise ProposerEvidenceUnavailable(
                    "winning protected diagnostics are unavailable"
                )
            protected_diagnostics = [diagnostics_src]
    else:
        # Injected/offline proposers have no Codex turn record.  Their explicit
        # workspace transcript remains the only admissible fallback; never pick
        # the newest protected glob, which may belong to a failed debrief.
        transcript_src = os.path.join(ws, "proposer_last.log")
    transcript_dst = os.path.join(evidence_dir, "proposer_last.log")
    if transcript_src is not None and os.path.isfile(transcript_src):
        _atomic_host_write(
            transcript_dst, _read_single_link_regular(transcript_src)
        )
    elif not os.path.exists(transcript_dst):
        _atomic_host_write(transcript_dst, b"")

    codex_transcripts = []
    codex_evidence_dir = os.path.join(evidence_dir, "codex_turns")
    codex_sources = protected_sources
    for source in codex_sources:
        os.makedirs(codex_evidence_dir, exist_ok=True)
        name = os.path.basename(source)
        destination = os.path.join(codex_evidence_dir, name)
        _atomic_host_write(
            destination, _read_single_link_regular(source)
        )
        codex_transcripts.append({
            "path": os.path.join("codex_turns", name),
            "sha256": _sha256_file(destination),
        })

    codex_diagnostics = []
    for source in protected_diagnostics:
        os.makedirs(codex_evidence_dir, exist_ok=True)
        name = os.path.basename(source)
        destination = os.path.join(codex_evidence_dir, name)
        _atomic_host_write(
            destination, _read_single_link_regular(source)
        )
        codex_diagnostics.append({
            "path": os.path.join("codex_turns", name),
            "sha256": _sha256_file(destination),
        })

    parent_manifest = None
    parent_hash = None
    prior = sorted(glob.glob(os.path.join(evidence_root, "level_*", "manifest.json")))
    prior = [path for path in prior if os.path.dirname(path) != evidence_dir]
    if prior:
        parent_manifest = os.path.relpath(prior[-1], art)
        parent_hash = _sha256_file(prior[-1])

    files_dir = os.path.join(evidence_dir, "files")
    os.makedirs(files_dir, exist_ok=True)
    file_hashes = {}
    for name in PROMOTED_FILES:
        path = os.path.join(art, name)
        if os.path.isfile(path):
            evidence_path = os.path.join(files_dir, name)
            _copy_boundary_checked(path, evidence_path, name)
            file_hashes[name] = _sha256_file(evidence_path)
    manifest = {
        "schema": 1,
        "game": game,
        "level": rep.reached,
        "validated": bool(rep.validated),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_manifest": parent_manifest,
        "parent_manifest_sha256": parent_hash,
        "promoted_files_sha256": file_hashes,
        "transcript": "proposer_last.log",
        "transcript_sha256": _sha256_file(transcript_dst),
        "codex_transcripts": codex_transcripts,
        "codex_diagnostics": codex_diagnostics,
        "taint_verdict": "clean",
        "proposer_workspace": os.path.basename(os.path.abspath(ws)),
        "proposer_workspace_root": os.path.abspath(ws),
        "authorized_turn_transcript": (
            authorized_turn.get("transcript")
            if isinstance(authorized_turn, dict)
            else None
        ),
        "promotion_authority_kind": authority_kind or "injected_proposer",
        **_filesystem_boundary_policy_binding(),
    }
    _atomic_host_write(
        os.path.join(evidence_dir, "manifest.json"),
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
    )


def _deduplicate_level_records(rep: Report) -> int:
    """Keep the last record for each level and remove its earlier charges.

    This makes checkpoint recovery idempotent if an older overlapping run adopted
    a level that another process had just recorded. New runs are prevented from
    overlapping by the workspace lock below; this normalization repairs legacy
    duplicate rows when they are loaded or saved.
    """
    seen = set()
    kept_reversed = []
    removed_cost = 0
    for record in reversed(rep.records):
        if record.level in seen:
            removed_cost += record.marginal_C
            continue
        seen.add(record.level)
        kept_reversed.append(record)
    if removed_cost:
        rep.records = list(reversed(kept_reversed))
    # Records are canonical; the cached aggregate may come from an interrupted
    # or formerly overlapping promotion and must never be trusted independently.
    rep.total_marginal_C = sum(record.marginal_C for record in rep.records)
    return removed_cost


def _record_level(rep: Report, level: int, marginal_C: int,
                  reached: bool = True) -> None:
    """Insert or replace one level's charge; a level may occur only once."""
    old = [record for record in rep.records if record.level == level]
    if old:
        rep.total_marginal_C -= sum(record.marginal_C for record in old)
        rep.records = [record for record in rep.records if record.level != level]
    rep.records.append(LevelRecord(level=level, marginal_C=marginal_C, reached=reached))
    rep.records.sort(key=lambda record: record.level)
    rep.total_marginal_C += marginal_C


def _atomic_host_write(path: str, payload: bytes) -> None:
    """Atomically replace one host-owned file without following aliases."""
    parent = os.path.dirname(os.path.abspath(path))
    if os.path.islink(parent) or not os.path.isdir(parent):
        raise WorkspaceTainted(
            f"unsafe host-write directory: {parent}"
        )
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        metadata = None
    if metadata is not None and (
        not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
    ):
        raise WorkspaceTainted(
            f"refusing aliased/non-regular host-write target: {path}"
        )
    descriptor, temporary = tempfile.mkstemp(prefix=".host-write.", dir=parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(
            parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _boundary_checked_payload(path: str, logical_path: str) -> bytes:
    """Seal one exact source image and apply the current boundary to its bytes."""
    arena_control_reason = _compatibility_arena_control_reason()
    if arena_control_reason:
        raise WorkspaceTainted(arena_control_reason)
    payload = _read_single_link_regular(path)
    suffix = Path(logical_path).suffix.lower()
    if suffix not in APB.SOURCE_SUFFIXES:
        return payload
    try:
        source = payload.decode("utf-8")
    except UnicodeError as exc:
        raise WorkspaceTainted(
            f"non-UTF-8 executable source {logical_path}: {type(exc).__name__}"
        ) from exc
    if suffix in {".py", ".pyw"}:
        scaffold_hashes = _trusted_host_scaffold_hashes(
            os.path.dirname(path)
        ).get(logical_path, ())
        findings = APB.scan_python_source(
            source,
            logical_path=logical_path,
            arena_module_root=Path(__file__).resolve().parent,
            allow_host_scaffold=(
                hashlib.sha256(payload).hexdigest() in scaffold_hashes
            ),
        )
    else:
        findings = APB.scan_shell_command(
            source,
            logical_path=logical_path,
            line=1,
            arena_module_root=Path(__file__).resolve().parent,
        )
    findings = _filter_trusted_scaffold_root_literal(
        os.path.dirname(path),
        findings,
        sealed_payloads={logical_path: payload},
    )
    reason = APB.first_reason(findings)
    if reason:
        raise WorkspaceTainted(
            "filesystem boundary changed after its prior scan: " + reason
        )
    return payload


def _copy_boundary_checked(path: str, destination: str, logical_path: str) -> None:
    """Copy exactly the bytes accepted by the post-scan boundary gate."""
    _atomic_host_write(
        destination, _boundary_checked_payload(path, logical_path)
    )


def _save_checkpoint(ws: str, rep: Report) -> None:
    """Persist the Report so a later restart restores the full marginal-C history."""
    _deduplicate_level_records(rep)
    data = {
        "game": rep.game,
        "reached": rep.reached,
        "total_marginal_C": rep.total_marginal_C,
        "records": [{"level": r.level, "marginal_C": r.marginal_C, "reached": r.reached}
                     for r in rep.records],
        "final_path": rep.final_path,
        "validated": rep.validated,
    }
    payload = json.dumps(
        data, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    _atomic_host_write(os.path.join(ws, CHECKPOINT_FILE), payload)


def _load_checkpoint(ws: str) -> Optional[Report]:
    """Restore a structurally valid host checkpoint, or return ``None``.

    Proposer workspaces are not a checkpoint trust boundary.  A proposer may
    leave a path-only or partially written ``checkpoint.json`` behind; recovery
    harvests its path separately.  Never construct ``Report`` from such bytes:
    malformed field types used to survive the old three-key test and fail later
    with ``KeyError``/``TypeError`` in scheduling or accounting.
    """
    path = os.path.join(ws, CHECKPOINT_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    # The proposer can legitimately emit ``checkpoint.json`` as an untrusted
    # winning-path artifact.  Only the harness writes the richer resumable
    # Report schema.  A path-only file must remain available to
    # ``recover_discovered_path_artifact`` without crashing checkpoint adoption
    # or being mistaken for trusted accounting/provenance state.
    required = {
        "game",
        "reached",
        "total_marginal_C",
        "records",
        "final_path",
        "validated",
    }
    if not isinstance(data, dict) or set(data) != required:
        return None
    game = data["game"]
    reached = data["reached"]
    total = data["total_marginal_C"]
    records = data["records"]
    final_path = data["final_path"]
    validated = data["validated"]
    if (
        not isinstance(game, str)
        or not game
        or not isinstance(reached, int)
        or isinstance(reached, bool)
        or reached < 0
        or not isinstance(total, int)
        or isinstance(total, bool)
        or total < 0
        or not isinstance(records, list)
        or not isinstance(final_path, list)
        or not isinstance(validated, bool)
    ):
        return None
    parsed_records = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "level", "marginal_C", "reached"
        }:
            return None
        level = record["level"]
        marginal = record["marginal_C"]
        record_reached = record["reached"]
        if (
            not isinstance(level, int)
            or isinstance(level, bool)
            or not 1 <= level <= reached
            or not isinstance(marginal, int)
            or isinstance(marginal, bool)
            or marginal < 0
            or not isinstance(record_reached, bool)
        ):
            return None
        parsed_records.append(
            LevelRecord(
                level=level,
                marginal_C=marginal,
                reached=record_reached,
            )
        )
    if not all(_valid_replay_action(action) for action in final_path):
        return None
    if reached == 0 and (records or final_path or total):
        return None
    if validated and reached > 0:
        if not final_path:
            return None
        if any(not record.reached for record in parsed_records):
            return None
    try:
        rep = Report(
            game=game,
            reached=reached,
            total_marginal_C=total,
            records=parsed_records,
            final_path=final_path,
            validated=validated,
        )
    except (TypeError, ValueError):
        return None
    _deduplicate_level_records(rep)
    return rep


def _open_unaliased_lock(path: str, *, create: bool = True):
    """Open a host lock without following or accepting an inode alias."""
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    if create:
        flags |= os.O_CREAT
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise RuntimeError(
            f"lock must be an unaliased regular host file: {path}"
        ) from exc
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        os.close(descriptor)
        raise RuntimeError(
            f"lock must be an unaliased regular host file: {path}"
        )
    os.fchmod(descriptor, 0o600)
    return os.fdopen(descriptor, "r+", encoding="utf-8")


def _workspace_lock_path(ws: str) -> Path:
    """Return the host-owned lock paired with a proposer workspace."""

    workspace = Path(ws).absolute()
    digest = hashlib.sha256(os.fspath(workspace).encode("utf-8")).hexdigest()
    return (
        workspace.parent
        / ".workspace_locks"
        / f"{digest}.lock"
    )


def _acquire_workspace_lock(ws: str):
    """Hold an exclusive process lock for one orchestrator per scratch workspace."""
    workspace = Path(ws)
    if workspace.is_symlink() or not workspace.is_dir():
        raise RuntimeError(
            f"workspace lock root must be a regular host directory: {ws}"
        )
    # A previous runner version placed this file inside the proposer-writable
    # tree.  Respect an active legacy lock, but delete an unlocked one before
    # launching: it has no authority and must not become a context channel.
    legacy_path = os.path.join(ws, ".orchestrate.lock")
    if os.path.lexists(legacy_path):
        legacy = _open_unaliased_lock(legacy_path, create=False)
        try:
            try:
                fcntl.flock(
                    legacy.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                )
            except BlockingIOError:
                raise RuntimeError(
                    f"another orchestrator is already using workspace {ws}"
                )
            opened = os.fstat(legacy.fileno())
            named = os.lstat(legacy_path)
            if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
                raise RuntimeError(
                    f"legacy workspace lock changed during migration: {ws}"
                )
            os.unlink(legacy_path)
        finally:
            legacy.close()
    path = _workspace_lock_path(ws)
    os.makedirs(path.parent, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise RuntimeError(
            f"workspace lock root must be a regular host directory: {path.parent}"
        )
    lock = _open_unaliased_lock(os.fspath(path))
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError(f"another orchestrator is already using workspace {ws}")
    lock.seek(0)
    lock.truncate()
    lock.write(f"pid={os.getpid()}\n")
    lock.flush()
    return lock


def _acquire_lineage_lock(game: str):
    """Allow only one writer for a game within one artifact lineage root.

    Workspace locks prevent two processes from sharing an identically tagged
    scratch directory.  They do not prevent different tags from racing to
    promote the same ``<game>_legs`` artifact.  The lineage lock is rooted
    beside that artifact, so canonical and deliberately isolated candidate
    reacquisitions remain independent while duplicate canonical writers fail
    before seeding or proposing.
    """
    artifact = artifact_dir(game)
    lock_dir = os.path.join(os.path.dirname(artifact), ".campaign_locks")
    os.makedirs(lock_dir, exist_ok=True)
    lock_root = Path(lock_dir)
    if lock_root.is_symlink() or not lock_root.is_dir():
        raise RuntimeError(
            f"lineage lock root must be a regular host directory: {lock_dir}"
        )
    path = os.path.join(lock_dir, f"{game}.lock")
    lock = _open_unaliased_lock(path)
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError(
            f"another orchestrator is already writing the {game} artifact lineage"
        )
    lock.seek(0)
    lock.truncate()
    lock.write(f"pid={os.getpid()}\nartifact={artifact}\n")
    lock.flush()
    return lock


def _release_workspace_lock(lock) -> None:
    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    lock.close()


def _adopt_workspace_checkpoint(game: str, ws: str, rep: Report,
                                verbose: bool = True) -> Report:
    """Recognize a candidate path without adopting proposer bookkeeping.

    ``recover_discovered_path_artifact`` independently replays checkpoint path
    candidates immediately after this call.  Records, cached marginal totals,
    ``validated``, and reached values remain exclusively host-owned.
    """
    ws_rep = _load_checkpoint(ws)
    if ws_rep is None or ws_rep.game != game:
        return rep
    if ws_rep.reached < rep.reached or not ws_rep.final_path:
        return rep
    if not A.validate(game, ws_rep.final_path, ws_rep.reached):
        return rep
    if verbose:
        print(
            "recognized replay-valid proposer path candidate without "
            f"adopting checkpoint accounting: claimed_reached={ws_rep.reached} "
            f"path_len={len(ws_rep.final_path)}"
        )
    return rep


def artifact_dir(game: str, tag: str = "") -> str:
    """Stable, repo-visible storage for the latest verified leg-library artifact."""
    labdir = os.path.dirname(os.path.abspath(__file__))
    root = os.environ.get("GKM_ARTIFACTS_ROOT")
    if root:
        root = os.path.abspath(root)
    else:
        root = os.path.join(labdir, "agent_solutions")
    return os.path.join(root, f"{game}_legs")


def _workspace_has_unpromoted_solver_source(
    game: str, ws: str, tag: str = ""
) -> bool:
    """Whether restored/interrupted source differs from the promoted parent."""
    art = artifact_dir(game, tag)
    for name in ("legs.py", "players.py", "solve.py"):
        workspace_path = os.path.join(ws, name)
        promoted_path = os.path.join(art, name)
        if not os.path.isfile(workspace_path):
            continue
        if not os.path.isfile(promoted_path):
            return True
        if _read(workspace_path) != _read(promoted_path):
            return True
    return False


def _frontier_marginal_baseline(
    game: str,
    ws: str,
    reached: int,
    tag: str = "",
) -> tuple[str, str]:
    """Return the promoted parent source for frontier novelty accounting.

    Clean WIP may accumulate source over several proposer turns.  Charging
    only the last retry's edit undercounts the level's actual novelty, so a
    validated promoted artifact at exactly ``reached`` is authoritative.
    Zero-seed and not-yet-promoted roots fall back to the current workspace.
    """
    workspace_legs = _read(os.path.join(ws, "legs.py"))
    workspace_players = _read(os.path.join(ws, "players.py"))
    art = artifact_dir(game, tag)
    parent = _load_checkpoint(art)
    if (
        parent is None
        or parent.game != game
        or parent.reached != reached
        or not parent.validated
    ):
        return workspace_legs, workspace_players
    parent_legs_path = os.path.join(art, "legs.py")
    parent_players_path = os.path.join(art, "players.py")
    if (
        not os.path.isfile(parent_legs_path)
        or not os.path.isfile(parent_players_path)
    ):
        return workspace_legs, workspace_players
    return _read(parent_legs_path), _read(parent_players_path)


def _clean_unpromoted_source_overlay(
    game: str, ws: str, tag: str = ""
) -> Dict[str, str]:
    """Capture clean source edits that belong on the validated parent.

    Startup seeding refreshes the supervisor-owned checkpoint and promoted
    files.  If a prior proposer already wrote a clean winning (or speculative)
    source before its harness exited, blindly seeding first destroys the exact
    source before orphan recovery can replay it.  Preserve only when scratch
    and artifact carry the same validated parent boundary; a new/stale scratch
    workspace must never override the artifact merely because its templates
    differ.
    """
    if not _workspace_has_unpromoted_solver_source(game, ws, tag):
        return {}
    workspace_parent = _load_checkpoint(ws)
    artifact_parent = _load_checkpoint(artifact_dir(game, tag))
    if (
        workspace_parent is None
        or artifact_parent is None
        or not workspace_parent.validated
        or not artifact_parent.validated
        or workspace_parent.game != game
        or artifact_parent.game != game
        or workspace_parent.reached != artifact_parent.reached
        or workspace_parent.final_path != artifact_parent.final_path
    ):
        return {}
    if _workspace_or_protected_taint_reason(ws):
        return {}
    overlay = {}
    for name in ("legs.py", "players.py", "solve.py", "legs_log.md"):
        path = os.path.join(ws, name)
        if os.path.isfile(path):
            try:
                overlay[name] = _boundary_checked_payload(
                    path, name
                ).decode("utf-8")
            except (UnicodeError, OSError, WorkspaceTainted):
                return {}
    return overlay


def _clean_wip_source_overlay(
    game: str, level: int, tag: str = "",
    expected_attempt: Optional[str] = None,
) -> Dict[str, str]:
    """Recover taint-clean same-parent source from a saved WIP snapshot.

    A continuation commonly uses a new scratch tag. Probe restoration alone
    must not silently replace the prior turn's speculative ``legs.py`` and
    ``players.py`` with the promoted parent. Search newest-first for a regular,
    taint-clean snapshot whose embedded checkpoint is exactly the current
    validated parent and whose solver source actually differs from that parent.
    The trusted checkpoint itself is never restored from WIP.
    """
    art = artifact_dir(game, tag)
    parent = _load_checkpoint(art)
    if parent is None or not parent.validated or parent.game != game:
        return {}
    level_dir = _wip_level_dir(art, level)
    if not os.path.isdir(level_dir):
        return {}

    candidates = []
    for attempt in os.listdir(level_dir):
        if expected_attempt is not None and attempt != expected_attempt:
            continue
        attempt_dir = os.path.join(level_dir, attempt)
        files_dir = os.path.join(attempt_dir, "files")
        meta_path = os.path.join(attempt_dir, "metadata.json")
        if (
            os.path.islink(attempt_dir)
            or not os.path.isdir(files_dir)
            or os.path.islink(files_dir)
            or not os.path.isfile(meta_path)
            or os.path.islink(meta_path)
        ):
            continue
        try:
            with open(meta_path) as handle:
                metadata = json.load(handle)
        except Exception:
            continue
        if (
            not isinstance(metadata, dict)
            or metadata.get("game") != game
            or metadata.get("level") != level
            or not _wip_uses_current_boundary_policy(Path(meta_path))
        ):
            continue
        created = metadata.get("created_at")
        if not isinstance(created, str):
            created = ""
        candidates.append((created, attempt, files_dir))

    parent_sources = {}
    required = ("legs.py", "players.py", "solve.py")
    for name in required:
        path = os.path.join(art, name)
        if not os.path.isfile(path) or os.path.islink(path):
            return {}
        try:
            parent_sources[name] = _boundary_checked_payload(
                path, name
            ).decode("utf-8")
        except (UnicodeError, OSError, WorkspaceTainted):
            return {}

    for _, _, files_dir in sorted(candidates, reverse=True):
        snapshot_parent = _load_checkpoint(files_dir)
        if (
            snapshot_parent is None
            or not snapshot_parent.validated
            or snapshot_parent.game != parent.game
            or snapshot_parent.reached != parent.reached
            or snapshot_parent.final_path != parent.final_path
        ):
            continue
        if _workspace_taint_reason(files_dir):
            continue
        if any(
            not os.path.isfile(os.path.join(files_dir, name))
            or os.path.islink(os.path.join(files_dir, name))
            for name in required
        ):
            continue
        overlay = {}
        try:
            for name in (*required, "legs_log.md"):
                path = os.path.join(files_dir, name)
                if os.path.isfile(path) and not os.path.islink(path):
                    overlay[name] = _boundary_checked_payload(
                        path, name
                    ).decode("utf-8")
        except (UnicodeError, OSError, WorkspaceTainted):
            continue
        if any(overlay[name] != parent_sources[name] for name in required):
            return overlay
    return {}


def _restore_source_overlay(ws: str, overlay: Dict[str, str]) -> None:
    for name, source in overlay.items():
        _atomic_host_write(
            os.path.join(ws, name), source.encode("utf-8")
        )


@dataclass(frozen=True)
class _WorkspaceRollbackPoint:
    files: Dict[str, Tuple[bytes, int]]
    directories: Dict[str, int]
    root_mode: int


def _seal_workspace_rollback_point(ws: str) -> _WorkspaceRollbackPoint:
    """Seal all mutable workspace state before an optional debrief.

    WIP snapshots intentionally omit disposable caches and local Git metadata.
    A rollback point cannot: a failed debrief could otherwise leave knowledge
    in those omitted namespaces for the next proposer.  Only the live
    supervisor-owned lock is excluded, and the production proposer process
    group has already been proved quiescent before its debrief call returns.
    """

    assert_workspace_not_tainted(ws)
    root = os.path.abspath(ws)
    root_metadata = os.lstat(root)
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise WorkspaceTainted("rollback root is not a physical directory")
    files: Dict[str, Tuple[bytes, int]] = {}
    directories: Dict[str, int] = {}
    for directory, dirs, names in os.walk(root, topdown=True, followlinks=False):
        dirs.sort()
        names.sort()
        for name in dirs:
            path = os.path.join(directory, name)
            rel = os.path.relpath(path, root)
            metadata = os.lstat(path)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise WorkspaceTainted(
                    f"refusing aliased/non-directory rollback state: {path}"
                )
            directories[rel] = stat.S_IMODE(metadata.st_mode)
        for name in names:
            path = os.path.join(directory, name)
            rel = os.path.relpath(path, root)
            if rel in SNAPSHOT_SKIP_FILES:
                continue
            metadata = os.lstat(path)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
            ):
                raise WorkspaceTainted(
                    f"refusing aliased/non-regular rollback state: {path}"
                )
            files[rel] = (
                _read_single_link_regular(path),
                stat.S_IMODE(metadata.st_mode),
            )
    return _WorkspaceRollbackPoint(
        files=files,
        directories=directories,
        root_mode=stat.S_IMODE(root_metadata.st_mode),
    )


def _restore_workspace_rollback_point(
    ws: str, sealed: _WorkspaceRollbackPoint
) -> None:
    """Remove every debrief delta and restore the exact sealed clean tree."""
    root = os.path.abspath(ws)
    os.chmod(root, sealed.root_mode | 0o700)
    for directory, dirs, _files in os.walk(
        root,
        topdown=True,
        followlinks=False,
        onerror=lambda exc: (_ for _ in ()).throw(exc),
    ):
        for name in dirs:
            path = os.path.join(directory, name)
            if not os.path.islink(path):
                metadata = os.lstat(path)
                os.chmod(path, stat.S_IMODE(metadata.st_mode) | 0o700)
    for directory, dirs, files in os.walk(root, topdown=False, followlinks=False):
        for name in files:
            path = os.path.join(directory, name)
            rel = os.path.relpath(path, root)
            os.unlink(path)
        for name in dirs:
            path = os.path.join(directory, name)
            if os.path.islink(path):
                os.unlink(path)
            else:
                os.rmdir(path)
    for name, mode in sorted(
        sealed.directories.items(), key=lambda item: len(Path(item[0]).parts)
    ):
        path = os.path.join(root, name)
        os.mkdir(path, mode)
        os.chmod(path, mode)
    for name, (payload, mode) in sealed.files.items():
        path = os.path.join(root, name)
        _atomic_host_write(path, payload)
        os.chmod(path, mode)
    os.chmod(root, sealed.root_mode)
    assert_workspace_not_tainted(root)


def _wip_level_dir(art: str, level: int) -> str:
    return os.path.join(art, "wip_context", f"level_{level:02d}")


def _workspace_snapshot_files(ws: str) -> List[str]:
    """Return every admissible workspace file as a stable relative path.

    Hard frontiers often grow small agent-authored subdirectories containing
    search outputs or probe families.  Those are part of the reproducible WIP
    state just as much as top-level probes, so omit only harness locks and
    disposable VCS/cache trees.
    """
    files = []
    for root, dirs, names in os.walk(ws):
        kept_dirs = []
        for directory in sorted(dirs):
            path = os.path.join(root, directory)
            if os.path.islink(path):
                raise WorkspaceTainted(
                    f"refusing workspace symlink during WIP inventory: {path}"
                )
            if directory not in SNAPSHOT_SKIP_DIRS:
                kept_dirs.append(directory)
        dirs[:] = kept_dirs
        for name in sorted(names):
            path = os.path.join(root, name)
            rel = os.path.relpath(path, ws)
            if rel in SNAPSHOT_SKIP_FILES:
                continue
            try:
                metadata = os.lstat(path)
            except OSError as exc:
                raise WorkspaceTainted(
                    f"workspace WIP inventory changed: {path}: {exc}"
                ) from exc
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
            ):
                raise WorkspaceTainted(
                    f"refusing aliased/non-regular WIP source: {path}"
                )
            files.append(rel)
    return sorted(files)


def _snapshot_digest(
    ws: str,
    phase: str,
    names: List[str],
    payloads: Optional[Dict[str, bytes]] = None,
) -> str:
    h = hashlib.sha256()
    h.update(phase.encode("utf-8"))
    for name in names:
        h.update(name.encode("utf-8"))
        payload = (
            payloads[name]
            if payloads is not None
            else _boundary_checked_payload(os.path.join(ws, name), name)
        )
        h.update(payload)
    return h.hexdigest()[:12]


def _artifact_readme(game: str, rep: Report) -> str:
    rows = "\n".join(f"- L{r.level}: marginal_C={r.marginal_C}" for r in rep.records)
    if not rows:
        rows = "- No per-level marginal-C records in this artifact."
    return (
        f"# {game} legs artifact\n\n"
        "Latest replay-validated leg-library state promoted by `gkm_legs.py`.\n\n"
        f"- Game: `{game}`\n"
        f"- Verified through level: {rep.reached}\n"
        f"- Replay validated: {rep.validated}\n"
        f"- Total marginal_C: {rep.total_marginal_C}\n"
        f"- Final path length: {len(rep.final_path or [])}\n\n"
        "Per-level novelty:\n\n"
        f"{rows}\n\n"
        "Files here are the clean state to resume from. New runs seed the scratch\n"
        "workspace from this directory before asking a proposer for the next level.\n"
    )


def _artifact_run_log(game: str, rep: Report) -> str:
    lines = [
        f"=== {game}: reached level {rep.reached} | validated={rep.validated} | "
        f"total_marginal_C={rep.total_marginal_C} | F={rep.free_energy:.3f} ==="
    ]
    if rep.records:
        lines.append(
            "per-level marginal novelty: "
            + ", ".join(f"L{r.level}:{r.marginal_C}" for r in rep.records)
        )
    return "\n".join(lines) + "\n"


def snapshot_wip_context(game: str, ws: str, level: int, phase: str,
                         reached: Optional[int] = None,
                         err: Optional[str] = None,
                         tag: str = "",
                         verbose: bool = True) -> str:
    """Persist unverified live probe context outside scratch without promoting code.

    Verified artifacts at the artifact root stay clean.  This copies the current
    scratch files into a content-addressed WIP snapshot so interrupted runs leave
    their probe scripts, proposer transcript, and failed candidates available for
    later continuation.
    """
    # Fail closed at the persistence boundary itself.  Callers normally scan
    # immediately after a proposer turn, but cancellation/copy-finalization
    # races can expose taint between that scan and this write.  A tainted
    # workspace must never create or replace a resumable ``latest.json``.
    taint_reason = _workspace_or_protected_taint_reason(ws)
    if taint_reason:
        raise WorkspaceTainted(
            "refusing to snapshot tainted proposer workspace: "
            f"{taint_reason}"
        )
    art = artifact_dir(game, tag)
    os.makedirs(art, exist_ok=True)
    try:
        frontier_binding = CCS.exact_frontier_binding(
            Path(art), game=game, target_level=level
        )
    except ValueError:
        # Injected/offline runs can intentionally operate without a promoted
        # artifact parent.  Their snapshots remain forensic but cannot become
        # scheduler-authorized WIP; the status reducer requires a valid binding.
        frontier_binding = None
    names = _workspace_snapshot_files(ws)
    # Seal once, then use the same accepted bytes for both the attempt digest
    # and archive. A post-scan rewrite cannot change what is retained.
    payloads = {
        name: _boundary_checked_payload(os.path.join(ws, name), name)
        for name in names
    }
    digest = _snapshot_digest(ws, phase, names, payloads)
    level_dir = _wip_level_dir(art, level)
    attempt = f"{phase}_{digest}"
    attempt_dir = os.path.join(level_dir, attempt)
    files_dir = os.path.join(attempt_dir, "files")
    os.makedirs(files_dir, exist_ok=True)
    for name in names:
        dst = os.path.join(files_dir, name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        _atomic_host_write(dst, payloads[name])
    meta = {
        "game": game,
        "level": level,
        "phase": phase,
        "reached": reached,
        "err": err,
        "attempt": attempt,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "taint_verdict": "clean",
        "frontier_binding": frontier_binding,
        "files": names,
        "proposer_workspace": os.path.basename(os.path.abspath(ws)),
        "proposer_workspace_root": os.path.abspath(ws),
        **_filesystem_boundary_policy_binding(),
    }
    with open(os.path.join(attempt_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(attempt_dir, "README.md"), "w") as f:
        f.write(
            f"# WIP context for {game} level {level}\n\n"
            "This is unverified continuation context, not a promoted solution.\n\n"
            f"- Phase: `{phase}`\n"
            f"- Observed reached: `{reached}`\n"
            f"- Error: `{err}`\n"
            f"- Attempt id: `{attempt}`\n\n"
            "The canonical verified artifact files remain at the artifact root. "
            "Files under `files/` are the scratch state and proposer transcript from "
            "this attempt, saved so future continuations do not lose live probes.\n"
        )
    with open(os.path.join(level_dir, "latest.json"), "w") as f:
        json.dump({"attempt": attempt, "metadata": meta}, f, indent=2)
    if verbose:
        print(f"saved WIP context: {attempt_dir}")
    return attempt_dir


def _legs_hash(legs_code: str) -> str:
    return hashlib.sha256((legs_code or "").encode("utf-8")).hexdigest()[:12]


def _auto_solve_failed_before(ws: str, K: int, legs_code: str) -> bool:
    """True when auto-solve already failed at level K against this exact legs.py."""
    data = json.loads(_read(os.path.join(ws, AUTO_SOLVE_LOG)) or "{}")
    return f"{K}:{_legs_hash(legs_code)}" in data.get("failed", [])


def _record_auto_solve_failure(ws: str, K: int, legs_code: str) -> None:
    path = os.path.join(ws, AUTO_SOLVE_LOG)
    data = json.loads(_read(path) or "{}")
    key = f"{K}:{_legs_hash(legs_code)}"
    failed = data.setdefault("failed", [])
    if key not in failed:
        failed.append(key)
    with open(path, "w") as f:
        json.dump(data, f)


def _wip_path_targets_other_level(path: str, level: int) -> bool:
    """Return true for explicitly level-numbered scratch from another frontier."""
    normalized = path.replace(os.sep, "/")
    match = re.search(
        r"(?:^|/)codex_turn_[^/]*_L(\d+)_(?:propose|debrief)\.jsonl$",
        normalized,
        re.IGNORECASE,
    )
    if match:
        return int(match.group(1)) != level
    match = re.search(
        r"(?:^|/)(?:probe|search)(?:_level|_l)(\d+)(?:\D|$)",
        normalized,
        re.IGNORECASE,
    )
    return bool(match and int(match.group(1)) != level)


def _validate_expected_wip_attempt(
    game: str, level: int, expected_attempt: str, tag: str = ""
) -> dict:
    """Reopen one scheduler-selected exact-frontier WIP capsule.

    The status reducer validates the sealed pointer and complete inventory;
    this proposer-side gate additionally applies the current taint policy just
    before any capsule byte enters a fresh workspace.
    """
    if (
        not isinstance(expected_attempt, str)
        or not expected_attempt
        or Path(expected_attempt).name != expected_attempt
    ):
        raise ValueError("expected WIP attempt is not one path component")
    art = Path(artifact_dir(game, tag))
    parent = _load_checkpoint(os.fspath(art))
    reached = parent.reached if parent is not None and parent.validated else 0
    binding = CCS.exact_frontier_binding(
        art, game=game, target_level=level
    )
    descriptor = CCS.latest_wip_descriptor(
        art,
        game=game,
        reached=reached,
        target_level=level,
        frontier_binding=binding,
    )
    if descriptor.get("warm_wip_available") is not True:
        raise ValueError(
            "scheduler-selected WIP is no longer eligible: "
            f"{descriptor.get('warm_wip_validation')}"
        )
    if descriptor.get("warm_wip_attempt") != expected_attempt:
        raise ValueError(
            "scheduler-selected WIP pointer changed before dispatch"
        )
    files_dir = art / "wip_context" / f"level_{level:02d}" / (
        expected_attempt
    ) / "files"
    metadata_path = files_dir.parent / "metadata.json"
    if not _wip_uses_current_boundary_policy(metadata_path):
        raise WorkspaceTainted(
            "scheduler-selected WIP predates the current filesystem boundary "
            "policy and is forensic-only"
        )
    taint_reason = _workspace_taint_reason(os.fspath(files_dir))
    if taint_reason:
        raise WorkspaceTainted(
            "scheduler-selected WIP fails the current taint policy: "
            f"{taint_reason}"
        )
    return descriptor


def _wip_uses_current_boundary_policy(metadata_path: Path) -> bool:
    try:
        metadata = json.loads(
            _read_single_link_regular(os.fspath(metadata_path)).decode("utf-8")
        )
        if not isinstance(metadata, dict):
            return False
        return (
            metadata.get("filesystem_boundary_policy_schema")
            == APB.POLICY_SCHEMA
            and metadata.get("filesystem_boundary_policy_sha256")
            == APB.policy_sha256()
            and metadata.get("compatibility_arena_module_sha256")
            == _LOADED_ARENA_MODULE_SHA256
            and metadata.get("compatibility_boundary_authority")
            == "behavioral_defense_in_depth"
        )
    except (OSError, UnicodeError, json.JSONDecodeError, RuntimeError,
            WorkspaceTainted):
        return False


def _restore_wip_probes(game: str, ws: str, level: int, tag: str = "",
                        verbose: bool = True,
                        expected_attempt: Optional[str] = None) -> int:
    """Copy the latest WIP snapshot's probe context for `level` into the workspace.

    Restores the latest snapshot's non-promoted files and NEVER the promoted names
    (those are unverified candidates in a snapshot; the artifact root is the
    verified source of truth). Stale scratch from an older attempt must not mask
    the coherent latest WIP context, so the latest snapshot overwrites scratch
    files that are older than its copies (snapshots preserve mtimes via copy2) --
    but scratch modified AFTER the latest snapshot is live WIP from a run that
    died before snapshotting, and is never clobbered. Backfill snapshots only
    fill gaps. This puts earlier probe scripts and the prior proposer transcript
    back on disk where the next proposer can find them itself -- context lives in
    the filesystem, never stitched into the prompt.
    """
    level_dir = _wip_level_dir(artifact_dir(game, tag), level)
    latest_path = os.path.join(level_dir, "latest.json")
    if expected_attempt is not None:
        _validate_expected_wip_attempt(
            game, level, expected_attempt, tag
        )
    if not os.path.exists(latest_path):
        if expected_attempt is not None:
            raise ValueError("scheduler-selected WIP pointer disappeared")
        return 0
    try:
        with open(latest_path) as f:
            latest_attempt = json.load(f).get("attempt")
    except Exception:
        if expected_attempt is not None:
            raise ValueError("scheduler-selected WIP pointer is unreadable")
        return 0
    if expected_attempt is not None and latest_attempt != expected_attempt:
        raise ValueError(
            "scheduler-selected WIP pointer changed before restore"
        )

    attempts = []
    for attempt in sorted(os.listdir(level_dir)):
        if expected_attempt is not None and attempt != expected_attempt:
            continue
        attempt_dir = os.path.join(level_dir, attempt)
        files_dir = os.path.join(attempt_dir, "files")
        meta_path = os.path.join(attempt_dir, "metadata.json")
        if not os.path.isdir(files_dir):
            continue
        created = ""
        metadata = None
        try:
            with open(meta_path) as f:
                metadata = json.load(f)
                created = metadata.get("created_at", "")
        except Exception:
            pass
        if not _wip_uses_current_boundary_policy(Path(meta_path)):
            if expected_attempt is not None:
                raise WorkspaceTainted(
                    "scheduler-selected WIP is not bound to the current "
                    "filesystem boundary policy"
                )
            continue
        attempts.append((attempt == latest_attempt, created, attempt, files_dir))
    if not attempts:
        if expected_attempt is not None:
            raise ValueError("scheduler-selected WIP capsule disappeared")
        return 0
    attempts.sort(key=lambda t: t[1], reverse=True)
    attempts.sort(key=lambda t: not t[0])
    # Harness-generated templates must come from the current runner, not an old
    # WIP copy.  Restore only agent-authored probes/context around them.
    skip = set(PROMOTED_FILES) | {
        "gkm_try.py", "perception.py", "solver_index.md", "frontier_brief.md",
    }
    restored = 0
    latest_done = False
    for is_latest, _, attempt, files_dir in attempts:
        # Re-evaluate every candidate with the *current* scanner. A snapshot
        # that passed an older policy can become inadmissible after the taint
        # rules are tightened; restoring even one of its files would invalidate
        # the new turn before the proposer starts. If the latest snapshot is
        # now tainted, continue newest-first to the most recent clean
        # same-frontier fallback.
        taint_reason = _workspace_taint_reason(files_dir)
        if taint_reason:
            if expected_attempt is not None:
                raise WorkspaceTainted(
                    "scheduler-selected WIP became tainted before restore: "
                    f"{taint_reason}"
                )
            if verbose:
                print(
                    "skipping tainted WIP snapshot "
                    f"{attempt}: {taint_reason}"
                )
            continue
        snapshot_files = []
        for root, dirs, names in os.walk(files_dir):
            dirs[:] = sorted(
                d for d in dirs
                if d not in SNAPSHOT_SKIP_DIRS
                and not os.path.islink(os.path.join(root, d))
            )
            for name in sorted(names):
                src = os.path.join(root, name)
                if os.path.isfile(src) and not os.path.islink(src):
                    snapshot_files.append(os.path.relpath(src, files_dir))
        planned: List[Tuple[str, str, bytes]] = []
        try:
            for name in sorted(snapshot_files):
                if name in skip:
                    continue
                if _wip_path_targets_other_level(name, level):
                    continue
                src = os.path.join(files_dir, name)
                dst = os.path.join(ws, name)
                if os.path.exists(dst):
                    if latest_done:
                        continue
                    if os.path.getmtime(dst) >= os.path.getmtime(src):
                        continue
                planned.append((name, dst, _boundary_checked_payload(src, name)))
        except (OSError, WorkspaceTainted) as exc:
            if expected_attempt is not None:
                raise WorkspaceTainted(
                    "scheduler-selected WIP changed during restore: "
                    f"{exc}"
                ) from exc
            if verbose:
                print(
                    "skipping WIP snapshot that changed during restore: "
                    f"{attempt}: {exc}"
                )
            continue
        for name, dst, payload in planned:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            _atomic_host_write(dst, payload)
            restored += 1
        latest_done = True
    # A reviewed scaffold is a level-scoped intervention created after the prior
    # attempts.  It is deliberately outside immutable attempt snapshots, but is
    # copied into the clean room and included in the generated frontier brief.
    scaffold = os.path.join(level_dir, "frontier_scaffold.json")
    if os.path.isfile(scaffold):
        scaffold_taint = _file_taint_reason(scaffold, "frontier_scaffold.json")
        if scaffold_taint:
            if verbose:
                print(
                    "skipping tainted reviewed frontier scaffold: "
                    f"{scaffold_taint}"
                )
        else:
            dst = os.path.join(ws, "frontier_scaffold.json")
            if (
                not os.path.exists(dst)
                or os.path.getmtime(dst) < os.path.getmtime(scaffold)
            ):
                _copy_boundary_checked(
                    scaffold, dst, "frontier_scaffold.json"
                )
                restored += 1
    if verbose and restored:
        print(f"restored {restored} WIP probe file(s) for level {level} "
              f"from latest/backfill snapshots")
    if expected_attempt is not None and restored == 0:
        raise ValueError(
            "scheduler-selected WIP contained no restorable context"
        )
    return restored


def seed_workspace_from_artifact(game: str, ws: str, tag: str = "", verbose: bool = True,
                                 restore_wip: bool = True,
                                 expected_wip_attempt: Optional[str] = None) -> Optional[Report]:
    """Overwrite scratch with the latest promoted verified state, if one exists.

    Scratch is treated as disposable and possibly contaminated by an unfinished next
    level. The repo artifact is the source of truth for resuming. Unverified probe
    context for the NEXT level can be restored alongside (fill-gaps-only), so an
    interrupted attempt's probes survive scratch loss without contaminating the
    verified files. Set restore_wip=False for a clean continuation that retains
    only the verified Kolmogorov-Schmidhuber backbone.
    """
    art = artifact_dir(game, tag)
    artifact_taint = promoted_artifact_taint_reason(art)
    if artifact_taint:
        raise WorkspaceTainted(
            "refusing to seed from artifact outside the current boundary: "
            f"{artifact_taint}"
        )
    rep = _load_checkpoint(art)
    if rep is None or not rep.validated:
        if restore_wip:
            _restore_wip_probes(
                game, ws, 1, tag, verbose=verbose,
                expected_attempt=expected_wip_attempt,
            )
        return None
    for name in PROMOTED_FILES:
        src = os.path.join(art, name)
        if os.path.exists(src):
            _copy_boundary_checked(src, os.path.join(ws, name), name)
    if restore_wip:
        _restore_wip_probes(
            game, ws, rep.reached + 1, tag, verbose=verbose,
            expected_attempt=expected_wip_attempt,
        )
    if verbose:
        print(f"seeded workspace from artifact: {art} (reached={rep.reached})")
    return rep


def exact_level_boundary(game: str, path: Sequence, expected: int) -> Optional[List]:
    """Return the shortest replay prefix that first reaches ``expected``.

    Verifiers return every action the program committed, including actions taken
    after a newly won level while attempting the next one.  A promoted checkpoint
    must end at the exact acquisition boundary; otherwise resuming from it can
    replay an exhausted move budget and make the next frontier impossible.
    """
    if expected <= 0:
        return []
    if not path:
        return None
    try:
        env = A.Arena(game)
    except Exception:
        # Injectable unit-test games may not have a real Arena. Their verifier
        # remains the authority, but production games always take the exact path.
        return list(path)
    for index, action in enumerate(path, 1):
        if env.terminal():
            break
        env.step(action)
        if env.levels_completed >= expected:
            return list(path[:index])
    return None


def _stage_and_replay_winning_tree(
    game: str,
    ws: str,
    level: int,
    verify_fn: Callable,
) -> tuple[int, List, Dict[str, bytes]]:
    """Replay an immutable host-staged tree and return its promotable bytes."""

    assert_workspace_not_tainted(ws)
    payloads = {
        name: _boundary_checked_payload(os.path.join(ws, name), name)
        for name in _workspace_snapshot_files(ws)
    }
    required = {"legs.py", "players.py", "solve.py"}
    if not required.issubset(payloads):
        raise RuntimeError("winning solver tree is incomplete before promotion")
    parent = Path(ws).absolute().parent / ".promotion_replays"
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise WorkspaceTainted(
            f"promotion replay root is not a physical host directory: {parent}"
        )
    stage = Path(tempfile.mkdtemp(prefix="winning-", dir=parent))

    def assert_stage_clean() -> None:
        trusted = _trusted_host_scaffold_hashes(ws)
        findings = APB.scan_workspace(
            stage,
            arena_module_root=Path(__file__).resolve().parent,
            trusted_host_scaffolds=trusted,
        )
        reason = APB.first_reason(
            _filter_trusted_scaffold_root_literal(
                stage, findings, trusted=trusted
            )
        ) or _workspace_marker_taint_reason(os.fspath(stage))
        if reason:
            raise WorkspaceTainted(
                f"host-staged winning tree is tainted: {reason}"
            )

    try:
        for name, payload in payloads.items():
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            _atomic_host_write(os.fspath(destination), payload)
        assert_stage_clean()
        levels, path, _ = verify_fn(game, os.fspath(stage / "solve.py"))
        # The verifier receives no authority to edit the staged source.  Reopen
        # every byte after execution before accepting its behavioral result.
        for name, payload in payloads.items():
            if _read_single_link_regular(os.fspath(stage / name)) != payload:
                raise WorkspaceTainted(
                    f"host-staged winning tree changed during replay: {name}"
                )
        assert_stage_clean()
        exact = exact_level_boundary(game, path, level)
        if exact is None or not A.validate(game, exact, level):
            raise RuntimeError(
                f"sealed solver tree did not replay to exact level {level}"
            )
        promoted = {
            name: payload
            for name, payload in payloads.items()
            if name in PROMOTED_FILES and name != CHECKPOINT_FILE
        }
        return levels, exact, promoted
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def promote_verified_artifact(
    game: str,
    ws: str,
    rep: Report,
    tag: str = "",
    verbose: bool = True,
    *,
    sealed_promoted_payloads: Optional[Dict[str, bytes]] = None,
    authorized_turn: Optional[dict] = None,
) -> bool:
    """Idempotently publish the latest replay-validated workspace state.

    Promotion is intentionally gated on replay validation. This prevents speculative
    edits for an unfinished next level from replacing the last known-good artifact.
    """
    if not rep.validated or rep.reached <= 0:
        return False
    # Promotion must not execute replay or inspect candidate files before the
    # current boundary has rejected aliases and parent/absolute capabilities.
    assert_workspace_not_tainted(ws)
    boundary = exact_level_boundary(game, rep.final_path, rep.reached)
    if boundary is None:
        if verbose:
            print(
                f"refused promotion: path does not replay to level {rep.reached}"
            )
        return False
    if len(boundary) < len(rep.final_path):
        if verbose:
            print(
                f"trimmed checkpoint to exact level-{rep.reached} boundary: "
                f"{len(rep.final_path)} -> {len(boundary)} actions"
            )
        rep.final_path = boundary
    try:
        rep.validated = A.validate(game, rep.final_path, rep.reached)
    except Exception:
        # Synthetic injected games used by the harness tests have no real Arena;
        # retain their injected verifier verdict. Authoritative games never use
        # this fallback.
        rep.validated = bool(rep.validated)
    if not rep.validated:
        if verbose:
            print(
                f"refused promotion: exact boundary failed independent replay "
                f"for level {rep.reached}"
            )
        return False
    assert_workspace_not_tainted(ws)
    art = artifact_dir(game, tag)
    old = _load_checkpoint(art)
    if old is not None and old.validated and old.reached > rep.reached:
        if verbose:
            print(f"kept artifact at level {old.reached}; current verified level {rep.reached} is older")
        return False
    promote_files = old is None or not old.validated or old.reached < rep.reached
    _save_checkpoint(ws, rep)
    if promote_files:
        if sealed_promoted_payloads is None:
            promoted_payloads = {
                name: _boundary_checked_payload(os.path.join(ws, name), name)
                for name in PROMOTED_FILES
                if os.path.exists(os.path.join(ws, name))
            }
        else:
            promoted_payloads = dict(sealed_promoted_payloads)
            promoted_payloads[CHECKPOINT_FILE] = _boundary_checked_payload(
                os.path.join(ws, CHECKPOINT_FILE), CHECKPOINT_FILE
            )
    else:
        src = os.path.join(ws, CHECKPOINT_FILE)
        promoted_payloads = (
            {CHECKPOINT_FILE: _boundary_checked_payload(src, CHECKPOINT_FILE)}
            if os.path.exists(src)
            else {}
        )
    # No artifact byte is changed until every source image has been sealed and
    # accepted under the post-replay boundary policy.
    os.makedirs(art, exist_ok=True)
    if promote_files:
        for name in PROMOTED_FILES:
            if name in promoted_payloads:
                _atomic_host_write(
                    os.path.join(art, name), promoted_payloads[name]
                )
        _write_promotion_evidence(
            game, ws, art, rep, authorized_turn=authorized_turn
        )
    else:
        # Same verified level: refresh metadata only. Scratch may contain
        # speculative next-level code, so do not overwrite clean solution files.
        if CHECKPOINT_FILE in promoted_payloads:
            _atomic_host_write(
                os.path.join(art, CHECKPOINT_FILE),
                promoted_payloads[CHECKPOINT_FILE],
            )
    with open(os.path.join(art, "README.md"), "w") as f:
        f.write(_artifact_readme(game, rep))
    with open(os.path.join(art, "run.log"), "w") as f:
        f.write(_artifact_run_log(game, rep))
    if verbose:
        action = "promoted verified artifact" if promote_files else "refreshed verified artifact metadata"
        print(f"{action}: {art} (reached={rep.reached})")
    return True


# ---------------------------------------------------------------------------
# workspace + real verifier (running the agent's solve.py on the real game)
# ---------------------------------------------------------------------------
TESTER = '''import importlib.util, json, os, sys
sys.path.insert(0, {labdir!r})
import gkm_legs as G
A = G.A
taint_reason = G._workspace_taint_reason(os.getcwd())
if taint_reason:
    raise SystemExit(f"TAINTED WORKSPACE: {{taint_reason}}")
spec = importlib.util.spec_from_file_location("solve", "solve.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
def resumed_solve(env):
    ck = None
    # Prefix optimization must replay the edited solver from level 1 without
    # mutating the supervisor-owned campaign checkpoint.
    if os.environ.get("GKM_FRESH_REPLAY") != "1" and os.path.exists("checkpoint.json"):
        with open("checkpoint.json") as f:
            ck = json.load(f)
    if ck and ck.get("game") == {game!r} and ck.get("validated") and ck.get("final_path"):
        for act in ck["final_path"]:
            env.step(act)
    m.solve(env)
levels, path, err = A.run_program({game!r}, resumed_solve)
ok = A.validate({game!r}, path, levels) if path else False
print(f"RESULT levels={{levels}} moves={{len(path)}} replay_ok={{ok}} err={{err}}")
'''

# Policy v1 was emitted before Arena compatibility loading moved behind the
# authenticated ``gkm_legs.A`` binding.  Long proposer turns launched under
# that policy retain authority only when their host-owned ``gkm_try.py`` is an
# exact rendering of these frozen bytes.  Keep this template immutable; a new
# historical policy requires a new explicitly versioned template.
_HOST_TESTER_POLICY_V1 = '''import importlib.util, json, os, sys
sys.path.insert(0, {labdir!r})
import gkm_legs as G
import gkm_arena as A
taint_reason = G._workspace_taint_reason(os.getcwd())
if taint_reason:
    raise SystemExit(f"TAINTED WORKSPACE: {{taint_reason}}")
spec = importlib.util.spec_from_file_location("solve", "solve.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
def resumed_solve(env):
    ck = None
    # Prefix optimization must replay the edited solver from level 1 without
    # mutating the supervisor-owned campaign checkpoint.
    if os.environ.get("GKM_FRESH_REPLAY") != "1" and os.path.exists("checkpoint.json"):
        with open("checkpoint.json") as f:
            ck = json.load(f)
    if ck and ck.get("game") == {game!r} and ck.get("validated") and ck.get("final_path"):
        for act in ck["final_path"]:
            env.step(act)
    m.solve(env)
levels, path, err = A.run_program({game!r}, resumed_solve)
ok = A.validate({game!r}, path, levels) if path else False
print(f"RESULT levels={{levels}} moves={{len(path)}} replay_ok={{ok}} err={{err}}")
'''

_TRUSTED_HOST_TESTER_TEMPLATES = (TESTER, _HOST_TESTER_POLICY_V1)

_SOLVER_IMPORT_LOCK = threading.RLock()


def _codex_boundary_reasons(
    boundary_monitor,
    transcript_path: Path,
    *,
    final: bool = False,
) -> tuple[str, ...]:
    """Return host-control and proposer-boundary failures for one poll.

    This single helper is used during live polling and at terminal sealing so
    neither path can omit the authenticated Arena control.
    """

    arena_control_reason = _compatibility_arena_control_reason()
    if arena_control_reason:
        return (arena_control_reason,)
    findings = (
        *boundary_monitor.scan_workspace(),
        *boundary_monitor.scan_transcript(transcript_path, final=final),
    )
    findings = _filter_trusted_scaffold_root_literal(
        boundary_monitor.workspace_root,
        findings,
        trusted=boundary_monitor.trusted_host_scaffolds,
    )
    return tuple(finding.describe() for finding in findings)


def _workspace_import_roots(wsdir: str) -> set[str]:
    """Return top-level module names that one solver workspace can provide."""
    roots = {"solve", "players", "legs"}
    try:
        entries = tuple(Path(wsdir).iterdir())
    except OSError:
        return roots
    for entry in entries:
        if entry.is_file() and entry.suffix == ".py":
            name = entry.stem
        elif entry.is_dir():
            # Include namespace-package directories too. Generated solvers may
            # import them even without an ``__init__.py``.
            name = entry.name
        else:
            continue
        if name.isidentifier():
            roots.add(name)
    return roots


def _module_is_below(module: object, directory: str) -> bool:
    origin = getattr(module, "__file__", None)
    if not isinstance(origin, str):
        return False
    try:
        return os.path.commonpath(
            (os.path.realpath(origin), directory)
        ) == directory
    except (OSError, ValueError):
        return False


def run_solve_file(
    game: str,
    solve_path: str,
    *,
    time_cap: int = 600,
    resume_checkpoint: bool = True,
):
    """Import and execute one generated solver in an isolated module scope.

    The workspace must be on ``sys.path`` so generated sibling imports resolve.
    Every top-level module the workspace can provide is shadowed for the complete
    execution and removed afterward. Without that boundary, an auxiliary module
    such as ``perception.py`` can leak from one game's replay into the next.
    """
    import sys
    wsdir = os.path.realpath(
        os.path.dirname(os.path.abspath(solve_path))
    )
    roots = _workspace_import_roots(wsdir)
    with _SOLVER_IMPORT_LOCK:
        shadowed = {
            name: module
            for name, module in tuple(sys.modules.items())
            if any(
                name == root or name.startswith(f"{root}.")
                for root in roots
            )
        }
        for name in shadowed:
            sys.modules.pop(name, None)
        added = wsdir not in sys.path
        if added:
            sys.path.insert(0, wsdir)
        try:
            spec = importlib.util.spec_from_file_location(
                "solve", solve_path
            )
            m = importlib.util.module_from_spec(spec)
            sys.modules["solve"] = m
            spec.loader.exec_module(m)
            ckpt = _load_checkpoint(wsdir)

            def resumed_solve(env):
                if (
                    resume_checkpoint
                    and ckpt
                    and ckpt.game == game
                    and ckpt.validated
                    and ckpt.final_path
                ):
                    for act in ckpt.final_path:
                        env.step(act)
                m.solve(env)

            return A.run_program(
                game, resumed_solve, time_cap=time_cap
            )
        finally:
            for name, module in tuple(sys.modules.items()):
                if (
                    any(
                        name == root or name.startswith(f"{root}.")
                        for root in roots
                    )
                    or _module_is_below(module, wsdir)
                ):
                    sys.modules.pop(name, None)
            sys.modules.update(shadowed)
            if added and wsdir in sys.path:
                sys.path.remove(wsdir)


def _candidate_path_files(ws: str, K: int) -> List[str]:
    """Return regular, unaliased path exports from this attempt only."""
    patterns = [
        os.path.join(ws, "base*.json"),
        os.path.join(ws, f"seg_L{K}.json"),
        os.path.join(ws, f"seg_{K}.json"),
        os.path.join(ws, f"*path*.json"),
        os.path.join(ws, f"*candidate*.json"),
        os.path.join(ws, f"*replay*.json"),
        os.path.join(ws, f"win{K}*.json"),
        os.path.join(ws, f"*win*{K}*.json"),
    ]
    out = []
    for pat in patterns:
        out.extend(glob.glob(pat))
    # Newer first, de-duped.
    seen = set()
    ordered = []
    workspace = os.path.realpath(ws)
    dated = []
    for path in out:
        try:
            dated.append((os.lstat(path).st_mtime, path))
        except OSError:
            continue
    level_token = re.compile(
        rf"(?:^|[^a-z0-9])(?:level|l|seg[_-]?l?)[_-]?0*{K}"
        r"(?:[^0-9]|$)",
        re.IGNORECASE,
    )
    # Current-frontier exports dominate old WIP probes.  Modification time is
    # only a tie-breaker; otherwise a recently touched L5 probe could displace
    # the actual L9 candidate under the bounded recovery search.
    dated.sort(
        key=lambda item: (
            bool(level_token.search(os.path.basename(item[1]))),
            item[0],
        ),
        reverse=True,
    )
    for _mtime, path in dated:
        rp = os.path.realpath(path)
        try:
            metadata = os.lstat(path)
            contained = os.path.commonpath((workspace, rp)) == workspace
        except (OSError, ValueError):
            continue
        if (
            not contained
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or rp in seen
        ):
            continue
        seen.add(rp)
        ordered.append(path)
        if len(ordered) >= MAX_RECOVERY_PATH_CANDIDATES:
            break
    return ordered


def _load_action_path(value) -> Optional[list]:
    """Normalize JSON/log candidates into replayable key or coordinate actions."""
    if isinstance(value, dict):
        for key in ("path", "actions", "win", "solution", "final_path"):
            if key in value:
                value = value[key]
                break
    if isinstance(value, list) and value:
        normalized = []
        for action in value:
            if (
                isinstance(action, int)
                and not isinstance(action, bool)
                and 1 <= action <= 7
                and action != 6
            ):
                normalized.append(action)
            elif _valid_replay_action(action):
                normalized.append([6, action[1], action[2]])
            else:
                return None
        return normalized
    return None


def _action_path_key(path) -> tuple:
    """Hashable cache key for integer and ``[6, x, y]`` replay tokens."""
    return tuple(tuple(action) if isinstance(action, (list, tuple)) else action
                 for action in path)


def _candidate_paths_from_log(ws: str) -> List[List[int]]:
    txt = _read(os.path.join(ws, "proposer_last.log"))
    paths = []
    for m in re.finditer(r"(?:WIN|PATH)\s+(\[[^\]\n]{3,20000}\])", txt):
        try:
            path = _load_action_path(ast.literal_eval(m.group(1)))
        except (SyntaxError, ValueError):
            path = None
        if path:
            paths.append(path)
    return paths


def _candidate_paths_from_checkpoint(ws: str) -> List[List[int]]:
    """Treat a proposer-mutated checkpoint only as an untrusted path artifact."""
    try:
        data = json.load(open(os.path.join(ws, CHECKPOINT_FILE)))
    except (OSError, json.JSONDecodeError):
        return []
    path = _load_action_path(data.get("final_path") if isinstance(data, dict) else None)
    return [path] if path else []


def _verify_candidate_suffix(game: str, prefix: List[int], suffix: List[int], K: int):
    """Return combined replay path if suffix advances from prefix to at least K."""
    try:
        env = A.Arena(game)
        if hasattr(env, "reset"):
            env.reset()
        for a in prefix or []:
            if env.terminal():
                return None
            env.step(a)
        base = env.levels_completed
        for a in suffix:
            if env.terminal():
                return None
            env.step(a)
        if base >= K:
            return None
        combined = list(prefix or []) + list(suffix)
        if env.levels_completed >= K:
            levels, path, err = A.run_program(game, lambda e: [e.step(a) for a in combined])
            if levels >= K and not err and A.validate(game, path, levels):
                return path, levels
    except Exception:
        return None
    return None


def _run_candidate_replay(game: str, path: List[int]):
    try:
        return A.run_program(game, lambda e: [e.step(a) for a in path])
    except Exception as ex:
        return 0, [], f"{type(ex).__name__}: {ex}"


def _validated_prefix_floor(game: str, path: List[int], floor: int) -> bool:
    levels, replay_path, err = _run_candidate_replay(game, path)
    return levels >= floor and not err and A.validate(game, replay_path, levels)


def _record_failed_glue_context(ws: str, K: int, prefix_source: str, suffix_source: str,
                                prefix_len: int, suffix_len: int,
                                levels: int, moves: int, err) -> None:
    note = os.path.join(ws, "wip_glue_notes.md")
    line = (
        f"- L{K}: direct replay of `{prefix_source}` + `{suffix_source}` failed: "
        f"prefix_len={prefix_len}, suffix_len={suffix_len}, "
        f"observed_levels={levels}, observed_moves={moves}, err={err}. "
        "Treat these as potentially cofibrant pieces, not as a proven composition; "
        "a bridge/morphism may be needed or the suffix may need rederivation.\n"
    )
    old = _read(note)
    if line not in old:
        with open(note, "a") as f:
            f.write(line)


def _verify_candidate_path(game: str, prefix: List[int], candidate: List[int], K: int):
    """Verify either a level suffix or a full replay path from a proposer artifact."""
    verified = _verify_candidate_suffix(game, prefix, candidate, K)
    if verified is not None:
        combined, reached = verified
        return combined, reached, list(candidate)
    if prefix and len(candidate) > len(prefix) and candidate[:len(prefix)] == list(prefix):
        try:
            levels, path, err = _run_candidate_replay(game, candidate)
            if levels >= K and not err and A.validate(game, path, levels):
                return list(path), levels, list(candidate[len(prefix):])
        except Exception:
            return None
    return None


def _install_literal_player(ws: str, K: int, suffix: List[int], source: str) -> None:
    """Install a verified discovered path as a thin player composition."""
    players_p = os.path.join(ws, "players.py")
    players = _read(players_p)
    block = (
        f"\n\ndef play_level_{K}(env):\n"
        f"    # Recovered from verified proposer path artifact: {source}\n"
        f"    for action in {suffix!r}:\n"
        f"        env.step(action)\n"
    )
    pat = re.compile(rf"\n\ndef play_level_{K}\(env\):\n.*?(?=\n\ndef play_level_\d+\(env\):|\Z)", re.S)
    if pat.search(players):
        players = pat.sub(block, players, count=1)
    else:
        players = players.rstrip() + block
    with open(players_p, "w") as f:
        f.write(players.rstrip() + "\n")
    with open(os.path.join(ws, "legs_log.md"), "a") as f:
        f.write(
            f"\n## Level {K}: recovered verified path artifact\n\n"
            f"The proposer found a winning suffix but did not integrate it before "
            f"the time budget ended. Harness recovery validated `{source}` and "
            f"installed a thin replay player for the recovered suffix.\n"
        )


def recover_discovered_path_artifact(game: str, ws: str, K: int, prefix: List[int],
                                     verbose: bool = True):
    """Validate and install any proposer-discovered path artifact for level K."""
    # Recovery is an admission boundary, not a forensic parser: never inspect
    # a candidate path or transcript from a workspace that fails the current
    # capability gate.
    assert_workspace_not_tainted(ws)
    raw_candidates = []
    raw_candidates.extend(
        (p, "proposer_last.log")
        for p in _candidate_paths_from_log(ws)
    )
    raw_candidates.extend(
        (p, CHECKPOINT_FILE)
        for p in _candidate_paths_from_checkpoint(ws)
    )
    for path in _candidate_path_files(ws, K):
        try:
            value = json.load(open(path))
        except (OSError, json.JSONDecodeError):
            continue
        suffix = _load_action_path(value)
        if suffix:
            raw_candidates.append((suffix, path))
    candidates = []
    seen_candidate_paths = set()
    for candidate, source in raw_candidates:
        key = _action_path_key(candidate)
        if key in seen_candidate_paths:
            continue
        seen_candidate_paths.add(key)
        candidates.append((candidate, source))
        if len(candidates) >= MAX_RECOVERY_PATH_CANDIDATES:
            break

    # A proposer may legitimately optimize earlier players and export a new
    # full L1..K path that no longer begins with the old host checkpoint.  This
    # is not a suffix and must never be glued to the stale prefix.  Admit it
    # only when both independent artifacts agree operationally: the exported
    # path replays from zero, and the current workspace source separately
    # replays from zero to the same frontier.  The harness then adopts the
    # source-produced exact path; it does not install a literal player or trust
    # proposer-owned checkpoint/accounting fields.
    for candidate, source in candidates:
        candidate_levels, candidate_path, candidate_err = (
            _run_candidate_replay(game, candidate)
        )
        if (
            candidate_levels < K
            or candidate_err
            or not candidate_path
            or not A.validate(game, candidate_path, candidate_levels)
        ):
            continue
        source_levels, source_path, source_err = run_solve_file(
            game,
            os.path.join(ws, "solve.py"),
            resume_checkpoint=False,
        )
        if (
            source_levels < K
            or source_err
            or not source_path
            or not A.validate(game, source_path, source_levels)
        ):
            continue
        if verbose:
            print(
                f"level {K}: recovered replay-validated fresh-prefix "
                f"replacement from {source} "
                f"(candidate_len={len(candidate_path)} "
                f"source_len={len(source_path)} reached={source_levels})"
            )
        return source_levels, source_path, None

    # Some successful WIP states naturally factor the replay into a compressed
    # verified prefix plus a next-level suffix. Harvest both halves without
    # forcing the proposer to remember to rewrite checkpoint.json before timeout.
    failed_glues = 0
    prefix_ok = {}
    eligible_prefixes = 0
    glue_attempts = 0
    for prefix_path, prefix_source in candidates:
        key = _action_path_key(prefix_path)
        if key not in prefix_ok:
            prefix_ok[key] = _validated_prefix_floor(game, prefix_path, K - 1)
        if not prefix_ok[key]:
            continue
        eligible_prefixes += 1
        if eligible_prefixes > MAX_RECOVERY_PREFIX_CANDIDATES:
            break
        for suffix_path, suffix_source in candidates:
            if suffix_path is prefix_path:
                continue
            if glue_attempts >= MAX_RECOVERY_GLUE_ATTEMPTS:
                break
            glue_attempts += 1
            verified = _verify_candidate_suffix(game, prefix_path, suffix_path, K)
            if verified is None:
                direct = list(prefix_path) + list(suffix_path)
                levels, path, err = _run_candidate_replay(game, direct)
                if levels < K or err:
                    _record_failed_glue_context(
                        ws, K, prefix_source, suffix_source,
                        len(prefix_path), len(suffix_path),
                        levels, len(path or []), err)
                    failed_glues += 1
                continue
            combined, reached = verified
            suffix = list(suffix_path)
            _install_literal_player(ws, K, suffix, f"{prefix_source}+{suffix_source}")
            if verbose:
                print(f"level {K}: recovered verified joined path artifacts from "
                      f"{prefix_source}+{suffix_source} "
                      f"(prefix_len={len(prefix_path)} suffix_len={len(suffix)} "
                      f"reached={reached})")
            return reached, combined, None
        if glue_attempts >= MAX_RECOVERY_GLUE_ATTEMPTS:
            break
    if verbose and failed_glues:
        print(f"level {K}: recorded {failed_glues} failed direct WIP glue attempt(s) "
              "for proposer context")

    for candidate, source in candidates:
        verified = _verify_candidate_path(game, prefix, candidate, K)
        if verified is None:
            continue
        combined, reached, suffix = verified
        _install_literal_player(ws, K, suffix, source)
        if verbose:
            print(f"level {K}: recovered verified path artifact from {source} "
                  f"(suffix_len={len(suffix)} reached={reached})")
        return reached, combined, None
    return None


def _read(path: str) -> str:
    return open(path).read() if os.path.exists(path) else ""


def _try_auto_solve(K: int, legs_code: str, players_code: str,
                    players_p: str, solve_p: str, game: str,
                    verify_fn) -> Optional[tuple]:
    """Try to solve level K using existing legs only.

    Appends a minimal ``play_level_K`` stub that calls the most general
    existing solver leg and runs the verifier.  If it succeeds the proposer
    is skipped entirely (marginal_C ~0 for the player stub).  If it fails
    the original ``players.py`` is restored and ``None`` is returned.

    Returns ``(levels, path, err)`` on success, ``None`` on failure.
    """
    import ast
    try:
        tree = ast.parse(legs_code)
    except SyntaxError:
        return None
    # Find the first public function whose name suggests it is a
    # general-purpose solver (takes env, all-default params).
    candidates = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name.startswith("_"):
            continue
        if not node.args.args:
            continue
        if node.name in ("normalized_frame_key", "replay_actions"):
            continue
        candidates.append(node.name)
    if not candidates:
        return None
    # Prefer names that suggest "solve the whole level"
    pref = [n for n in candidates
            if any(kw in n.lower() for kw in ("clear", "solve", "search", "bfs", "find", "level", "path"))]
    ordered = pref + [n for n in candidates if n not in pref]
    for name in ordered[:AUTO_SOLVE_MAX_CANDIDATES]:
        stub = f"\n\ndef play_level_{K}(env):\n    {name}(env)\n"
        with open(players_p, "a") as f:
            f.write(stub)
        if (
            verify_fn is run_solve_file
            or getattr(verify_fn, "_gkm_run_solve_file", False)
        ):
            lv, path, err = verify_fn(
                game, solve_p, time_cap=AUTO_SOLVE_CANDIDATE_SECONDS
            )
        else:
            lv, path, err = verify_fn(game, solve_p)
        if lv >= K:
            return lv, path, err
        with open(players_p, "w") as f:
            f.write(players_code)
    return None


PERCEPTION_SEED = '''"""Source-free frame perception helpers for cracking.

This module is deliberately observational: it derives compact symbolic state
from `env.frame()` and `env.clone()` only. It is a cofibration-style scaffold:
raw pixels are embedded into a monotone tower of reusable observations
(components -> objects -> action deltas -> replay states). Candidate level
logic should be written against these quotients, then replay-validated by the
harness. No game source or prior solution history is read here.
"""
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

UP, DOWN, LEFT, RIGHT, USE = 1, 2, 3, 4, 5
ACTIONS = (UP, DOWN, LEFT, RIGHT, USE)
DIRS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
ACTION_NAME = {UP: "UP", DOWN: "DOWN", LEFT: "LEFT", RIGHT: "RIGHT", USE: "USE"}


@dataclass(frozen=True)
class Blob:
    color: int
    bbox: Tuple[int, int, int, int]  # r0, c0, r1, c1 inclusive
    area: int
    centroid: Tuple[float, float]

    @property
    def top_left(self):
        return self.bbox[0], self.bbox[1]

    @property
    def size(self):
        r0, c0, r1, c1 = self.bbox
        return r1 - r0 + 1, c1 - c0 + 1


def arr(frame) -> np.ndarray:
    return np.asarray(frame)


def color_counts(frame) -> Dict[int, int]:
    vals, cnts = np.unique(arr(frame), return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, cnts)}


def connected_components(frame, colors: Optional[Iterable[int]] = None,
                         min_area: int = 1) -> List[Blob]:
    f = arr(frame)
    wanted = None if colors is None else {int(c) for c in colors}
    seen = np.zeros(f.shape, dtype=bool)
    out: List[Blob] = []
    rows, cols = f.shape[:2]
    for r in range(rows):
        for c in range(cols):
            if seen[r, c]:
                continue
            color = int(f[r, c])
            if wanted is not None and color not in wanted:
                seen[r, c] = True
                continue
            q = [(r, c)]
            seen[r, c] = True
            pts = []
            while q:
                x, y = q.pop()
                pts.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < rows and 0 <= ny < cols and not seen[nx, ny] and int(f[nx, ny]) == color:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            if len(pts) >= min_area:
                rs = [p[0] for p in pts]
                cs = [p[1] for p in pts]
                out.append(Blob(color, (min(rs), min(cs), max(rs), max(cs)),
                                len(pts), (sum(rs) / len(pts), sum(cs) / len(pts))))
    return sorted(out, key=lambda b: (b.color, b.bbox))


def block_signatures(frame, cell: int = 4) -> Dict[Tuple[int, int], Tuple[int, ...]]:
    """Partition a frame into fixed cells and return each cell's color signature."""
    f = arr(frame)
    out = {}
    for r in range(0, f.shape[0], cell):
        for c in range(0, f.shape[1], cell):
            out[(r // cell, c // cell)] = tuple(int(v) for v in sorted(np.unique(f[r:r+cell, c:c+cell])))
    return out


def object_candidates(frame, cell: int = 4, min_area: int = 4) -> List[dict]:
    """A compact, game-agnostic object list from color components and cell signatures."""
    f = arr(frame)
    blobs = connected_components(f, min_area=min_area)
    sigs = block_signatures(f, cell)
    objects = []
    for b in blobs:
        r0, c0, r1, c1 = b.bbox
        objects.append({
            "color": b.color,
            "bbox": b.bbox,
            "top_left": b.top_left,
            "size": b.size,
            "area": b.area,
            "centroid": b.centroid,
            "cell": (r0 // cell, c0 // cell),
            "cell_sig": sigs.get((r0 // cell, c0 // cell)),
        })
    return objects


def frame_delta(before, after) -> dict:
    a, b = arr(before), arr(after)
    ys, xs = np.where(a != b)
    if len(ys) == 0:
        return {"count": 0, "bbox": None, "samples": []}
    samples = [(int(y), int(x), int(a[y, x]), int(b[y, x])) for y, x in zip(ys[:80], xs[:80])]
    return {
        "count": int(len(ys)),
        "bbox": (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())),
        "samples": samples,
    }


def normalize_public_action(action):
    """Validate one public key or coordinate action without touching ``env``."""
    if isinstance(action, (list, tuple)):
        if (
            len(action) != 3
            or action[0] != 6
            or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in action
            )
        ):
            raise ValueError(
                "coordinate action must be (6, x, y) with integer x,y in 0..63"
            )
        normalized = (6, int(action[1]), int(action[2]))
        if not (0 <= normalized[1] < 64 and 0 <= normalized[2] < 64):
            raise ValueError(
                "coordinate action must be (6, x, y) with integer x,y in 0..63"
            )
        return normalized
    if (
        not isinstance(action, int)
        or isinstance(action, bool)
        or action not in range(1, 8)
    ):
        raise ValueError("key action must be an integer in 1..7")
    if action == 6:
        raise ValueError(
            "bare ACTION6 is invalid; use (6, x, y) with integer x,y in 0..63"
        )
    return int(action)


def safe_step(env, action):
    """Validate locally, then apply one public key or coordinate action."""
    normalized = normalize_public_action(action)
    if isinstance(normalized, tuple):
        env.step(*normalized)
    else:
        env.step(normalized)
    return normalized


def action_deltas(env, actions=None) -> Dict[object, dict]:
    """Compare valid cloned actions; bare ACTION6 is never sent to the arena."""
    if actions is None:
        actions = tuple(action for action in env.actions if action != 6)
    normalized_actions = tuple(
        normalize_public_action(action) for action in actions
    )
    base = arr(env.frame()).copy()
    out = {}
    for action in normalized_actions:
        clone = env.clone()
        safe_step(clone, action)
        out[action] = frame_delta(base, clone.frame())
    return out


def replay(env, actions: Sequence):
    normalized_actions = tuple(
        normalize_public_action(action) for action in actions
    )
    clone = env.clone()
    for action in normalized_actions:
        if clone.terminal():
            break
        safe_step(clone, action)
    return clone


def path_result(env, actions: Sequence[int]) -> dict:
    clone = replay(env, actions)
    return {
        "levels_completed": int(clone.levels_completed),
        "terminal": bool(clone.terminal()),
        "path_len": len(actions),
        "colors": color_counts(clone.frame()),
        "objects": object_candidates(clone.frame()),
    }


def changed_signature(env, actions: Sequence[int], cell: int = 4):
    before = block_signatures(env.frame(), cell)
    clone = replay(env, actions)
    after = block_signatures(clone.frame(), cell)
    return {k: (before.get(k), after.get(k)) for k in sorted(set(before) | set(after))
            if before.get(k) != after.get(k)}


def bounded_bfs(env, goal_fn, actions=None,
                key_fn=None, max_states: int = 20000, max_depth: int = 80):
    """BFS over advertised key actions by default; use small bounds first."""
    if actions is None:
        actions = tuple(action for action in env.actions if action != 6)
    normalized_actions = tuple(
        normalize_public_action(action) for action in actions
    )
    if key_fn is None:
        key_fn = lambda e: arr(e.frame()).tobytes()
    start_key = key_fn(env)
    q = deque([(env.clone(), [])])
    seen = {start_key}
    while q and len(seen) <= max_states:
        node, path = q.popleft()
        if goal_fn(node, path):
            return path
        if len(path) >= max_depth:
            continue
        for action in normalized_actions:
            child = node.clone()
            normalized = safe_step(child, action)
            key = key_fn(child)
            if key in seen:
                continue
            seen.add(key)
            q.append((child, path + [normalized]))
    return None


def bounded_replay_bfs(env, goal_fn, action_fn,
                       key_fn=None, max_states: int = 20000, max_depth: int = 80):
    """Path-only BFS for games whose deep Arena clones become expensive.

    The queue retains compact action paths, not recursively deep-copied runtime
    states. Each node is reconstructed from one root clone. ``action_fn(node)``
    may return integer actions or coordinate tuples such as ``(6, x, y)``.
    """
    if key_fn is None:
        key_fn = lambda e: arr(e.frame()).tobytes()

    def reconstruct(path):
        node = env.clone()
        for action in path:
            safe_step(node, action)
        return node

    start = reconstruct([])
    q = deque([[]])
    seen = {key_fn(start)}
    while q and len(seen) <= max_states:
        path = q.popleft()
        node = reconstruct(path)
        if goal_fn(node, path):
            return path
        if len(path) >= max_depth or node.terminal():
            continue
        actions = tuple(
            normalize_public_action(action)
            for action in action_fn(node)
        )
        for action in actions:
            child_path = path + [action]
            child = reconstruct(child_path)
            key = key_fn(child)
            if key in seen:
                continue
            seen.add(key)
            if goal_fn(child, child_path):
                return child_path
            q.append(child_path)
    return None


def level_goal(base_level: int):
    return lambda env, path: env.levels_completed > base_level
'''


def setup_workspace(
    game: str, tag: str = "", *, isolated_generation: bool = False
) -> str:
    suffix = f"_{tag}" if tag else ""
    if isolated_generation:
        os.makedirs(SCRATCH, exist_ok=True)
        ws = tempfile.mkdtemp(
            prefix=f"gkm_legs_ws_{game}{suffix}_", dir=SCRATCH
        )
    else:
        ws = os.path.join(SCRATCH, f"gkm_legs_ws_{game}{suffix}")
        os.makedirs(ws, exist_ok=True)
    labdir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(ws, "gkm_try.py"), "w") as fh:
        fh.write(TESTER.format(labdir=labdir, game=game))
    for name, seed in (
        ("legs.py", "# Shared leg library: small, named, reusable skills.\n"
                     "# Players import from here; add a NEW leg only when no existing leg fits.\n"),
        ("players.py", "# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.\n"
                       "from legs import *  # noqa\n"),
        ("solve.py", "import players\n\n"
                     "def solve(env):\n"
                     "    # dispatch to the per-level player for the current level, in a loop\n"
                     "    while not env.terminal():\n"
                     "        k = env.levels_completed + 1\n"
                     "        fn = getattr(players, f'play_level_{k}', None)\n"
                     "        if fn is None:\n"
                     "            return\n"
                     "        before = env.levels_completed\n"
                     "        fn(env)\n"
                     "        if env.levels_completed <= before:\n"
                     "            return  # no progress -> stop\n"),
        ("legs_log.md", "# Leg-library debrief log\n\nRecurring composition patterns and repeated novelty.\n"),
        ("perception.py", PERCEPTION_SEED),
    ):
        p = os.path.join(ws, name)
        if name == "perception.py" or not os.path.exists(p):
            with open(p, "w") as fh:
                fh.write(seed)
    return ws


def _solver_source_index(ws: str) -> str:
    """Return a compact navigational index without copying function bodies."""
    sections = [
        "# Generated solver source index",
        "",
        "Use line ranges to inspect only definitions relevant to the current level.",
    ]
    for name in ("players.py", "legs.py", "perception.py", "solve.py"):
        path = os.path.join(ws, name)
        source = _read(path)
        if not source:
            continue
        sections.extend(("", f"## {name}"))
        try:
            tree = ast.parse(source, filename=name)
        except SyntaxError as exc:
            sections.append(f"- parse error: {exc}")
            continue
        lines = source.splitlines()
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            first_body_line = node.body[0].lineno if getattr(node, "body", None) else node.end_lineno
            header = " ".join(
                line.strip()
                for line in lines[node.lineno - 1:max(node.lineno, first_body_line - 1)]
            )
            header = re.sub(r"\s+", " ", header)[:240]
            doc = (ast.get_docstring(node, clean=True) or "").splitlines()
            summary = re.sub(r"\s+", " ", doc[0]).strip()[:180] if doc else ""
            calls = sorted({
                child.func.id
                for child in ast.walk(node)
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
            })
            call_note = f"; calls: {', '.join(calls[:10])}" if calls else ""
            if len(calls) > 10:
                call_note += f", +{len(calls) - 10}"
            line_range = f"L{node.lineno}--{node.end_lineno or node.lineno}"
            description = f" — {summary}" if summary else ""
            sections.append(
                f"- {line_range} `{header}`{description}{call_note}"
            )
    return "\n".join(sections) + "\n"


def _write_solver_source_index(ws: str) -> str:
    path = os.path.join(ws, "solver_index.md")
    with open(path, "w") as handle:
        handle.write(_solver_source_index(ws))
    return path


def _frontier_brief(ws: str, game: str, level: int,
                    max_chars: int = 6000) -> str:
    """Distill prior clean WIP narration without copying bulky tool output.

    Codex JSONL command output often contains tens of thousands of pixels, source
    lines, and repeated probe states.  The agent's own progress messages are a much
    smaller index into that work.  They remain explicitly unverified hypotheses:
    the next proposer must reproduce any fact it relies on.
    """
    log = os.path.join(ws, "proposer_last.log")
    messages: List[str] = []
    if os.path.isfile(log):
        for raw in _read(log).splitlines():
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            item = event.get("item")
            if (
                event.get("type") == "item.completed"
                and isinstance(item, dict)
                and item.get("type") == "agent_message"
                and isinstance(item.get("text"), str)
            ):
                text = re.sub(r"\s+", " ", item["text"]).strip()
                if text:
                    messages.append(text)

    standard = set(PROMOTED_FILES) | {
        "gkm_try.py", "perception.py", "solver_index.md", "frontier_brief.md",
        "proposer_last.log", AUTO_SOLVE_LOG,
    }
    probe_files = [
        name for name in sorted(os.listdir(ws))
        if os.path.isfile(os.path.join(ws, name))
        and name not in standard
        and not name.startswith("codex_turn_")
        and name.endswith((".py", ".md", ".json", ".txt"))
    ]
    checkpoint = _load_checkpoint(ws)
    if not messages and not probe_files and not (
        checkpoint and checkpoint.validated and checkpoint.final_path
    ):
        return ""

    lines = [
        f"# Unverified frontier brief: {game} level {level}",
        "",
        "This is a compact index of the latest clean WIP, not solver evidence.",
        "Reproduce every observation you rely on with the documented local API.",
        "Do not reread the full proposer transcript unless a named ambiguity requires it.",
        "",
    ]
    if checkpoint and checkpoint.validated and checkpoint.final_path:
        step_cap = int(getattr(A, "DEFAULT_STEP_CAP", 600))
        used = len(checkpoint.final_path)
        lines.extend([
            "## Verified parent budget",
            "",
            f"- Exact parent boundary: level {checkpoint.reached} at {used} actions.",
            f"- Remaining real-action budget under the harness cap: "
            f"{max(0, step_cap - used)} of {step_cap}.",
            "- This budget is verifier evidence. If it is insufficient, optimize "
            "earlier composed legs rather than searching an uncommittable suffix.",
            "",
        ])
    if messages:
        lines.extend(["## Prior proposer progress", ""])
        used = 0
        for message in messages[-12:]:
            remaining = max_chars - used
            if remaining <= 0:
                break
            clipped = message[:remaining]
            lines.append(f"- {clipped}")
            used += len(clipped)
        lines.append("")
    if probe_files:
        lines.extend([
            "## Preserved local probes",
            "",
            *[
                f"- `{name}` ({os.path.getsize(os.path.join(ws, name))} bytes)"
                for name in probe_files
            ],
            "",
            "Run or inspect the smallest relevant probe before writing another one.",
            "",
        ])
    return "\n".join(lines)


def _write_frontier_brief(ws: str, game: str, level: int) -> Optional[str]:
    text = _frontier_brief(ws, game, level)
    path = os.path.join(ws, "frontier_brief.md")
    if not text:
        if os.path.exists(path):
            os.unlink(path)
        return None
    with open(path, "w") as handle:
        handle.write(text)
    return path


def _initialize_codex_workspace_git(ws: str) -> None:
    """Give Codex a repository boundary at the scratch root.

    Codex routinely runs ``git diff`` before finishing.  Without a local
    repository, Git walks upward into the real project and can expose unrelated
    parent metadata.  A tiny local baseline keeps every such read and diff
    confined to the clean-room workspace while still letting the agent inspect
    its own edits.
    """
    git_dir = os.path.join(ws, ".git")
    if os.path.lexists(git_dir):
        raise WorkspaceTainted(
            "refusing pre-existing Git metadata in fresh proposer workspace"
        )
    subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "init", "--quiet", ws],
        check=True,
    )
    subprocess.run(
        ["git", "-C", ws, "config", "user.name", "GKM clean-room harness"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", ws, "config", "user.email", "gkm-clean-room@invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", ws, "config", "core.hooksPath", "/dev/null"],
        check=True,
    )
    tracked = [
        name for name in (
            "gkm_try.py", "perception.py", "legs.py", "players.py", "solve.py",
            "legs_log.md", "solver_index.md", "frontier_brief.md",
            "frontier_scaffold.json", CHECKPOINT_FILE, AUTO_SOLVE_LOG,
        )
        if os.path.isfile(os.path.join(ws, name))
    ]
    if tracked:
        subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", "-C", ws,
             "add", "--", *tracked],
            check=True,
        )
        staged = subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", "-C", ws,
             "diff", "--cached", "--quiet"],
            check=False,
        )
        if staged.returncode == 1:
            subprocess.run(
                ["git", "-c", "core.hooksPath=/dev/null", "-C", ws,
                 "commit", "--quiet", "-m", "verified starting point"],
                check=True,
            )
        elif staged.returncode != 0:
            raise RuntimeError(f"could not inspect local Codex Git baseline in {ws}")
    subprocess.run(
        ["git", "-C", ws, "config", "status.showUntrackedFiles", "no"],
        check=True,
    )


# ---------------------------------------------------------------------------
# default proposer: the real Claude Code agent (tools) -- needs credits
# ---------------------------------------------------------------------------
# Provider failures are classified only from error payloads where the transport
# supplies them (Codex JSONL), never from arbitrary solver prose.  Keep these
# phrases specific: the old bare ``insufficient`` marker misclassified a frontier
# brief saying "if the move budget is insufficient" as account exhaustion.
_CREDIT_OUT_MARKERS = (
    "out of usage credits",
    "usage limit",
    "credit balance",
    "session limit reached",
    "insufficient credits",
    "insufficient quota",
    "quota exceeded",
    "quota has been exceeded",
    "not logged in",
    "please run /login",
    "spend limit reached",
    "usage-credits exhausted",
)


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


class CreditOut(RuntimeError):
    """Raised when the proposer subprocess reports it is out of credits/quota, so the
    orchestrator can stop the whole sequence cleanly instead of burning the budget."""


class ProposerInfrastructureError(RuntimeError):
    """A retryable provider/transport failure, not solver no-progress or quota."""


class ProposerEvidenceUnavailable(ProposerInfrastructureError):
    """A turn whose protected transcript cannot authorize reuse or promotion.

    This is deliberately *not* retryable within the same generation.  Source
    edits and probes may have been learned during the unrecorded turn, so the
    entire generation is quarantine-only even when its solver happens to replay.
    """


class ProposerContainmentTimeout(ProposerInfrastructureError):
    """A hard wall-time containment stop, never a clean solver no-progress result."""


class ProposerProtocolViolation(ProposerEvidenceUnavailable):
    """A public-action violation invalidated the complete proposer generation."""


class ProposerBoundaryViolation(ProposerEvidenceUnavailable):
    """A filesystem-capability violation invalidated the proposer generation."""


# markers of a transient infrastructure failure (dropped connection, server error):
# the proposer never worked on the level, so the attempt is retried, not judged.
_TRANSIENT_MARKERS = (
    "api error",
    "connection closed",
    "connection error",
    "connection refused",
    "overloaded",
    "internal server error",
    "service unavailable",
    "selected model is at capacity",
    "model is at capacity",
    "rate limit",
    "too many requests",
)

_TRANSIENT_RETRIES = 2
"""Extra proposer attempts per level when the failure looks infrastructural."""


def _classify_provider_error_message(message: str) -> str:
    """Return ``credit_out``, ``infrastructure``, or ``other`` for an error.

    Callers must pass a provider/CLI error payload, not an entire transcript.
    This prevents ordinary task text, probe output, or model reasoning from
    becoming control-plane signals.
    """
    blob = message.lower()
    if any(marker in blob for marker in _CREDIT_OUT_MARKERS):
        return "credit_out"
    if any(marker in blob for marker in _TRANSIENT_MARKERS):
        return "infrastructure"
    return "other"


def _codex_terminal_error_messages(log_path: str) -> list[str]:
    """Extract only top-level Codex JSONL terminal error payloads.

    Command output and assistant reasoning are deliberately ignored even when
    they contain words such as "quota", "capacity", or "insufficient".
    """
    messages: list[str] = []
    try:
        with open(log_path, encoding="utf-8") as handle:
            for raw in handle:
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                message = None
                if event.get("type") == "error":
                    message = event.get("message")
                elif event.get("type") == "turn.failed":
                    error = event.get("error")
                    if isinstance(error, dict):
                        message = error.get("message")
                if isinstance(message, str) and message and message not in messages:
                    messages.append(message)
    except OSError:
        return []
    return messages


def _usage_guard_error_is_cost_block(error: Exception) -> bool:
    """Whether a guard error is an explicit configured/provider cost stop.

    Lock contention, app-server timeouts, malformed rate-limit responses, and
    other guard transport failures are infrastructure. Only these specific
    finite-pool/campaign-cap messages are allowed to become ``CreditOut``.
    """
    message = str(error).lower()
    return any(marker in message for marker in (
        "weekly codex allowance is",
        "weekly codex allowance has only",
        "local campaign run cap reached",
        "local campaign token cap reached",
    ))


_SECRET_ENV_FILES = ("ANTHROPIC_API_KEY.env.local",)


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_secret_env_file(path: str) -> bool:
    """Load KEY=value secrets without printing them; existing env wins."""
    if not os.path.exists(path):
        return False
    loaded = False
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("'\"")
            else:
                key, value = "ANTHROPIC_API_KEY", line.strip().strip("'\"")
            if key and value and key not in os.environ:
                os.environ[key] = value
                loaded = True
    return loaded


def _ensure_anthropic_api_key() -> None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    for name in _SECRET_ENV_FILES:
        _load_secret_env_file(os.path.join(_repo_root(), name))


def _redact_secrets(text: str) -> str:
    out = text
    for key, value in os.environ.items():
        if value and ("KEY" in key or "TOKEN" in key or "SECRET" in key):
            out = out.replace(value, "[REDACTED]")
    out = re.sub(r"(ANTHROPIC_API_KEY\s*=\s*)\S+", r"\1[REDACTED]", out)
    return out


def _transient_proposer_failure(ws: str, code_changed: bool = True) -> bool:
    """True when proposer_last.log shows an aborted run rather than real work.

    A genuine capability failure leaves a substantial transcript; an aborted one
    leaves a short log -- an error banner (dropped connection, server error) or a
    sign-off with no work behind it (e.g. an agent that backgrounded its probe and
    ended its turn expecting a wakeup that headless mode never delivers). Requiring
    a short log avoids retrying a real hour-long attempt that happened to mention a
    transient API blip along the way."""
    txt = _read(os.path.join(ws, "proposer_last.log"))
    if len(txt) >= 2000:
        return False
    blob = txt.lower()
    if any(m in blob for m in _TRANSIENT_MARKERS):
        return True
    return not code_changed  # said little AND wrote nothing: no real attempt was made


# The headless `claude -p` proposer must run as a standalone SUBSCRIPTION session, not
# billed against API/Console dollars (a separate org spend cap unrelated to the
# subscription's session/weekly allowance).  Two things route it to the wrong pool: an
# API key in the environment, and the CLAUDE_CODE_* variables that mark this process as
# a child of the parent Claude Code session.  Strip both so the CLI authenticates with
# the logged-in subscription, mirroring how `_codex_environment` isolates Codex.
_CLAUDE_STRIP_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL")
_CLAUDE_STRIP_ENV_PREFIXES = ("CLAUDE_CODE_",)


def _claude_subscription_env() -> dict:
    return {
        key: value for key, value in os.environ.items()
        if key not in _CLAUDE_STRIP_ENV
        and not any(key.startswith(prefix) for prefix in _CLAUDE_STRIP_ENV_PREFIXES)
    }


def _claude_agent(ws: str, task: str, model: Optional[str], minutes: int, *,
                  guard: bool = False,
                  ledger_path: Optional[str] = None,
                  window_hours: float = CLG.DEFAULT_WINDOW_HOURS,
                  max_turns: Optional[int] = None,
                  max_wall_minutes: Optional[float] = None,
                  max_output_tokens: Optional[int] = None,
                  max_cost_usd: Optional[float] = None,
                  run_label: Optional[str] = None,
                  game: Optional[str] = None,
                  target_level: Optional[int] = None) -> None:
    """Run one headless Claude Code proposer turn.

    Unlike Codex, the Claude subscription exposes no readable remaining allowance,
    so ``guard=True`` enforces a LOCAL budget only: a serialized ledger of observed
    per-turn cost (wall time plus tokens/dollars from ``--output-format json``) with
    cumulative per-window caps.  There is no live provider read; reactive credit-out
    still aborts the sequence.  ``guard=False`` keeps the original unmetered behavior.
    """
    labdir = os.path.dirname(os.path.abspath(__file__))
    # JSON output lets us meter observed usage; we still persist the human-readable
    # result text (not the JSON envelope) so the taint gate and path-artifact
    # extraction keep operating on plain text exactly as before.
    cmd = ["claude", "-p", task, "--allowedTools", "Bash", "Read", "Write", "Edit",
           "--dangerously-skip-permissions", "--add-dir", labdir, "--output-format", "json"]
    if model:
        cmd += ["--model", model]

    ledger = (ledger_path or os.fspath(CLG.DEFAULT_LEDGER)) if guard else None
    caps = CLG.WindowCaps(
        max_turns=max_turns, max_output_tokens=max_output_tokens,
        max_wall_minutes=max_wall_minutes, max_cost_usd=max_cost_usd,
    ) if guard else None
    lock = None
    if guard:
        try:
            lock = CLG.campaign_lock(ledger)
            lock.__enter__()
            CLG.preflight(caps=caps, window_hours=window_hours, ledger_path=ledger)
        except CLG.ClaudeUsageGuardError as exc:
            if lock is not None:
                lock.__exit__(None, None, None)
            raise CreditOut(f"Claude campaign guard stopped the run: {exc}") from exc

    try:
        started = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        out = err = ""
        timed_out = False
        try:
            r = subprocess.run(cmd, cwd=ws, capture_output=True, text=True,
                               timeout=minutes * 60,
                               env=_claude_subscription_env())
            out, err = r.stdout or "", r.stderr or ""
        except subprocess.TimeoutExpired as ex:
            # Out of the per-level time budget. Whatever the agent already wrote to
            # the workspace (legs.py/players.py) persists; verify that partial work
            # instead of crashing the whole run.
            timed_out = True
            out = (ex.stdout or b"").decode("utf-8", "replace") if isinstance(ex.stdout, bytes) else (ex.stdout or "")
            err = (ex.stderr or b"").decode("utf-8", "replace") if isinstance(ex.stderr, bytes) else (ex.stderr or "")
            print(f"[proposer hit {minutes}min budget; verifying partial work]")
        duration = round(time.monotonic() - started, 3)
        usage = CLG.parse_claude_json_usage(out)
        with open(os.path.join(ws, "proposer_last.log"), "w") as fh:
            fh.write(usage["result_text"] + ("\n--- STDERR ---\n" + err if err else ""))
        blob = (out + " " + err).lower()
        credit_out = any(m in blob for m in _CREDIT_OUT_MARKERS)
        if guard:
            CLG.append_ledger({
                "event": "claude_exec",
                "started_at": started_at,
                "duration_seconds": duration,
                "run_label": run_label,
                "workspace": os.path.basename(os.path.abspath(ws)),
                "proposer": "claude",
                "billing_pool": "subscription",  # sanitized env -> Team subscription
                "model": model or "default",
                "minutes_limit": minutes,
                "timed_out": timed_out,
                "credit_out": credit_out,
                "game": game,
                "target_level": target_level,
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "total_cost_usd": usage["total_cost_usd"],
                "num_turns": usage["num_turns"],
                "usage_reported": usage["usage_reported"],
            }, ledger)
        if credit_out:
            raise CreditOut(f"proposer reported no credits/quota (see {ws}/proposer_last.log)")
    finally:
        if lock is not None:
            lock.__exit__(None, None, None)


def _opencode_agent(ws: str, task: str, model: Optional[str], minutes: int) -> None:
    """Run the opencode agent headlessly as the proposer.

    Pipes the task via stdin (avoids CLI arg length/encoding issues). Starts
    a fresh session each call. Cross-run resume is handled at the orchestrate
    level (checkpoint + persistent workspace files), so no ``--continue``
    flag is used here.
    """
    cmd = ["opencode", "run", "--auto", "--dir", ws]
    if model:
        cmd += ["-m", model]
    # Inject permission overrides so subagents don't get stuck on external_directory prompts.
    # NO agent may read game source code or prior solutions — they must discover
    # mechanics purely by experiment on clones, not by reading the implementation.
    env = {**os.environ, "OPENCODE_CONFIG_CONTENT": json.dumps({
        "$schema": "https://opencode.ai/config.json",
        "permission": {"external_directory": {
            "*": "allow",
            "**/environment_files/**": "deny",
            "**/agent_solutions/**": "deny",
            "**/FINDINGS.md": "deny",
        }},
    })}
    out = err = ""
    try:
        r = subprocess.run(cmd, cwd=ws, capture_output=True, text=True,
                           timeout=minutes * 60, input=task, env=env)
        out, err = r.stdout or "", r.stderr or ""
    except subprocess.TimeoutExpired as ex:
        out = (ex.stdout or b"").decode("utf-8", "replace") if isinstance(ex.stdout, bytes) else (ex.stdout or "")
        print(f"[opencode proposer hit {minutes}min budget; verifying partial work]")
    with open(os.path.join(ws, "proposer_last.log"), "w") as fh:
        fh.write(out + ("\n--- STDERR ---\n" + err if err else ""))
    blob = (out + " " + err).lower()
    if any(m in blob for m in _CREDIT_OUT_MARKERS):
        raise CreditOut(f"proposer reported no credits/quota (see {ws}/proposer_last.log)")


DEFAULT_CODEX_MODEL = "gpt-5.6-sol"
DEFAULT_CODEX_ALLOCATION_POLICY = "drain"
CODEX_REASONING_EFFORTS = {"medium", "high", "xhigh", "max"}


def _codex_command(ws: str, task: str, model: Optional[str],
                   reasoning_effort: str) -> list[str]:
    """Build a deterministic, noninteractive and fail-closed Codex invocation."""
    if reasoning_effort not in CODEX_REASONING_EFFORTS:
        raise ValueError(
            f"Codex effort must be one of {sorted(CODEX_REASONING_EFFORTS)}, "
            f"not {reasoning_effort!r}"
        )
    return [
        "codex", "exec",
        "--json",
        "--ephemeral",
        "--ignore-user-config",
        "--strict-config",
        "--model", model or DEFAULT_CODEX_MODEL,
        "--config", f'model_reasoning_effort="{reasoning_effort}"',
        "--config", 'web_search="disabled"',
        "--config", "sandbox_workspace_write.network_access=false",
        "--config", 'approval_policy="never"',
        "--sandbox", "workspace-write",
        "--cd", ws,
        "--skip-git-repo-check",
        "--color", "never",
        task,
    ]


def _codex_environment() -> dict[str, str]:
    """Pass authentication and ordinary shell basics, but no API-key secrets."""
    allowed = {
        "PATH", "HOME", "CODEX_HOME", "TMPDIR", "TMP", "TEMP", "LANG",
        "LC_ALL", "LC_CTYPE", "TERM", "USER", "LOGNAME", "SHELL",
        "VIRTUAL_ENV", "SSL_CERT_FILE", "SSL_CERT_DIR",
    }
    result = {key: value for key, value in os.environ.items() if key in allowed}
    result["GKM_SANITIZE_PROPOSER_INTERRUPTS"] = "1"
    return result


def _codex_usage_from_jsonl(path: str) -> dict:
    """Extract the one turn's ID and token counters from a raw JSONL transcript."""
    result = {
        "thread_id": None,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
        "usage_reported": False,
        "observed_tokens": 0,
    }
    try:
        lines = _read(path).splitlines()
    except OSError:
        return result
    for raw in lines:
        try:
            event = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "thread.started" and isinstance(event.get("thread_id"), str):
            result["thread_id"] = event["thread_id"]
        if event.get("type") != "turn.completed" or not isinstance(event.get("usage"), dict):
            continue
        usage = event["usage"]
        for field in (
            "input_tokens", "cached_input_tokens", "output_tokens",
            "reasoning_output_tokens",
        ):
            value = usage.get(field)
            if isinstance(value, int) and value >= 0:
                result[field] = value
        result["usage_reported"] = True
    result["observed_tokens"] = result["input_tokens"] + result["output_tokens"]
    return result


def _process_group_exists(process_group: int) -> bool:
    """Return whether any process still belongs to ``process_group``."""
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # A group we created but can no longer inspect is not a clean terminal
        # boundary; callers must continue to treat it as present.
        return True
    return True


def _stop_process_group(
    proc: subprocess.Popen, grace_seconds: float = 5.0
) -> bool:
    """Terminate the Codex CLI and every shell command it spawned.

    The session leader may already have exited while a background tool remains.
    Never return merely because ``proc.poll()`` is terminal: the process-group
    boundary, not the leader PID, defines a completed proposer turn.
    """
    process_group = proc.pid
    if not _process_group_exists(process_group):
        return True
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except PermissionError:
        # Some nested test/launcher sandboxes refuse group signalling even for
        # the child they just created.  Give a naturally terminating leader the
        # same bounded grace period; never claim containment if any group
        # member survives it.
        if proc.poll() is None:
            try:
                proc.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                return False
        return not _process_group_exists(process_group)
    deadline = time.monotonic() + grace_seconds
    if proc.poll() is None:
        try:
            proc.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process_group, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                return False
            proc.wait()
    # SIGTERM was delivered to the complete group.  A remaining group gets one
    # unconditional SIGKILL; nested macOS sandboxes can deny the post-reap
    # existence probe even when the group is already gone, so a reaped leader
    # plus a successfully delivered group TERM remains a valid containment
    # boundary for this compatibility harness.  The contiguous container runner
    # uses descriptor/PID-namespace receipts for the stronger production proof.
    if _process_group_exists(process_group):
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            return True
        except PermissionError:
            return proc.poll() is not None
    if proc.poll() is None:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            return False
    return proc.poll() is not None


def _scan_transcript_protocol_marker(
    path: str, offset: int, carry: bytes
) -> tuple[bool, int, bytes]:
    """Incrementally scan an append-only protected transcript for violations.

    The proposer cannot access this path.  Keeping only a marker-sized overlap
    avoids rereading a multi-hour JSONL transcript on every liveness poll while
    still detecting a marker split across writes.
    """
    marker = A.PUBLIC_ACTION_PROTOCOL_VIOLATION_MARKER.encode("utf-8")
    try:
        size = os.path.getsize(path)
        if size < offset:
            offset = 0
            carry = b""
        with open(path, "rb") as handle:
            handle.seek(offset)
            appended = handle.read()
            new_offset = handle.tell()
    except OSError:
        return False, offset, carry
    combined = carry + appended
    found = marker in combined
    keep = max(0, len(marker) - 1)
    return found, new_offset, combined[-keep:] if keep else b""


def _taint_failure_detail_class(reason: Optional[str]) -> str:
    """Map a scanner reason to one stable accounting classification."""

    normalized = (reason or "").lower()
    if (
        "host_process_introspection" in normalized
        or "host process introspection" in normalized
    ):
        return "host_process_introspection"
    if "external web/network" in normalized or "external_network" in normalized:
        return "external_web_or_network"
    if (
        "private game/runtime introspection" in normalized
        or "runtime_introspection" in normalized
    ):
        return "private_runtime_introspection"
    if "public action protocol violation" in normalized:
        return "public_action_protocol_violation"
    return "filesystem_boundary_violation"


def _codex_agent(ws: str, task: str, model: Optional[str], minutes: int, *,
                 reasoning_effort: str = "medium",
                 allocation_policy: str = DEFAULT_CODEX_ALLOCATION_POLICY,
                 weekly_reserve: int = 80,
                 weekly_headroom: int = 1,
                 max_campaign_tokens: int = 2_000_000,
                 max_campaign_runs: int = 12,
                 ledger_path: Optional[str] = None,
                 run_label: Optional[str] = None,
                 game: Optional[str] = None,
                 target_level: Optional[int] = None,
                 frontier_binding: Optional[dict] = None) -> dict:
    """Run one metered Codex proposer turn under the campaign guard.

    The live raw ``--json`` stream is written outside the proposer-writable
    workspace, then copied back as an immutable turn log and
    ``proposer_last.log`` after the process exits. Attempted
    source/runtime/network access therefore remains visible to the taint gate
    even if the proposer tries to edit or move its own audit record. The
    local token cap is an admission cap rather than a provider-side hard token
    ceiling; wall time and the live weekly reserve are the hard pre-turn bounds.
    Finite pools remain serialized across the full transaction.  A
    provider-confirmed unlimited pool releases the admission lock after
    preflight, allowing disjoint model turns to run concurrently while durable
    ledger appends remain serialized separately.
    """
    if minutes <= 0:
        raise ValueError("Codex minutes must be positive")
    if allocation_policy not in {"hard", "drain"}:
        raise ValueError(
            "Codex allocation_policy must be either 'hard' or 'drain'"
        )
    validated_frontier_binding = None
    if frontier_binding is not None:
        validated_frontier_binding = CCS.validate_frontier_binding(
            frontier_binding,
            expected_game=game,
            expected_target_level=target_level,
        )
    chosen_model = model or DEFAULT_CODEX_MODEL
    ledger = ledger_path or os.fspath(CUG.DEFAULT_LEDGER)
    cmd = _codex_command(ws, task, chosen_model, reasoning_effort)
    latest_log_path = os.path.join(ws, "proposer_last.log")
    transcript_root = _protected_codex_transcript_dir(ws)
    os.makedirs(transcript_root, exist_ok=True)

    lock_held = False
    try:
        # Concurrent unlimited turns need only serialize the short live
        # preflight.  Wait through that overlap instead of misreporting a
        # transient admission-lock collision as credit exhaustion.
        lock = CUG.campaign_lock(ledger, wait_seconds=30.0)
        lock.__enter__()
        lock_held = True
    except CUG.CodexUsageGuardError as exc:
        raise ProposerInfrastructureError(
            f"Codex campaign admission failed: {exc}"
        ) from exc
    try:
        try:
            before = CUG.preflight(
                reserve_percent=weekly_reserve,
                minimum_headroom_percent=weekly_headroom,
                max_campaign_tokens=max_campaign_tokens,
                max_campaign_runs=max_campaign_runs,
                ledger_path=ledger,
            )
        except CUG.CodexUsageGuardError as exc:
            if _usage_guard_error_is_cost_block(exc):
                raise CreditOut(
                    f"Codex campaign guard stopped the run: {exc}"
                ) from exc
            raise ProposerInfrastructureError(
                f"Codex rate-limit preflight failed: {exc}"
            ) from exc
        if not before.get("cost_control_enabled", True):
            lock.__exit__(None, None, None)
            lock_held = False

        started = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_label or "turn").strip("_")
        log_name = f"codex_turn_{stamp}_{safe_label}.jsonl"
        log_path = os.path.join(transcript_root, log_name)
        workspace_log_path = os.path.join(ws, log_name)
        diagnostics_name = f"codex_turn_{stamp}_{safe_label}.stderr.log"
        diagnostics_path = os.path.join(transcript_root, diagnostics_name)
        workspace_diagnostics_path = os.path.join(ws, diagnostics_name)
        proc = None
        timed_out = False
        allocation_expired = False
        interrupted = False
        protocol_violation = False
        boundary_violation_reason = None
        launch_error = None
        transcript_bytes = None
        diagnostics_bytes = None
        transcript_evidence_error = None
        postflight_taint_reason = None
        surviving_process_group = False
        process_group_quiesced = True
        process_group_stop_attempted = False
        try:
            with open(log_path, "w") as log, open(
                diagnostics_path, "w"
            ) as diagnostics_log:
                boundary_monitor = APB.LiveBoundaryMonitor(
                    Path(ws),
                    arena_module_root=Path(__file__).resolve().parent,
                    trusted_host_scaffolds=(
                        _trusted_host_scaffold_hashes(ws)
                    ),
                )
                prelaunch_boundary_reasons = _codex_boundary_reasons(
                    boundary_monitor, Path(log_path)
                )
                if prelaunch_boundary_reasons:
                    boundary_violation_reason = (
                        prelaunch_boundary_reasons[0]
                    )
                else:
                    proc = subprocess.Popen(
                        cmd,
                        cwd=ws,
                        # The task is already the final argv element.
                        # Inheriting the supervisor PTY makes ``codex exec``
                        # also probe that stream and prepend non-JSON input to
                        # an otherwise strict JSONL acquisition transcript.
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        # Keep CLI diagnostics in a separately sealed sideband
                        # so the authoritative event stream remains strict.
                        stderr=diagnostics_log,
                        text=True,
                        env=_codex_environment(),
                        start_new_session=True,
                    )
                try:
                    deadline = time.monotonic() + minutes * 60
                    scan_offset = 0
                    scan_carry = b""
                    diagnostics_scan_offset = 0
                    diagnostics_scan_carry = b""
                    while proc is not None and proc.poll() is None:
                        boundary_reasons = _codex_boundary_reasons(
                            boundary_monitor, Path(log_path)
                        )
                        if boundary_reasons:
                            boundary_violation_reason = boundary_reasons[0]
                            process_group_stop_attempted = True
                            process_group_quiesced = _stop_process_group(proc)
                            print(
                                "[codex proposer crossed the clean-room "
                                "filesystem boundary; terminating and "
                                "quarantining the complete turn]"
                            )
                            break
                        (
                            marker_found,
                            scan_offset,
                            scan_carry,
                        ) = _scan_transcript_protocol_marker(
                            log_path, scan_offset, scan_carry
                        )
                        (
                            diagnostics_marker_found,
                            diagnostics_scan_offset,
                            diagnostics_scan_carry,
                        ) = _scan_transcript_protocol_marker(
                            diagnostics_path,
                            diagnostics_scan_offset,
                            diagnostics_scan_carry,
                        )
                        if marker_found or diagnostics_marker_found:
                            protocol_violation = True
                            process_group_stop_attempted = True
                            process_group_quiesced = _stop_process_group(proc)
                            print(
                                "[codex proposer emitted a public-action "
                                "protocol violation; terminating and "
                                "quarantining the complete turn]"
                            )
                            break
                        remaining = deadline - time.monotonic()
                        if remaining <= 0 and not allocation_expired:
                            allocation_expired = True
                            if allocation_policy == "drain":
                                print(
                                    f"[codex proposer crossed its {minutes}min "
                                    "soft allocation; draining the healthy live "
                                    "turn without signalling it]"
                                )
                            else:
                                timed_out = True
                                process_group_stop_attempted = True
                                process_group_quiesced = _stop_process_group(
                                    proc
                                )
                                print(
                                    f"[codex proposer hit {minutes}min hard "
                                    "containment limit; preserving partial work "
                                    "as a non-solver outcome]"
                                )
                                break
                        wait_seconds = (
                            0.5
                            if allocation_expired
                            else max(0.01, min(0.5, remaining))
                        )
                        try:
                            proc.wait(timeout=wait_seconds)
                        except subprocess.TimeoutExpired:
                            continue
                    # A marker can be the final write immediately before a
                    # normal process exit, between the last poll and reap.
                    (
                        marker_found,
                        scan_offset,
                        scan_carry,
                    ) = _scan_transcript_protocol_marker(
                        log_path, scan_offset, scan_carry
                    )
                    (
                        diagnostics_marker_found,
                        diagnostics_scan_offset,
                        diagnostics_scan_carry,
                    ) = _scan_transcript_protocol_marker(
                        diagnostics_path,
                        diagnostics_scan_offset,
                        diagnostics_scan_carry,
                    )
                    protocol_violation = (
                        protocol_violation
                        or marker_found
                        or diagnostics_marker_found
                    )
                    terminal_boundary_reasons = _codex_boundary_reasons(
                        boundary_monitor, Path(log_path), final=True
                    )
                    if terminal_boundary_reasons:
                        boundary_violation_reason = (
                            boundary_violation_reason
                            or terminal_boundary_reasons[0]
                        )
                except KeyboardInterrupt:
                    interrupted = True
                    if proc is not None:
                        process_group_stop_attempted = True
                        process_group_quiesced = _stop_process_group(proc)
        except (OSError, subprocess.SubprocessError) as exc:
            launch_error = exc

        # A terminal leader PID is insufficient: a backgrounded tool may still
        # mutate the workspace or hold the raw transcript descriptor.  Seal
        # bytes only after terminating that residual process group, and mark
        # the whole generation non-authoritative.
        if (
            proc is not None
            and not process_group_stop_attempted
            and _process_group_exists(proc.pid)
        ):
            surviving_process_group = True
            process_group_stop_attempted = True
            process_group_quiesced = _stop_process_group(proc)
        if not process_group_quiesced:
            transcript_evidence_error = PermissionError(
                "could not terminate the complete proposer process group"
            )

        # Copy only after Codex and every child tool process has exited.  The
        # proposer never receives a writable handle or path to the live audit
        # stream.  A missing/unlinked/replaced protected pathname invalidates
        # this entire generation: never fall back to an older workspace log.
        if transcript_evidence_error is None:
            try:
                transcript_bytes = _read_single_link_regular(log_path)
                diagnostics_bytes = _read_single_link_regular(
                    diagnostics_path
                )
            except (OSError, WorkspaceTainted) as exc:
                transcript_evidence_error = exc
                # The event stream and diagnostic sideband form one evidence
                # pair.  Never report either half as sealed when the other
                # could not be reopened byte-exactly after process quiescence.
                transcript_bytes = None
                diagnostics_bytes = None
            else:
                _atomic_host_write(workspace_log_path, transcript_bytes)
                _atomic_host_write(latest_log_path, transcript_bytes)
                _atomic_host_write(
                    workspace_diagnostics_path, diagnostics_bytes
                )
                _atomic_host_write(
                    os.path.join(ws, "proposer_last.stderr.log"),
                    diagnostics_bytes,
                )
                protocol_violation = (
                    protocol_violation
                    or A.PUBLIC_ACTION_PROTOCOL_VIOLATION_MARKER.encode(
                        "utf-8"
                    )
                    in transcript_bytes
                    or A.PUBLIC_ACTION_PROTOCOL_VIOLATION_MARKER.encode(
                        "utf-8"
                    )
                    in diagnostics_bytes
                )
                # The incremental monitor is the early-stop mechanism.  This
                # terminal reopening is the accounting choke point: every
                # scanner class, including a fast broad ``ps``/``pgrep`` that
                # completed between polls, must become a typed taint row
                # before control can return to orchestration or the scheduler.
                if (
                    not protocol_violation
                    and boundary_violation_reason is None
                ):
                    postflight_taint_reason = (
                        _workspace_or_protected_taint_reason(ws)
                    )
                    if postflight_taint_reason is not None:
                        boundary_violation_reason = postflight_taint_reason
        usage = (
            _codex_usage_from_jsonl(workspace_log_path)
            if transcript_bytes is not None
            else _codex_usage_from_jsonl("")
        )
        postflight = None
        postflight_error = None
        try:
            postflight = CUG.weekly_allowance(CUG.query_rate_limits()).as_dict()
        except CUG.CodexUsageGuardError as exc:
            postflight_error = str(exc)

        terminal_errors = (
            _codex_terminal_error_messages(workspace_log_path)
            if proc is not None and not timed_out and proc.returncode != 0
            and transcript_bytes is not None
            else []
        )
        failure_detail_class = "unknown_cli"
        failure_class = "infrastructure"
        for message in terminal_errors:
            classification = _classify_provider_error_message(message)
            if classification == "credit_out":
                failure_class = classification
                failure_detail_class = "provider_credit_out"
                break
            if classification == "infrastructure":
                failure_class = classification
                failure_detail_class = "known_transient"
        if launch_error is not None:
            failure_class = "infrastructure"
            failure_detail_class = "launch_error"

        allowance_before = before["allowance"]
        record = {
            "event": "codex_exec",
            "started_at": started_at,
            "duration_seconds": round(time.monotonic() - started, 3),
            "run_label": run_label,
            "game": game,
            "target_level": target_level,
            **(validated_frontier_binding or {}),
            "transcript": log_name,
            "diagnostics": diagnostics_name,
            "workspace": os.path.basename(os.path.abspath(ws)),
            "model": chosen_model,
            "reasoning_effort": reasoning_effort,
            "minutes_limit": minutes,
            "allocation_policy": allocation_policy,
            "allocation_expired": allocation_expired,
            "timed_out": timed_out,
            "interrupted": interrupted,
            "returncode": proc.returncode if proc is not None else None,
            "launch_error": type(launch_error).__name__ if launch_error else None,
            "failure_class": (
                "evidence"
                if (
                    transcript_evidence_error is not None
                    or surviving_process_group
                )
                else "taint"
                if protocol_violation or boundary_violation_reason is not None
                else "containment"
                if timed_out
                else failure_class
                if launch_error is not None
                or (
                    proc is not None
                    and not timed_out
                    and proc.returncode != 0
                )
                else None
            ),
            "failure_detail_class": (
                "protected_transcript_unavailable"
                if transcript_evidence_error is not None
                else "surviving_process_group"
                if surviving_process_group
                else "public_action_protocol_violation"
                if protocol_violation
                else _taint_failure_detail_class(
                    boundary_violation_reason
                )
                if boundary_violation_reason is not None
                else "hard_wall_time"
                if timed_out
                else failure_detail_class
                if launch_error is not None
                or (
                    proc is not None
                    and not timed_out
                    and proc.returncode != 0
                )
                else None
            ),
            "protected_transcript_status": (
                "sealed"
                if transcript_bytes is not None
                else "unavailable"
            ),
            "protected_transcript_sha256": (
                hashlib.sha256(transcript_bytes).hexdigest()
                if transcript_bytes is not None
                else None
            ),
            "protected_diagnostics_status": (
                "sealed"
                if diagnostics_bytes is not None
                else "unavailable"
            ),
            "protected_diagnostics_sha256": (
                hashlib.sha256(diagnostics_bytes).hexdigest()
                if diagnostics_bytes is not None
                else None
            ),
            "protected_transcript_error": (
                type(transcript_evidence_error).__name__
                if transcript_evidence_error is not None
                else None
            ),
            "surviving_process_group": surviving_process_group,
            "public_action_protocol_violation": protocol_violation,
            "taint_verdict": (
                "tainted"
                if protocol_violation or boundary_violation_reason is not None
                else None
            ),
            "filesystem_boundary_violation": (
                boundary_violation_reason is not None
            ),
            "filesystem_boundary_violation_reason": (
                boundary_violation_reason
            ),
            **_filesystem_boundary_policy_binding(),
            "terminal_errors": terminal_errors,
            "weekly_used_before": allowance_before["used_percent"],
            "weekly_remaining_before": allowance_before["remaining_percent"],
            "weekly_resets_at": allowance_before["resets_at"],
            "weekly_window_before": allowance_before.get("window_name"),
            "weekly_limit_id_before": allowance_before.get("limit_id"),
            "weekly_used_after": postflight["used_percent"] if postflight else None,
            "weekly_remaining_after": postflight["remaining_percent"] if postflight else None,
            "weekly_window_after": postflight.get("window_name") if postflight else None,
            "weekly_limit_id_after": postflight.get("limit_id") if postflight else None,
            "postflight_error": postflight_error,
            **usage,
        }
        CUG.append_ledger(record, ledger)

        if transcript_evidence_error is not None:
            raise ProposerEvidenceUnavailable(
                "protected Codex transcript is unavailable or unstable; "
                "discarding this complete proposer generation without WIP "
                f"reuse or promotion ({log_path}: "
                f"{type(transcript_evidence_error).__name__})"
            ) from transcript_evidence_error
        if surviving_process_group:
            raise ProposerEvidenceUnavailable(
                "Codex leader exited while a spawned process remained alive; "
                "the process group was terminated and this proposer "
                "generation is discarded without WIP reuse or promotion"
            )
        if protocol_violation:
            raise ProposerProtocolViolation(
                "Codex attempted an action outside the public protocol; "
                "the complete proposer generation was terminated and is "
                "discarded without WIP reuse or promotion"
            )
        if boundary_violation_reason is not None:
            raise ProposerBoundaryViolation(
                "Codex crossed the clean-room filesystem boundary; the "
                "complete proposer generation was terminated and is "
                "discarded without WIP reuse or promotion: "
                f"{boundary_violation_reason}"
            )
        if launch_error is not None:
            raise ProposerInfrastructureError(
                f"could not launch Codex CLI: {launch_error}"
            ) from launch_error
        if interrupted:
            raise KeyboardInterrupt
        assert proc is not None
        if timed_out:
            raise ProposerContainmentTimeout(
                f"Codex proposer hit the {minutes}min hard containment limit; "
                "the partial turn is not clean solver no-progress"
            )
        if not timed_out and proc.returncode != 0:
            detail = terminal_errors[-1] if terminal_errors else (
                f"Codex CLI exited with status {proc.returncode}"
            )
            if failure_class == "credit_out":
                raise CreditOut(f"codex reported no credits/quota (see {log_path})")
            if failure_class == "infrastructure":
                raise ProposerInfrastructureError(
                    f"Codex provider/transport failure: {detail} (see {log_path})"
                )
            raise ProposerInfrastructureError(
                f"Codex CLI failed with unclassified status {proc.returncode}: "
                f"{detail} (see {log_path})"
            )
        return record
    finally:
        if lock_held:
            lock.__exit__(None, None, None)


def _record_codex_level_outcome(turn: Optional[dict], *, ledger_path: Optional[str],
                                game: str, level: int, reached_before: int,
                                reached_after: int, path: list,
                                marginal_C: int) -> None:
    """Join provider usage to the independently verified level outcome."""
    if not turn:
        return
    ledger = ledger_path or os.fspath(CUG.DEFAULT_LEDGER)
    outcome = {
        "event": "codex_level_outcome",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "thread_id": turn.get("thread_id"),
        "codex_exec_transcript": turn.get("transcript"),
        "run_label": turn.get("run_label"),
        "model": turn.get("model"),
        "reasoning_effort": turn.get("reasoning_effort"),
        "game": game,
        "target_level": level,
        **{
            field: turn.get(field)
            for field in (
                *CCS.FRONTIER_BINDING_FIELDS,
                "reached",
                "parent_action_count",
            )
            if turn.get(field) is not None
        },
        "reached_before": reached_before,
        "reached_after": reached_after,
        "solved_target": reached_after >= level,
        "winning_path_present": bool(path),
        "winning_marginal_C": marginal_C if reached_after >= level else None,
        "taint_verdict": "clean",
    }
    try:
        # Outcome joins are bookkeeping appends, not cost-admission decisions.
        # The dedicated blocking append lock preserves every concurrent
        # unlimited-lane record without contending on the zero-wait campaign
        # admission lock.
        CUG.append_ledger(outcome, ledger)
    except CUG.CodexUsageGuardError as exc:
        print(f"[warning: could not append Codex level outcome: {exc}]")


def _api_agent(ws: str, task: str, model: Optional[str], minutes: int, *,
               guard: bool = False,
               ledger_path: Optional[str] = None,
               window_hours: float = CLG.DEFAULT_WINDOW_HOURS,
               max_cost_usd: Optional[float] = None,
               max_turns: Optional[int] = None,
               max_wall_minutes: Optional[float] = None,
               run_label: Optional[str] = None,
               game: Optional[str] = None,
               target_level: Optional[int] = None) -> None:
    """Run the Messages-API agentic loop (gkm_api_agent) as the proposer.

    Bills ANTHROPIC_API_KEY Console dollars -- a pool separate from the Claude
    subscription (``_claude_agent``).  ``guard=True`` meters each turn's OBSERVED
    dollar cost (tokens x model price) to an API ledger and refuses once a rolling
    per-window dollar cap is reached; there is no provider balance read, so the cap
    is a spend ceiling.  ``guard=False`` keeps the original unmetered behavior.
    """
    import gkm_api_agent
    ledger = (ledger_path or os.fspath(CLG.DEFAULT_API_LEDGER)) if guard else None
    caps = CLG.WindowCaps(
        max_cost_usd=max_cost_usd, max_turns=max_turns,
        max_wall_minutes=max_wall_minutes,
    ) if guard else None
    lock = None
    if guard:
        try:
            lock = CLG.campaign_lock(ledger)
            lock.__enter__()
            CLG.preflight(caps=caps, window_hours=window_hours, ledger_path=ledger,
                          event="api_exec")
        except CLG.ClaudeUsageGuardError as exc:
            if lock is not None:
                lock.__exit__(None, None, None)
            raise CreditOut(f"API campaign guard stopped the run: {exc}") from exc

    try:
        started = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        usage: dict = {}
        try:
            _ensure_anthropic_api_key()
            gkm_api_agent.run_agent(ws, task, model=model, minutes=minutes,
                                    usage_out=usage)
        except Exception as ex:
            with open(os.path.join(ws, "proposer_last.log"), "w") as fh:
                fh.write(_redact_secrets(f"API Error: {type(ex).__name__}: {ex}\n"))
        duration = round(time.monotonic() - started, 3)
        blob = _read(os.path.join(ws, "proposer_last.log")).lower()
        credit_out = any(m in blob for m in _CREDIT_OUT_MARKERS)
        if guard:
            CLG.append_ledger({
                "event": "api_exec",
                "started_at": started_at,
                "duration_seconds": duration,
                "run_label": run_label,
                "workspace": os.path.basename(os.path.abspath(ws)),
                "proposer": "api",
                "billing_pool": "api_console",
                "model": usage.get("model") or model or "default",
                "minutes_limit": minutes,
                "credit_out": credit_out,
                "game": game,
                "target_level": target_level,
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
                "total_cost_usd": usage.get("total_cost_usd"),
                "usage_reported": bool(usage),
            }, ledger)
        if credit_out:
            raise CreditOut(f"api proposer reported no credits/quota (see {ws}/proposer_last.log)")
    finally:
        if lock is not None:
            lock.__exit__(None, None, None)


CLEAN_ROOM_INSTRUCTION = (
    "CLEAN-ROOM BOUNDARY: work only with files in the current workspace. The sole "
    "outside-workspace capability is the host-injected exact module root needed to "
    "import gkm_arena and directly call its public run_program function; the "
    "supervisor-owned gkm_try.py may additionally call validate. This capability "
    "does not authorize constructing Arena, passing or aliasing the module/function, "
    "introspecting it, or listing/reading its module directory. The current-workspace "
    "gkm_try.py and perception.py are legitimate to inspect. Do not read parent "
    "directories, artifacts outside the admitted current-workspace lineage, hidden "
    "implementations, or underscore-prefixed runtime state. Same-lineage prior "
    "observations and proposer transcripts already present in this workspace are "
    "admissible, but reproduce every fact used. Do not inspect or signal host "
    "processes; run bounded probe workers in the foreground. Do not use any internet "
    "service. Any attempted boundary crossing invalidates the entire discovery "
    "lineage, even if it does not help."
    " TOKEN DISCIPLINE: start with solver_index.md, which lists signatures, line "
    "ranges, docstrings, and direct calls without bodies. Use those ranges instead "
    "of printing whole files. If frontier_brief.md exists, read it next: it indexes "
    "the previous clean attempt's unverified observations and preserved probes; "
    "reproduce any fact you rely on instead of rereading the full transcript. Keep probe "
    "outputs symbolic and compact. The workspace has its own local Git boundary; "
    "repository-wide status or diff inspection is unnecessary."
)


def _propose_task(game, K, context, legs_index):
    return (CLEAN_ROOM_INSTRUCTION + "\n\n" + A.PRECONCEPTIONS + "\n\n" + context +
            f"\n\nYou are growing a LEG LIBRARY across the levels of {game}. Existing "
            f"legs in legs.py: {legs_index or '(none yet)'}.\n"
            f"GOAL: make solve.py reach LEVEL {K}. First run `python gkm_try.py` to see "
            "where you are; solve.py dispatches to players.play_level_K. "
            "`checkpoint.json` is supervisor-owned campaign/accounting state: never "
            "edit, replace, delete, chmod, or regenerate it. If edits to earlier-level "
            "code or prefix optimization must be tested from level 1, run "
            "`GKM_FRESH_REPLAY=1 python gkm_try.py`; save any candidate action path "
            "under a different descriptive `.json` filename for independent harness "
            "adoption. On a clone at "
            f"level {K}, learn its structure. Use perception.py first: it is a "
            "source-free scaffold that turns frames into blobs, object candidates, "
            "action deltas, replay summaries, and bounded clone BFS keys. "
            "Inspect env.actions: key actions are integers; coordinate-only games "
            "use env.step(6, x, y), recorded in replay paths as [6, x, y]. "
            "ACTION6 x and y must be integers inside the returned 64x64 frame "
            "(0..63 inclusive). Never pass bare `6` to `env.step` or to an "
            "explicit perception action list; `action_deltas(env)` safely tests "
            "the advertised key actions by default, while coordinate probes must "
            "supply explicit `(6, x, y)` tuples. Never probe off-frame or malformed "
            "actions. Any "
            "such attempt invalidates and terminates the complete proposer turn even "
            "when a probe catches the exception. "
            f"Build small symbolic probes on top of those observations instead of repeatedly "
            f"dumping raw pixels. Then WRITE `play_level_{K}(env)` in "
            "players.py that ONLY COMPOSES legs imported from legs.py. REUSE existing "
            "legs wherever observations confirm that the level is an earlier mechanism "
            "in a new configuration; add NEW "
            "legs to legs.py ONLY when nothing fits, and keep them minimal and general. "
            "If a candidate repeatedly stalls or exhausts the real-move budget, falsify "
            "its core mechanic from a pristine level entry before extending its suffix; "
            "forced reuse is not compression. "
            "Do not put level logic inline in the player -- put reusable skills in "
            "legs.py. Iterate with `python gkm_try.py` until RESULT shows "
            f"levels>={K}. Keep clone use bounded (~300 steps/s).")


def _debrief_task(game, K):
    return (CLEAN_ROOM_INSTRUCTION + "\n\n" +
            f"DEBRIEF after clearing {game} level {K}. Compare play_level_{K} to the "
            "earlier players in players.py. Refactor any repeated code into shared, "
            "well-named legs in legs.py (write each skill ONCE) and update the players "
            "to call them; the players should be thin composition. Append the recurring "
            "composition pattern you notice (a candidate higher-order leg) to "
            "legs_log.md. Do NOT change behaviour: run `python gkm_try.py` and confirm "
            f"RESULT still shows levels>={K}.")


@dataclass
class LevelRecord:
    level: int
    marginal_C: int
    reached: bool


@dataclass
class Report:
    game: str
    reached: int
    records: List[LevelRecord] = field(default_factory=list)
    total_marginal_C: int = 0
    final_path: list = field(default_factory=list)
    validated: bool = False

    @property
    def free_energy(self):
        return free_energy(self.reached, self.total_marginal_C)


def _checked_codex_frontier_binding(
    game: str,
    target_level: int,
    tag: str,
    expected: Optional[dict],
) -> dict:
    """Resolve one exact parent and reject a stale scheduler decision."""
    try:
        binding = CCS.exact_frontier_binding(
            Path(artifact_dir(game, tag)),
            game=game,
            target_level=target_level,
        )
    except ValueError as exc:
        if expected is None:
            raise
        raise ValueError(
            "exact frontier changed after scheduler decision; "
            "refusing stale Codex dispatch"
        ) from exc
    if expected is not None and binding != expected:
        raise ValueError(
            "exact frontier changed after scheduler decision; "
            "refusing stale Codex dispatch"
        )
    return binding


def orchestrate(game="wa30", max_level=9, model=None, minutes_per=40,
                proposer="claude", tag="",
                seed_artifact: bool = True,
                restore_wip: bool = True,
                codex_effort: str = "medium",
                codex_debrief_effort: str = "medium",
                codex_allocation_policy: str = DEFAULT_CODEX_ALLOCATION_POLICY,
                debrief_policy: str = "always",
                debrief_threshold: int = 150,
                codex_weekly_reserve: int = 80,
                codex_weekly_headroom: int = 1,
                codex_max_campaign_tokens: int = 2_000_000,
                codex_max_campaign_runs: int = 12,
                transient_retries: int = _TRANSIENT_RETRIES,
                codex_ledger: Optional[str] = None,
                expected_frontier_binding: Optional[dict] = None,
                expected_wip_attempt: Optional[str] = None,
                claude_guard: bool = False,
                claude_ledger: Optional[str] = None,
                claude_window_hours: float = CLG.DEFAULT_WINDOW_HOURS,
                claude_max_turns: Optional[int] = None,
                claude_max_wall_minutes: Optional[float] = None,
                api_guard: bool = False,
                api_ledger: Optional[str] = None,
                api_window_hours: float = CLG.DEFAULT_WINDOW_HOURS,
                api_max_cost_usd: Optional[float] = None,
                api_max_turns: Optional[int] = None,
                api_max_wall_minutes: Optional[float] = None,
                propose_fn: Optional[Callable] = None,
                verify_fn: Optional[Callable] = None,
                debrief_fn: Optional[Callable] = None,
                verbose=True) -> Report:
    """Per-level compose->verify->optional-debrief with marginal-C accounting.

    propose_fn(ws,K) / verify_fn(game, solve_path)->(levels,path,err) /
    debrief_fn(ws,K) are injectable; defaults use either the Claude Code agent
    (``proposer="claude"``) or opencode (``proposer="opencode"``) as proposer,
    and the real game as verifier (credits needed for either proposer).
    """
    # Validate before acquiring a workspace lock so a bad campaign command
    # cannot strand an otherwise reusable scratch directory.
    should_run_debrief(
        debrief_policy,
        auto_solved=False,
        pre_debrief_marginal_C=0,
        threshold=debrief_threshold,
    )
    if transient_retries < 0:
        raise ValueError("transient_retries must be nonnegative")
    if (
        proposer == "codex"
        and propose_fn is None
        and not COMPATIBILITY_ARENA_CLOSURE_AUTHORITY
    ):
        raise RuntimeError(
            "native compatibility launch is disabled: the eager raw-Arena "
            "dependency closure still performs host initialization outside "
            "the sealed single-file receipt; use the container/RPC contiguous "
            "runner until that closure is purified and completely bound"
        )
    if codex_allocation_policy not in {"hard", "drain"}:
        raise ValueError(
            "codex_allocation_policy must be either 'hard' or 'drain'"
        )
    if expected_frontier_binding is not None:
        if proposer != "codex" or propose_fn is not None:
            raise ValueError(
                "expected frontier binding is supported only by the default "
                "Codex proposer"
            )
        expected_frontier_binding = CCS.validate_frontier_binding(
            expected_frontier_binding,
            expected_game=game,
            expected_target_level=max_level,
        )
    if expected_wip_attempt is not None:
        if expected_frontier_binding is None:
            raise ValueError(
                "expected WIP requires an exact scheduler frontier binding"
            )
        if not seed_artifact or not restore_wip:
            raise ValueError(
                "expected WIP requires verified-parent seeding and WIP restore"
            )
        if proposer != "codex" or propose_fn is not None:
            raise ValueError(
                "expected WIP is supported only by the default Codex proposer"
            )
        _validate_expected_wip_attempt(
            game, max_level, expected_wip_attempt, tag
        )
    authoritative_target = None
    if propose_fn is None:
        authoritative_target = authoritative_level_target(game)
        if (
            not isinstance(max_level, int)
            or isinstance(max_level, bool)
            or not 1 <= max_level <= authoritative_target
        ):
            raise ValueError(
                "requested max level is outside authoritative inventory: "
                f"{game} requested={max_level!r}, "
                f"authoritative_target={authoritative_target}"
            )
    lineage_lock = _acquire_lineage_lock(game)
    try:
        # Real proposer attempts always receive a new generation.  The
        # deterministic scratch name remains only for injected/offline tests.
        # This makes ``exclude`` literal and prevents zero-seed attempts from
        # inheriting old solver/probe bytes under a reused tag.
        if propose_fn is None:
            ws = setup_workspace(game, tag, isolated_generation=True)
        else:
            ws = setup_workspace(game, tag)
        initial_taint = _workspace_or_protected_taint_reason(ws)
        if initial_taint:
            raise WorkspaceTainted(
                "refusing to inspect or verify a pre-existing proposer "
                f"workspace across the clean-room boundary: {initial_taint}"
            )
        if not seed_artifact:
            # ``--fresh`` means zero-seed, not merely "start a new proposer
            # process".  It is safe for the first turn of a separately rooted
            # reacquisition, or when the same scratch workspace already carries
            # the current validated checkpoint.  A new/stale scratch workspace
            # beside a newer artifact would otherwise silently dispatch L1 and
            # waste a hard-frontier turn (the eventual promotion guard prevents
            # regression, but not the targeting error).
            artifact_checkpoint = _load_checkpoint(artifact_dir(game, tag))
            workspace_checkpoint = _load_checkpoint(ws)
            if (
                artifact_checkpoint is not None
                and artifact_checkpoint.validated
                and artifact_checkpoint.reached > 0
                and (
                    workspace_checkpoint is None
                    or not workspace_checkpoint.validated
                    or workspace_checkpoint.reached < artifact_checkpoint.reached
                )
            ):
                workspace_level = (
                    workspace_checkpoint.reached
                    if workspace_checkpoint is not None
                    and workspace_checkpoint.validated
                    else 0
                )
                raise ValueError(
                    "refusing zero-seed run against newer validated artifact: "
                    f"artifact reached={artifact_checkpoint.reached}, "
                    f"workspace reached={workspace_level}; omit --fresh to seed "
                    "the verified frontier"
                )
        run_lock = _acquire_workspace_lock(ws)
    except BaseException:
        _release_workspace_lock(lineage_lock)
        raise
    legs_p, players_p, solve_p = (os.path.join(ws, f) for f in ("legs.py", "players.py", "solve.py"))
    unpromoted_source_overlay = {}
    if seed_artifact:
        unpromoted_source_overlay = _clean_unpromoted_source_overlay(
            game, ws, tag
        )
        if restore_wip and not unpromoted_source_overlay:
            artifact_parent = _load_checkpoint(artifact_dir(game, tag))
            if artifact_parent is not None and artifact_parent.validated:
                unpromoted_source_overlay = _clean_wip_source_overlay(
                    game, artifact_parent.reached + 1, tag,
                    expected_attempt=expected_wip_attempt,
                )
    if seed_artifact:
        seed_workspace_from_artifact(
            game, ws, tag, verbose=verbose, restore_wip=restore_wip,
            expected_wip_attempt=expected_wip_attempt,
        )
        if unpromoted_source_overlay:
            _restore_source_overlay(ws, unpromoted_source_overlay)
            if verbose:
                print(
                    "preserved clean unpromoted solver source across parent "
                    "checkpoint seed; orphan replay will run before proposing"
                )
    elif verbose:
        print("fresh run requested: skipping artifact seed")
    if proposer == "codex" and propose_fn is None:
        _write_solver_source_index(ws)
        starting_checkpoint = _load_checkpoint(ws)
        next_level = starting_checkpoint.reached + 1 if starting_checkpoint else 1
        _write_frontier_brief(ws, game, next_level)
        taint_reason = _workspace_or_protected_taint_reason(ws)
        if taint_reason:
            raise WorkspaceTainted(
                f"refusing to expose tainted restored context to Codex: {taint_reason}"
            )
        _initialize_codex_workspace_git(ws)
        if expected_frontier_binding is not None:
            # This second scheduler-to-lineage handoff check runs only after
            # the per-game lineage lock is held and the workspace has been
            # seeded. It therefore catches a promotion/reset race even when
            # the newly promoted checkpoint already satisfies ``max_level``
            # and the proposal loop would otherwise be skipped.
            _checked_codex_frontier_binding(
                game,
                max_level,
                tag,
                expected_frontier_binding,
            )
    context = discovered_context(game) if propose_fn is None else ""
    codex_turn_records = {}
    codex_frontier_bindings = {}
    bind_default_codex_frontier = (
        propose_fn is None and proposer == "codex"
    )
    if propose_fn is None:
        agents = {"claude": _claude_agent, "opencode": _opencode_agent, "codex": _codex_agent, "api": _api_agent}
        _agent = agents[proposer]
        if proposer == "codex":
            def propose_fn(w, k):
                codex_turn_records[("propose", k)] = _agent(
                    w,
                    _propose_task(game, k, context, _defs(_read(legs_p))),
                    model,
                    minutes_per,
                    reasoning_effort=codex_effort,
                    allocation_policy=codex_allocation_policy,
                    weekly_reserve=codex_weekly_reserve,
                    weekly_headroom=codex_weekly_headroom,
                    max_campaign_tokens=codex_max_campaign_tokens,
                    max_campaign_runs=codex_max_campaign_runs,
                    ledger_path=codex_ledger,
                    run_label=f"{game}:L{k}:propose",
                    game=game,
                    target_level=k,
                    frontier_binding=codex_frontier_bindings[k],
                )
        elif proposer == "claude" and claude_guard:
            def propose_fn(w, k):
                _agent(
                    w,
                    _propose_task(game, k, context, _defs(_read(legs_p))),
                    model,
                    minutes_per,
                    guard=True,
                    ledger_path=claude_ledger,
                    window_hours=claude_window_hours,
                    max_turns=claude_max_turns,
                    max_wall_minutes=claude_max_wall_minutes,
                    run_label=f"{game}:L{k}:propose",
                    game=game,
                    target_level=k,
                )
        elif proposer == "api" and api_guard:
            def propose_fn(w, k):
                _agent(
                    w,
                    _propose_task(game, k, context, _defs(_read(legs_p))),
                    model,
                    minutes_per,
                    guard=True,
                    ledger_path=api_ledger,
                    window_hours=api_window_hours,
                    max_cost_usd=api_max_cost_usd,
                    max_turns=api_max_turns,
                    max_wall_minutes=api_max_wall_minutes,
                    run_label=f"{game}:L{k}:propose",
                    game=game,
                    target_level=k,
                )
        else:
            propose_fn = lambda w, k: _agent(
                w,
                _propose_task(game, k, context, _defs(_read(legs_p))),
                model,
                minutes_per,
            )
    if debrief_fn is None:
        agents = {"claude": _claude_agent, "opencode": _opencode_agent, "codex": _codex_agent, "api": _api_agent}
        _agent = agents[proposer]
        if proposer == "codex":
            def debrief_fn(w, k):
                codex_turn_records[("debrief", k)] = _agent(
                    w,
                    _debrief_task(game, k),
                    model,
                    max(10, minutes_per // 2),
                    reasoning_effort=codex_debrief_effort,
                    allocation_policy=codex_allocation_policy,
                    weekly_reserve=codex_weekly_reserve,
                    weekly_headroom=codex_weekly_headroom,
                    max_campaign_tokens=codex_max_campaign_tokens,
                    max_campaign_runs=codex_max_campaign_runs,
                    ledger_path=codex_ledger,
                    run_label=f"{game}:L{k}:debrief",
                    game=game,
                    target_level=k,
                )
        elif proposer == "claude" and claude_guard:
            def debrief_fn(w, k):
                _agent(
                    w, _debrief_task(game, k), model, max(10, minutes_per // 2),
                    guard=True,
                    ledger_path=claude_ledger,
                    window_hours=claude_window_hours,
                    max_turns=claude_max_turns,
                    max_wall_minutes=claude_max_wall_minutes,
                    run_label=f"{game}:L{k}:debrief",
                    game=game,
                    target_level=k,
                )
        elif proposer == "api" and api_guard:
            def debrief_fn(w, k):
                _agent(
                    w, _debrief_task(game, k), model, max(10, minutes_per // 2),
                    guard=True,
                    ledger_path=api_ledger,
                    window_hours=api_window_hours,
                    max_cost_usd=api_max_cost_usd,
                    max_turns=api_max_turns,
                    max_wall_minutes=api_max_wall_minutes,
                    run_label=f"{game}:L{k}:debrief",
                    game=game,
                    target_level=k,
                )
        else:
            debrief_fn = lambda w, k: _agent(
                w, _debrief_task(game, k), model, max(10, minutes_per // 2)
            )
    raw_verify_fn = verify_fn or run_solve_file

    def boundary_checked_verify(
        selected_game: str, selected_solve_path: str, *args, **kwargs
    ):
        # This is the single execution choke point for startup replay,
        # auto-solve, orphan recovery, proposal verification, and debrief.
        # Re-open the complete workspace/protected transcript boundary
        # immediately before importing or executing any proposer-authored byte.
        assert_workspace_not_tainted(ws)
        try:
            Path(selected_solve_path).resolve().relative_to(Path(ws).resolve())
        except (OSError, ValueError) as exc:
            raise WorkspaceTainted(
                "verifier source path escapes the attempt workspace: "
                f"{selected_solve_path}"
            ) from exc
        return raw_verify_fn(selected_game, selected_solve_path, *args, **kwargs)

    boundary_checked_verify._gkm_run_solve_file = (  # type: ignore[attr-defined]
        raw_verify_fn is run_solve_file
    )
    verify_fn = boundary_checked_verify

    rep = Report(game=game, reached=0)
    # resume from checkpoint (restores marginal-C history across restarts)
    ckpt = _load_checkpoint(ws)
    if ckpt is not None:
        rep = ckpt
        if authoritative_target is not None and rep.reached > authoritative_target:
            raise ValueError(
                "checkpoint exceeds authoritative inventory: "
                f"{game} reached={rep.reached}, "
                f"authoritative_target={authoritative_target}"
            )
        if verbose:
            print(f"resumed checkpoint: reached={rep.reached} total_marginal_C={rep.total_marginal_C}")
    # Also verify workspace files if there is no trusted promoted checkpoint.  For
    # clean continuations, the checkpoint was replay-validated before promotion; a
    # startup re-run of all solved levels can spend the next-level budget before the
    # proposer even starts.
    if ckpt is not None and ckpt.validated:
        if verbose:
            print(f"trusted validated checkpoint through level {rep.reached}; skipping startup replay")
    else:
        lv0, path0, _ = verify_fn(game, solve_p)
        if lv0 > rep.reached:
            rep.reached = lv0
            rep.final_path = path0
            rep.validated = A.validate(game, path0, lv0) if path0 else False
            _save_checkpoint(ws, rep)
            promote_verified_artifact(game, ws, rep, tag, verbose=verbose)
            if verbose:
                print(f"workspace solve.py clears level {lv0} (resuming from there)")
    # A hard interrupt (Ctrl-C / SIGTERM) must not lose the in-flight probe
    # context: snapshot it as phase 'interrupted', still checkpoint + promote
    # whatever is verified, then re-raise.
    interrupted = False
    protocol_tainted = False
    protocol_taint_detail = None
    try:
        signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    except ValueError:
        pass  # not in the main thread; Ctrl-C still covered
    try:
        while rep.reached < max_level:
            K = rep.reached + 1
            rollback_recovered_generation = False
            authorized_winning_turn = None
            reached_before_level = rep.reached
            assert_workspace_not_tainted(ws)
            if bind_default_codex_frontier:
                binding = _checked_codex_frontier_binding(
                    game,
                    K,
                    tag,
                    expected_frontier_binding,
                )
                codex_frontier_bindings[K] = binding
            legs_b, players_b = _frontier_marginal_baseline(
                game, ws, rep.reached, tag
            )

            def record_proposer_outcome(outcome_levels, outcome_path):
                turn = codex_turn_records.get(("propose", K))
                _record_codex_level_outcome(
                    turn,
                    ledger_path=codex_ledger,
                    game=game,
                    level=K,
                    reached_before=reached_before_level,
                    reached_after=outcome_levels,
                    path=outcome_path,
                    marginal_C=marginal_complexity(
                        legs_b, _read(legs_p), players_b, _read(players_p)
                    ),
                )

            # PHASE -1: an interrupted or orphaned clean turn may already have
            # written an exact winning solver even though its parent harness
            # died before snapshot/promotion.  Replaying that source first
            # preserves the original transcript and exact boundary instead of
            # overwriting it with auto-solve or paying for another proposer.
            assert_workspace_not_tainted(ws)
            if _workspace_has_unpromoted_solver_source(game, ws, tag):
                existing_levels, existing_path, existing_err = verify_fn(
                    game, solve_p
                )
                if (
                    existing_levels >= K
                    and not existing_err
                    and existing_path
                    and A.validate(game, existing_path, existing_levels)
                ):
                    assert_workspace_not_tainted(ws)
                    art = artifact_dir(game, tag)
                    parent_legs = (
                        _read(os.path.join(art, "legs.py"))
                        if os.path.isfile(os.path.join(art, "legs.py"))
                        else ""
                    )
                    parent_players = (
                        _read(os.path.join(art, "players.py"))
                        if os.path.isfile(os.path.join(art, "players.py"))
                        else ""
                    )
                    snapshot_wip_context(
                        game, ws, K, "recovered_existing_workspace_solver",
                        existing_levels, None, tag, verbose=verbose,
                    )
                    Cm = marginal_complexity(
                        parent_legs, _read(legs_p),
                        parent_players, _read(players_p),
                    )
                    exact_path = exact_level_boundary(
                        game, existing_path, K
                    )
                    if exact_path is None or not A.validate(
                        game, exact_path, K
                    ):
                        raise RuntimeError(
                            f"could not recover exact level-{K} boundary "
                            "from overshooting workspace solver"
                        )
                    _record_level(rep, K, Cm)
                    rep.reached = K
                    rep.final_path = exact_path
                    rep.validated = True
                    _save_checkpoint(ws, rep)
                    promote_verified_artifact(
                        game, ws, rep, tag, verbose=verbose
                    )
                    if verbose:
                        print(
                            f"level {K}: recovered exact winning workspace "
                            f"marginal_C={Cm} total_C={rep.total_marginal_C} "
                            f"validated=True F={rep.free_energy:.3f}"
                        )
                    continue

            # PHASE 0: auto-solve with existing legs (structural reuse, zero proposer cost).
            # Skip when it already failed at this level against this exact legs.py -- a
            # relaunch should not re-pay a long BFS for a known-negative result.
            if _auto_solve_failed_before(ws, K, legs_b):
                if verbose:
                    print(f"level {K}: auto-solve previously failed for current legs; skipping")
                auto_result = None
            else:
                auto_result = _try_auto_solve(K, legs_b, players_b,
                                              players_p, solve_p, game, verify_fn)
                if auto_result is None:
                    _record_auto_solve_failure(ws, K, legs_b)
            if auto_result is not None:
                levels, path, err = auto_result
                if bind_default_codex_frontier:
                    authorized_winning_turn = {
                        "authority_kind": "host_auto_solve"
                    }
                if verbose:
                    print(f"level {K}: auto-solved via existing legs")
                auto_marginal = marginal_complexity(
                    legs_b, _read(legs_p), players_b, _read(players_p)
                )
                if should_run_debrief(
                    debrief_policy,
                    auto_solved=True,
                    pre_debrief_marginal_C=auto_marginal,
                    threshold=debrief_threshold,
                ):
                    rollback_before_auto_debrief = (
                        _seal_workspace_rollback_point(ws)
                    )
                    snapshot_auto_debrief = True
                    try:
                        debrief_fn(ws, K)
                        levels2, path2, _ = verify_fn(game, solve_p)
                    except ProposerEvidenceUnavailable as ex:
                        # The auto-solve is independently replayed, but none of
                        # the unrecorded debrief's edits or probes may enter WIP
                        # or the promoted source.
                        _restore_workspace_rollback_point(
                            ws, rollback_before_auto_debrief
                        )
                        print(
                            f"EVIDENCE failure during debrief after level {K}: "
                            f"{ex}; discarding the debrief generation and "
                            "preserving the pre-debrief auto-solve"
                        )
                        levels2, path2, _ = verify_fn(game, solve_p)
                        if levels2 < K:
                            raise RuntimeError(
                                "sealed pre-debrief source no longer replays "
                                f"the auto-solved level {K}"
                            )
                        phase = "auto_solve_debrief_evidence_unavailable"
                        snapshot_auto_debrief = False
                        rollback_recovered_generation = True
                    except (CreditOut, ProposerInfrastructureError) as ex:
                        _restore_workspace_rollback_point(
                            ws, rollback_before_auto_debrief
                        )
                        label = (
                            "CREDIT-OUT"
                            if isinstance(ex, CreditOut)
                            else "INFRASTRUCTURE"
                        )
                        print(
                            f"{label} during debrief after level {K}: {ex}; "
                            "preserving solved level"
                        )
                        levels2, path2, _ = verify_fn(game, solve_p)
                        if levels2 < K:
                            raise RuntimeError(
                                "sealed pre-debrief source no longer replays "
                                f"the auto-solved level {K}"
                            )
                        phase = (
                            "auto_solve_debrief_credit_out"
                            if isinstance(ex, CreditOut)
                            else "auto_solve_debrief_infrastructure_failure"
                        )
                        rollback_recovered_generation = True
                    except BaseException:
                        _restore_workspace_rollback_point(
                            ws, rollback_before_auto_debrief
                        )
                        levels2, path2, _ = verify_fn(game, solve_p)
                        if levels2 < K:
                            raise RuntimeError(
                                "sealed pre-debrief source no longer replays "
                                f"the auto-solved level {K} after an unexpected "
                                "debrief failure"
                            )
                        raise
                    else:
                        phase = "after_auto_solve_debrief"
                        if levels2 < levels:
                            _restore_workspace_rollback_point(
                                ws, rollback_before_auto_debrief
                            )
                            levels2, path2, _ = verify_fn(game, solve_p)
                            if levels2 < K:
                                raise RuntimeError(
                                    "sealed pre-debrief auto-solve source "
                                    f"failed replay at level {K}"
                                )
                            phase = "auto_solve_debrief_regression_rolled_back"
                            rollback_recovered_generation = True
                        else:
                            authorized_winning_turn = codex_turn_records.get(
                                ("debrief", K)
                            )
                else:
                    levels2, path2 = levels, path
                    phase = "auto_solve_debrief_skipped"
                    snapshot_auto_debrief = True
                    if verbose:
                        print(f"level {K}: debrief skipped by {debrief_policy} policy")
                if snapshot_auto_debrief:
                    snapshot_wip_context(
                        game, ws, K, phase, max(levels, levels2), None, tag,
                        verbose=verbose,
                    )
                reached = max(levels, levels2)
                path = path2 if levels2 >= levels else path
                staged_levels, exact_path, sealed_promoted = (
                    _stage_and_replay_winning_tree(
                        game, ws, K, raw_verify_fn
                    )
                )
                reached = max(reached, staged_levels)
                Cm = marginal_complexity(
                    legs_b,
                    sealed_promoted["legs.py"].decode("utf-8"),
                    players_b,
                    sealed_promoted["players.py"].decode("utf-8"),
                )
                _record_level(rep, K, Cm)
                rep.reached = K
                rep.final_path = exact_path
                rep.validated = True
                _save_checkpoint(ws, rep)
                promote_verified_artifact(
                    game,
                    ws,
                    rep,
                    tag,
                    verbose=verbose,
                    sealed_promoted_payloads=sealed_promoted,
                    authorized_turn=authorized_winning_turn,
                )
                if verbose:
                    print(f"level {K}: reached={K} marginal_C={Cm} "
                          f"total_C={rep.total_marginal_C} validated={rep.validated} F={rep.free_energy:.3f}")
                if reached <= K - 1:
                    break
                continue

            # PHASE 1: proposer (existing legs could not solve the level). A transient
            # infrastructure failure (dropped connection, logged-out CLI that slipped
            # past the credit-out check) says nothing about the level, so it is retried;
            # only a real full-budget attempt that falls short stops the run.
            credit_out = False
            infrastructure_out = False
            pre_recovered = recover_discovered_path_artifact(
                game, ws, K, rep.final_path, verbose=verbose)
            if pre_recovered is not None:
                levels, path, err = pre_recovered
                snapshot_wip_context(game, ws, K, "recovered_existing_path_artifact",
                                     levels, None, tag, verbose=verbose)
            for attempt in range(0 if pre_recovered is not None else 1 + transient_retries):
                try:
                    propose_fn(ws, K)
                    assert_workspace_not_tainted(ws)
                except CreditOut as ex:
                    assert_workspace_not_tainted(ws)
                    print(f"CREDIT-OUT at level {K}: {ex}; stopping (reached={rep.reached})")
                    recovered = recover_discovered_path_artifact(
                        game, ws, K, rep.final_path, verbose=verbose)
                    _save_checkpoint(ws, rep)
                    if recovered is not None:
                        levels, path, err = recovered
                        credit_out = False
                        snapshot_wip_context(game, ws, K, "recovered_after_credit_out",
                                             levels, None, tag, verbose=verbose)
                        break
                    snapshot_wip_context(game, ws, K, "credit_out", rep.reached, str(ex), tag, verbose=verbose)
                    credit_out = True
                    break
                except (ProposerProtocolViolation,
                        ProposerBoundaryViolation,
                        WorkspaceTainted) as ex:
                    # A caught invalid action invalidates the complete generation.
                    # The protected transcript is the authority; do not
                    # recover, snapshot, retry, or promote any workspace byte.
                    print(
                        f"PROTOCOL TAINT at level {K}: {ex}; discarding "
                        "this proposer generation without WIP reuse or "
                        f"promotion (reached={rep.reached})"
                    )
                    protocol_tainted = True
                    protocol_taint_detail = str(ex)
                    break
                except ProposerEvidenceUnavailable as ex:
                    # Do not inspect, recover, retry, or snapshot anything the
                    # unrecorded turn wrote.  A retry in this same workspace
                    # would inherit knowledge that has no auditable transcript.
                    _save_checkpoint(ws, rep)
                    print(
                        f"EVIDENCE failure at level {K}: {ex}; discarding "
                        "this proposer generation without WIP reuse or "
                        f"promotion (reached={rep.reached})"
                    )
                    infrastructure_out = True
                    break
                except ProposerContainmentTimeout as ex:
                    assert_workspace_not_tainted(ws)
                    recovered = recover_discovered_path_artifact(
                        game, ws, K, rep.final_path, verbose=verbose
                    )
                    _save_checkpoint(ws, rep)
                    if recovered is not None:
                        levels, path, err = recovered
                        infrastructure_out = False
                        snapshot_wip_context(
                            game, ws, K, "recovered_after_containment_timeout",
                            levels, None, tag, verbose=verbose,
                        )
                        break
                    snapshot_wip_context(
                        game, ws, K, "containment_timeout", rep.reached,
                        str(ex), tag, verbose=verbose,
                    )
                    print(
                        f"CONTAINMENT at level {K}: {ex}; preserving clean "
                        "partial WIP without charging solver no-progress "
                        f"(reached={rep.reached})"
                    )
                    infrastructure_out = True
                    break
                except ProposerInfrastructureError as ex:
                    assert_workspace_not_tainted(ws)
                    recovered = recover_discovered_path_artifact(
                        game, ws, K, rep.final_path, verbose=verbose
                    )
                    _save_checkpoint(ws, rep)
                    if recovered is not None:
                        levels, path, err = recovered
                        infrastructure_out = False
                        snapshot_wip_context(
                            game, ws, K, "recovered_after_infrastructure_failure",
                            levels, None, tag, verbose=verbose,
                        )
                        break
                    snapshot_wip_context(
                        game, ws, K, "infrastructure_failure", rep.reached,
                        str(ex), tag, verbose=verbose,
                    )
                    if attempt < transient_retries:
                        if verbose:
                            print(
                                f"level {K}: transient infrastructure failure; "
                                "retrying without charging solver no-progress"
                            )
                        continue
                    print(
                        f"INFRASTRUCTURE at level {K}: {ex}; preserving WIP "
                        f"and stopping cleanly (reached={rep.reached})"
                    )
                    infrastructure_out = True
                    break
                rep = _adopt_workspace_checkpoint(game, ws, rep, verbose=verbose)
                recovered = recover_discovered_path_artifact(
                    game, ws, K, rep.final_path, verbose=verbose)
                _save_checkpoint(ws, rep)
                if recovered is not None:
                    levels, path, err = recovered
                    record_proposer_outcome(levels, path)
                    snapshot_wip_context(game, ws, K, "recovered_path_artifact",
                                         levels, None, tag, verbose=verbose)
                    break
                snapshot_wip_context(game, ws, K, "after_propose", rep.reached, None, tag, verbose=verbose)
                levels, path, err = verify_fn(game, solve_p)
                record_proposer_outcome(levels, path)
                if levels >= K:
                    break
                code_changed = (_read(legs_p) != legs_b or _read(players_p) != players_b)
                if attempt < transient_retries and _transient_proposer_failure(ws, code_changed):
                    if verbose:
                        print(f"level {K}: transient proposer failure (see proposer_last.log); retrying")
                    continue
                snapshot_wip_context(game, ws, K, "not_reached", levels, err, tag, verbose=verbose)
                if verbose:
                    print(f"level {K}: NOT reached (got {levels}, err={err}); stopping")
                break
            if (
                credit_out
                or infrastructure_out
                or protocol_tainted
                or levels < K
            ):
                break
            snapshot_wip_context(game, ws, K, "reached_before_debrief", levels, err, tag, verbose=verbose)
            authorized_winning_turn = codex_turn_records.get(("propose", K))
            rollback_before_debrief = _seal_workspace_rollback_point(ws)
            pre_debrief_marginal = marginal_complexity(
                legs_b, _read(legs_p), players_b, _read(players_p)
            )
            run_debrief = should_run_debrief(
                debrief_policy,
                auto_solved=False,
                pre_debrief_marginal_C=pre_debrief_marginal,
                threshold=debrief_threshold,
            )
            if not run_debrief:
                levels2, path2 = levels, path
                snapshot_wip_context(
                    game, ws, K, "debrief_skipped_policy", levels, None, tag,
                    verbose=verbose,
                )
                if verbose:
                    print(
                        f"level {K}: debrief skipped by {debrief_policy} policy "
                        f"(pre-debrief marginal_C={pre_debrief_marginal})"
                    )
            else:
                try:
                    debrief_fn(ws, K)
                    levels2, path2, _ = verify_fn(game, solve_p)  # behaviour preserved?
                except ProposerEvidenceUnavailable as ex:
                    # The winning proposal has its own sealed transcript.
                    # Restore its exact source and admit no bytes learned by
                    # the unrecorded optional debrief.
                    _restore_workspace_rollback_point(
                        ws, rollback_before_debrief
                    )
                    print(
                        f"EVIDENCE failure during debrief after level {K}: "
                        f"{ex}; discarding the debrief generation and "
                        "preserving the pre-debrief win"
                    )
                    levels2, path2, _ = verify_fn(game, solve_p)
                    if levels2 < K:
                        raise RuntimeError(
                            "sealed pre-debrief source no longer replays "
                            f"the proposed win at level {K}"
                        )
                    rollback_recovered_generation = True
                except (CreditOut, ProposerInfrastructureError) as ex:
                    _restore_workspace_rollback_point(
                        ws, rollback_before_debrief
                    )
                    label = (
                        "CREDIT-OUT"
                        if isinstance(ex, CreditOut)
                        else "INFRASTRUCTURE"
                    )
                    print(
                        f"{label} during debrief after level {K}: {ex}; "
                        "preserving solved level"
                    )
                    levels2, path2, _ = verify_fn(game, solve_p)
                    if levels2 < K:
                        raise RuntimeError(
                            "sealed pre-debrief source no longer replays "
                            f"the proposed win at level {K}"
                        )
                    phase = (
                        "debrief_credit_out"
                        if isinstance(ex, CreditOut)
                        else "debrief_infrastructure_failure"
                    )
                    snapshot_wip_context(
                        game, ws, K, phase, levels, str(ex), tag,
                        verbose=verbose,
                    )
                    rollback_recovered_generation = True
                except BaseException:
                    _restore_workspace_rollback_point(
                        ws, rollback_before_debrief
                    )
                    levels2, path2, _ = verify_fn(game, solve_p)
                    if levels2 < K:
                        raise RuntimeError(
                            "sealed pre-debrief source no longer replays "
                            f"the proposed win at level {K} after an unexpected "
                            "debrief failure"
                        )
                    raise
                else:
                    if levels2 < levels:
                        _restore_workspace_rollback_point(
                            ws, rollback_before_debrief
                        )
                        levels2, path2, _ = verify_fn(game, solve_p)
                        if levels2 < K:
                            raise RuntimeError(
                                "sealed pre-debrief proposal source failed "
                                f"replay at level {K}"
                            )
                        if verbose:
                            print(
                                f"level {K}: debrief regressed solve; "
                                "restored and replayed the complete pre-debrief tree"
                            )
                        debrief_phase = "debrief_regression_rolled_back"
                    else:
                        debrief_phase = "after_debrief"
                        authorized_winning_turn = codex_turn_records.get(
                            ("debrief", K)
                        )
                    snapshot_wip_context(
                        game, ws, K, debrief_phase, levels2, None, tag,
                        verbose=verbose,
                    )
            reached = max(levels, levels2)
            path = path2 if levels2 >= levels else path
            staged_levels, exact_path, sealed_promoted = (
                _stage_and_replay_winning_tree(
                    game, ws, K, raw_verify_fn
                )
            )
            reached = max(reached, staged_levels)
            Cm = marginal_complexity(
                legs_b,
                sealed_promoted["legs.py"].decode("utf-8"),
                players_b,
                sealed_promoted["players.py"].decode("utf-8"),
            )
            _record_level(rep, K, Cm)
            rep.reached = K
            rep.final_path = exact_path
            rep.validated = True
            _save_checkpoint(ws, rep)
            promote_verified_artifact(
                game,
                ws,
                rep,
                tag,
                verbose=verbose,
                sealed_promoted_payloads=sealed_promoted,
                authorized_turn=authorized_winning_turn,
            )
            if verbose:
                print(f"level {K}: reached={K} marginal_C={Cm} "
                      f"total_C={rep.total_marginal_C} validated={rep.validated} F={rep.free_energy:.3f}")
            if rollback_recovered_generation:
                if verbose:
                    print(
                        "ending this workspace generation after rollback-"
                        "recovered promotion; the scheduler must seed the next "
                        "level into a fresh clean workspace"
                    )
                break
            if reached <= K - 1:
                break

    except WorkspaceTainted as ex:
        # A defense-in-depth scan can fail inside a sibling exception handler
        # (credit, containment, or infrastructure) as well as after a normal
        # proposer return.  Converge every such path on the same terminal
        # generation-taint result; never inspect, recover, snapshot, or
        # promote its bytes.
        protocol_tainted = True
        protocol_taint_detail = str(ex)
        if verbose:
            print(
                f"PROTOCOL TAINT: {ex}; discarding this complete proposer "
                "generation without WIP reuse or promotion"
            )
    except KeyboardInterrupt:
        taint_reason = _workspace_or_protected_taint_reason(ws)
        if taint_reason:
            if verbose:
                print(
                    "interrupted workspace is tainted; suppressing resumable "
                    f"WIP snapshot: {taint_reason}"
                )
        else:
            snapshot_wip_context(
                game, ws, rep.reached + 1, "interrupted",
                rep.reached, "interrupted mid-level", tag, verbose=verbose,
            )
        interrupted = True
    except BaseException:
        _release_workspace_lock(run_lock)
        _release_workspace_lock(lineage_lock)
        raise

    if protocol_tainted:
        _release_workspace_lock(run_lock)
        _release_workspace_lock(lineage_lock)
        raise WorkspaceTainted(
            "clean-room taint invalidated the complete proposer generation; "
            "no WIP or promotion was written"
            + (
                f": {protocol_taint_detail}"
                if protocol_taint_detail
                else ""
            )
        )

    rep.validated = A.validate(game, rep.final_path, rep.reached) if rep.final_path else False
    _save_checkpoint(ws, rep)
    promote_verified_artifact(game, ws, rep, tag, verbose=verbose)
    if verbose:
        print(f"\n=== {game}: reached level {rep.reached} | validated={rep.validated} | "
              f"total_marginal_C={rep.total_marginal_C} | F={rep.free_energy:.3f} ===")
        print("  per-level marginal novelty (should trend DOWN as legs are reused): "
              + ", ".join(f"L{r.level}:{r.marginal_C}" for r in rep.records))
    _release_workspace_lock(run_lock)
    _release_workspace_lock(lineage_lock)
    if interrupted:
        raise KeyboardInterrupt
    return rep


def _defs(code: str):
    """Top-level function names defined in a module (the leg index)."""
    import ast
    try:
        return sorted(n.name for n in ast.parse(code or "").body
                      if isinstance(n, ast.FunctionDef))
    except Exception:
        return []


if __name__ == "__main__":
    import sys
    cli_usage = (
        "usage: gkm_legs.py --game=GAME --max-level=N "
        "[--proposer=codex] [--model=MODEL] [--minutes=N] "
        "[--codex-effort=medium|high|xhigh|max] "
        "[--codex-allocation-policy=hard|drain] "
        "[--artifacts-root=PATH] [--seed-mode=zero_seed|verified_parent] "
        "[--wip-mode=exclude|restore_clean_same_frontier] "
        "[--expected-parent-reached=N "
        "--expected-parent-action-count=N "
        "--expected-parent-checkpoint-sha256=HEX "
        "--expected-parent-source-tree-sha256=HEX "
        "--expected-frontier-sha256=HEX] "
        "[--expected-wip-attempt=ATTEMPT]"
    )
    game, model, minutes, maxl, proposer, tag = "wa30", None, 40, 9, "opencode", ""
    fresh, restore_wip = False, True
    seed_mode, wip_mode = None, None
    codex_effort, codex_debrief_effort, codex_weekly_reserve = "medium", "medium", 80
    codex_allocation_policy = DEFAULT_CODEX_ALLOCATION_POLICY
    debrief_policy, debrief_threshold = "always", 150
    codex_weekly_headroom = 1
    codex_max_campaign_tokens, codex_max_campaign_runs = 2_000_000, 12
    transient_retries = _TRANSIENT_RETRIES
    codex_ledger = None
    expected_frontier_parts = {}
    expected_wip_attempt = None
    claude_guard, claude_ledger, claude_max_turns, claude_max_wall_minutes = False, None, None, None
    claude_window_hours = CLG.DEFAULT_WINDOW_HOURS
    api_guard, api_ledger, api_max_cost_usd, api_max_turns, api_max_wall_minutes = False, None, None, None, None
    api_window_hours = CLG.DEFAULT_WINDOW_HOURS
    seen_cli_options = set()
    for a in sys.argv[1:]:
        if a in {"-h", "--help"}:
            print(cli_usage)
            raise SystemExit(0)
        option = a.split("=", 1)[0]
        if option in seen_cli_options:
            raise SystemExit(f"duplicate argument: {option}\n{cli_usage}")
        seen_cli_options.add(option)
        if a.startswith("--game="): game = a.split("=", 1)[1]
        elif a.startswith("--model="): model = a.split("=", 1)[1]
        elif a.startswith("--minutes="): minutes = int(a.split("=", 1)[1])
        elif a.startswith("--max-level="): maxl = int(a.split("=", 1)[1])
        elif a.startswith("--proposer="): proposer = a.split("=", 1)[1]
        elif a.startswith("--tag="): tag = a.split("=", 1)[1]
        elif a.startswith("--artifacts-root="):
            os.environ["GKM_ARTIFACTS_ROOT"] = a.split("=", 1)[1]
        elif a.startswith("--codex-effort="): codex_effort = a.split("=", 1)[1]
        elif a.startswith("--codex-debrief-effort="): codex_debrief_effort = a.split("=", 1)[1]
        elif a.startswith("--codex-allocation-policy="): codex_allocation_policy = a.split("=", 1)[1]
        elif a.startswith("--debrief-policy="): debrief_policy = a.split("=", 1)[1]
        elif a.startswith("--debrief-threshold="): debrief_threshold = int(a.split("=", 1)[1])
        elif a.startswith("--codex-weekly-reserve="): codex_weekly_reserve = int(a.split("=", 1)[1])
        elif a.startswith("--codex-weekly-headroom="): codex_weekly_headroom = int(a.split("=", 1)[1])
        elif a.startswith("--codex-max-campaign-tokens="): codex_max_campaign_tokens = int(a.split("=", 1)[1])
        elif a.startswith("--codex-max-campaign-runs="): codex_max_campaign_runs = int(a.split("=", 1)[1])
        elif a.startswith("--transient-retries="): transient_retries = int(a.split("=", 1)[1])
        elif a.startswith("--codex-ledger="): codex_ledger = a.split("=", 1)[1]
        elif a.startswith("--expected-parent-reached="):
            expected_frontier_parts["reached"] = int(a.split("=", 1)[1])
        elif a.startswith("--expected-parent-action-count="):
            expected_frontier_parts["parent_action_count"] = int(
                a.split("=", 1)[1]
            )
        elif a.startswith("--expected-parent-checkpoint-sha256="):
            expected_frontier_parts["parent_checkpoint_sha256"] = a.split(
                "=", 1
            )[1]
        elif a.startswith("--expected-parent-source-tree-sha256="):
            expected_frontier_parts["parent_source_tree_sha256"] = a.split(
                "=", 1
            )[1]
        elif a.startswith("--expected-frontier-sha256="):
            expected_frontier_parts["frontier_sha256"] = a.split("=", 1)[1]
        elif a.startswith("--expected-wip-attempt="):
            expected_wip_attempt = a.split("=", 1)[1]
        elif a == "--claude-guard": claude_guard = True
        elif a.startswith("--claude-ledger="): claude_ledger = a.split("=", 1)[1]
        elif a.startswith("--claude-window-hours="): claude_window_hours = float(a.split("=", 1)[1])
        elif a.startswith("--claude-max-turns="): claude_max_turns = int(a.split("=", 1)[1])
        elif a.startswith("--claude-max-wall-minutes="): claude_max_wall_minutes = float(a.split("=", 1)[1])
        elif a == "--api-guard": api_guard = True
        elif a.startswith("--api-ledger="): api_ledger = a.split("=", 1)[1]
        elif a.startswith("--api-window-hours="): api_window_hours = float(a.split("=", 1)[1])
        elif a.startswith("--api-max-cost-usd="): api_max_cost_usd = float(a.split("=", 1)[1])
        elif a.startswith("--api-max-turns="): api_max_turns = int(a.split("=", 1)[1])
        elif a.startswith("--api-max-wall-minutes="): api_max_wall_minutes = float(a.split("=", 1)[1])
        elif a == "--fresh": fresh = True
        elif a == "--no-wip-restore": restore_wip = False
        elif a.startswith("--seed-mode="): seed_mode = a.split("=", 1)[1]
        elif a.startswith("--wip-mode="): wip_mode = a.split("=", 1)[1]
        else:
            raise SystemExit(f"unknown argument: {a}\n{cli_usage}")
    required_cli_options = {
        "--game", "--max-level", "--proposer", "--model", "--minutes",
    }
    missing_cli_options = sorted(required_cli_options - seen_cli_options)
    if missing_cli_options:
        raise SystemExit(
            f"missing required arguments: {missing_cli_options}\n{cli_usage}"
        )
    if not game or not model or proposer not in {
        "claude", "opencode", "codex", "api",
    }:
        raise SystemExit(
            "game/model must be nonempty and proposer must be one of "
            "claude, opencode, codex, api"
        )
    if minutes <= 0:
        raise SystemExit("--minutes must be positive")
    if seed_mode is not None:
        if seed_mode not in {"zero_seed", "verified_parent"}:
            raise SystemExit(f"invalid --seed-mode={seed_mode}")
        if fresh:
            raise SystemExit("--fresh cannot be combined with explicit --seed-mode")
        if seed_mode == "zero_seed":
            existing = _load_checkpoint(artifact_dir(game, tag))
            if existing is not None and existing.validated and existing.reached > 0:
                raise SystemExit(
                    "zero_seed requested beside an existing validated artifact; "
                    "use a new artifact root"
                )
            # With WIP restore, the normal empty-artifact seed path restores L1
            # context. Without WIP, skip artifact seeding entirely.
            fresh = wip_mode != "restore_clean_same_frontier"
        else:
            fresh = False
    if wip_mode is not None:
        if wip_mode not in {"exclude", "restore_clean_same_frontier"}:
            raise SystemExit(f"invalid --wip-mode={wip_mode}")
        if wip_mode == "exclude":
            restore_wip = False
        else:
            restore_wip = True
    if expected_wip_attempt is not None and (
        wip_mode != "restore_clean_same_frontier"
    ):
        raise SystemExit(
            "--expected-wip-attempt requires "
            "--wip-mode=restore_clean_same_frontier"
        )
    expected_frontier_binding = None
    if expected_frontier_parts:
        required_expected = {
            "reached",
            "parent_action_count",
            "parent_checkpoint_sha256",
            "parent_source_tree_sha256",
            "frontier_sha256",
        }
        missing_expected = sorted(
            required_expected - set(expected_frontier_parts)
        )
        if missing_expected:
            raise SystemExit(
                "partial exact-frontier binding is forbidden; missing "
                f"{missing_expected}\n{cli_usage}"
            )
        expected_frontier_binding = {
            "frontier_binding_schema": CCS.FRONTIER_BINDING_SCHEMA,
            "game": game,
            "reached": expected_frontier_parts["reached"],
            "target_level": maxl,
            "parent_action_count":
                expected_frontier_parts["parent_action_count"],
            "parent_checkpoint_sha256":
                expected_frontier_parts["parent_checkpoint_sha256"],
            "parent_source_tree_sha256":
                expected_frontier_parts["parent_source_tree_sha256"],
            "frontier_sha256":
                expected_frontier_parts["frontier_sha256"],
        }
        try:
            expected_frontier_binding = CCS.validate_frontier_binding(
                expected_frontier_binding,
                expected_game=game,
                expected_target_level=maxl,
            )
        except ValueError as exc:
            raise SystemExit(
                f"invalid exact-frontier binding: {exc}\n{cli_usage}"
            ) from exc
    orchestrate(game=game, max_level=maxl, proposer=proposer, model=model,
                minutes_per=minutes, tag=tag, seed_artifact=not fresh,
                restore_wip=restore_wip, codex_effort=codex_effort,
                codex_debrief_effort=codex_debrief_effort,
                codex_allocation_policy=codex_allocation_policy,
                debrief_policy=debrief_policy,
                debrief_threshold=debrief_threshold,
                codex_weekly_reserve=codex_weekly_reserve,
                codex_weekly_headroom=codex_weekly_headroom,
                codex_max_campaign_tokens=codex_max_campaign_tokens,
                codex_max_campaign_runs=codex_max_campaign_runs,
                transient_retries=transient_retries,
                codex_ledger=codex_ledger,
                expected_frontier_binding=expected_frontier_binding,
                expected_wip_attempt=expected_wip_attempt,
                claude_guard=claude_guard,
                claude_ledger=claude_ledger,
                claude_window_hours=claude_window_hours,
                claude_max_turns=claude_max_turns,
                claude_max_wall_minutes=claude_max_wall_minutes,
                api_guard=api_guard,
                api_ledger=api_ledger,
                api_window_hours=api_window_hours,
                api_max_cost_usd=api_max_cost_usd,
                api_max_turns=api_max_turns,
                api_max_wall_minutes=api_max_wall_minutes)
MARGINAL_COMPLEXITY_CONTRACT = {
    "field": "marginal_C",
    "label": "positive net retained-description growth per source file",
    "formula": (
        "max(0, d(legs_after)-d(legs_before)) + "
        "max(0, d(players_after)-d(players_before))"
    ),
    "limitation": (
        "additions and deletions within the same file are netted before the "
        "positive part, so same-size replacement can receive zero"
    ),
}
