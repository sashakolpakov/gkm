#!/usr/bin/env python3
"""Audit promoted ARC artifacts without conflating them with forensic WIP.

The default report has three disjoint ledgers:

* ``canonical``: only files at each ``*_legs`` artifact root;
* ``successful_candidate_wip``: snapshots whose metadata says the active level
  was reached (these remain WIP, not proof of promotion);
* ``discarded_wip``: failed, interrupted, credit-out, or otherwise unverified
  snapshots.
* ``frontier_scaffolds``: reviewed level-scoped context that will be copied into a
  future clean room.
* ``proposer_containment``: immutable command/file-change surfaces, kept
  distinct from gameplay taint so parent-repository metadata exposure cannot be
  silently called clean or overstated as source-assisted solving.

A protected public-action protocol marker is also release-blocking, but is
reported by its own ``public_action_protocol_violation`` kind rather than being
described as source, environment, or game-description taint.

No WIP finding is propagated into the canonical verdict.  Historical ancestry
requires an explicit promotion manifest; file adjacency is not ancestry.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import tokenize
from pathlib import Path


PROMOTED_FILES = {
    "legs.py", "players.py", "solve.py", "legs_log.md", "checkpoint.json",
    "auto_solve_attempts.json",
}
MAX_TAINT_SCAN_BYTES = 50_000_000
GAME_SOURCE_NAMES = {
    f"{game}.py" for game in (
        "ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0 "
        "r11l re86 s5i5 sb26 sc25 sk48 sp80 su15 tn36 tr87 tu93 vc33 wa30"
    ).split()
}
HIDDEN_SOURCE_RE = re.compile(
    r"environment_files/|/environment_files/|agent_solutions/|/agent_solutions/|"
    + "|".join(re.escape(name) for name in sorted(GAME_SOURCE_NAMES)),
    re.IGNORECASE,
)
NETWORK_RE = re.compile(
    r"(?:^|[\n;&|])\s*(?:sudo\s+)?(?:curl|wget|lynx|links|nc|ncat|netcat|"
    r"telnet|ssh|scp|rsync)(?!\s*=)\s+"
    r"|\b(?:web[_ -]?search|browser\.open|search_query|open_url)\b"
    r"|\b(?:requests|httpx|aiohttp|urllib\.request|http\.client)\."
    r"|\bsocket\.(?:create_connection|socket|getaddrinfo|gethostbyname)\b"
    r"|https?://(?!localhost(?::\d+)?(?:/|\b)|127\.0\.0\.1(?::\d+)?(?:/|\b)|"
    r"\[?::1\]?(?::\d+)?(?:/|\b))",
    re.IGNORECASE,
)
PRIVATE_RUNTIME_RE = re.compile(
    r"\.\s*_(?:game|env|fd)\b|\bvars\s*\(\s*env\b|"
    r"object\.__getattribute__\s*\(\s*env\b",
    re.IGNORECASE,
)
PUBLIC_ACTION_PROTOCOL_VIOLATION_RE = re.compile(
    r"\bGKM_PUBLIC_ACTION_PROTOCOL_VIOLATION\b"
    r"|\bPublicActionProtocolViolation\b"
    r"|coordinate action requires integer x,y in 0\.\.63",
    re.IGNORECASE,
)
HARNESS_INTROSPECTION_RE = re.compile(
    r"inspect\.getsource\s*\(\s*(?:A\.|gkm_arena)|"
    r"\bdir\s*\(\s*(?:A|gkm_arena)(?:\.|\s*\))|"
    r"\bdir\s*\(\s*env\b|\.\s*_budget\b",
    re.IGNORECASE,
)
HOST_PROCESS_INTROSPECTION_RE = re.compile(
    r"(?:^|[\n;&|'\"|])\s*(?:sudo\s+)?"
    r"(?:ps|pgrep|pkill|lsof|top|htop|launchctl|systemctl)"
    r"(?:\s|$|(?=['\"]))",
    re.IGNORECASE,
)
OPERATIONAL_PROCESS_MONITORING_RE = re.compile(
    r"(?:ps|pgrep)\b[^\n]*\|\s*rg\s+['\"][^'\"]*"
    r"[A-Za-z0-9_.-]+\.py(?:\s|['\"])"
    r"|\bpgrep\s+-[A-Za-z]*f[A-Za-z]*\s+"
    r"['\"][A-Za-z0-9_.-]+\.py"
    r"(?:\s+[A-Za-z0-9_.-]+)*['\"]",
    re.IGNORECASE,
)
HOST_PROCESS_COMMAND_WORD_RE = re.compile(
    r"\b(?:ps|pgrep|pkill|lsof|top|htop|launchctl|systemctl)\b",
    re.IGNORECASE,
)
PYTHON_HEREDOC_START_RE = re.compile(
    r"\bpython(?:3(?:\.\d+)*)?\b[^\n]*?<<-?\s*"
    r"(?P<quote>['\"]?)(?P<delimiter>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?P=quote)[^\n]*\n",
    re.IGNORECASE,
)
PYTHON_PROCESS_LITERAL_CONTEXT_RE = re.compile(
    r"\b(?:subprocess\.(?:run|Popen|call|check_call|check_output)"
    r"|os\.(?:system|popen)|Popen)\s*\([^)\n]{0,256}$",
    re.IGNORECASE,
)
SHELL_COMMAND_STRING_CONTEXT_RE = re.compile(
    r"(?:^|\s)-[A-Za-z]*c\s*$",
    re.IGNORECASE,
)
INFORMATIONAL_HITS = {
    "harness_introspection",
    "operational_process_monitoring",
}
PARENT_REFERENCE_RE = re.compile(
    r"(?:^|[\s'\"=])\.\.(?:/|\\)",
)
HOST_FILESYSTEM_ENUMERATION_RE = re.compile(
    r"(?:^|[\n;&|]\s*)(?:find|tree|du|ls)\s+"
    r"(?:-[^\n;&|]+\s+)*(?:/|~|\.\.(?:/|\s|$))",
    re.IGNORECASE,
)
SUPERVISOR_INPUT_COMMAND_RE = re.compile(
    r"ARC_AGI3_CAMPAIGN_PLAN(?:\.md)?"
    r"|(?:^|[/\s])quarantined_attempts(?:[/\s]|$)"
    r"|(?:^|[/\s])agent_solutions(?:[/\s]|$)"
    r"|(?:^|[/\s])candidate_solutions(?:[/\s]|$)"
    r"|(?:^|[/\s])manuscript(?:[/\s]|$)"
    r"|\bcodex_campaign_(?:policy|runner|status)\.py\b"
    r"|\barc_agi3_contiguous_[A-Za-z0-9_]+\.py\b",
    re.IGNORECASE,
)
PARENT_GIT_METADATA_OUTPUT_RE = re.compile(
    r"(?m)^\s*(?:README\.md|REPRODUCE_ARC\.md|"
    r"arc/(?:README\.md|audit_[^/\s]+|crack_lab/|manuscript/)|"
    r"docs/|bongard/|foraging/|transduction/)\S*[ \t]+\|",
)


def _python_heredoc_name_spans(text: str) -> set[tuple[int, int]]:
    """Locate unquoted Python identifiers inside shell-recorded heredocs.

    The process scanner operates on complete agent-authored shell commands.
    Those commands often contain Python heredocs, where expressions such as
    ``bridges | pegs`` must not be mistaken for a shell pipe into ``ps``.
    Literal process commands remain visible because they are STRING tokens,
    not NAME tokens.
    """
    spans: set[tuple[int, int]] = set()
    search_from = 0
    while True:
        start = PYTHON_HEREDOC_START_RE.search(text, search_from)
        if start is None:
            break
        delimiter = re.escape(start.group("delimiter"))
        end = re.search(
            rf"(?m)^\t*{delimiter}(?:['\"])?"
            rf"(?:\s*(?:[;&|].*)?)?$",
            text[start.end():],
        )
        if end is None:
            search_from = start.end()
            continue
        body_start = start.end()
        body_end = body_start + end.start()
        body = text[body_start:body_end]
        line_offsets = [0]
        for match in re.finditer(r"\n", body):
            line_offsets.append(match.end())
        try:
            tokens = tokenize.generate_tokens(io.StringIO(body).readline)
            for token in tokens:
                if token.type != tokenize.NAME:
                    continue
                start_line, start_column = token.start
                end_line, end_column = token.end
                absolute_start = (
                    body_start + line_offsets[start_line - 1] + start_column
                )
                absolute_end = (
                    body_start + line_offsets[end_line - 1] + end_column
                )
                spans.add((absolute_start, absolute_end))
        except (IndentationError, tokenize.TokenError):
            # An incomplete investigative heredoc remains conservatively
            # scanned as shell text rather than receiving an exception.
            pass
        search_from = body_start + end.end()
    return spans


def _quoted_host_word_is_data(
    text: str,
    word_match: re.Match[str],
) -> bool:
    """Distinguish labels such as AWK's ``"top"`` from process commands."""
    start, end = word_match.span()
    if start == 0 or end >= len(text):
        return False
    quote = text[start - 1]
    if quote not in {"'", '"'} or text[end] != quote:
        return False
    prefix = text[max(0, start - 320):start - 1]
    if prefix.rstrip().endswith(("|", ";", "&")):
        return False
    if SHELL_COMMAND_STRING_CONTEXT_RE.search(prefix):
        return False
    if PYTHON_PROCESS_LITERAL_CONTEXT_RE.search(prefix):
        return False
    return True


def has_forbidden_host_process_command(text: str) -> bool:
    """Reject any forbidden command even beside an allowed own-worker query."""
    python_names = _python_heredoc_name_spans(text)
    for host_match in HOST_PROCESS_INTROSPECTION_RE.finditer(text):
        word_match = HOST_PROCESS_COMMAND_WORD_RE.search(
            text, host_match.start(), host_match.end()
        )
        if word_match is None:
            return True
        if (word_match.start(), word_match.end()) in python_names:
            continue
        if _quoted_host_word_is_data(text, word_match):
            continue
        command = word_match.group(0).lower()
        allowed = OPERATIONAL_PROCESS_MONITORING_RE.match(
            text, word_match.start()
        )
        if command in {"ps", "pgrep"} and allowed is not None:
            continue
        return True
    return False


def scan_text(
    text: str,
    *,
    strip_inline_code: bool = True,
    execution_surface: bool = False,
) -> list[str]:
    if strip_inline_code:
        text = re.sub(r"`[^`\n]*`", "", text)
    hits = []
    for label, pattern in (
        ("hidden_source_or_prior_solution", HIDDEN_SOURCE_RE),
        ("external_web_or_network", NETWORK_RE),
        ("direct_private_runtime", PRIVATE_RUNTIME_RE),
        ("harness_introspection", HARNESS_INTROSPECTION_RE),
    ):
        if pattern.search(text):
            hits.append(label)
    if execution_surface and HOST_PROCESS_INTROSPECTION_RE.search(text):
        if OPERATIONAL_PROCESS_MONITORING_RE.search(text):
            hits.append("operational_process_monitoring")
        if has_forbidden_host_process_command(text):
            hits.append("host_process_introspection")
    return hits


def codex_execution_surface(text: str) -> str | None:
    """Extract agent-authored actions from an immutable Codex JSONL transcript.

    Tool output is not an action.  In particular, a traceback emitted by the
    public ``env.clone()`` operation can contain the harness implementation's
    private field names.  The requested command, web-search item, and changed
    file paths remain separately available in the same transcript, while file
    contents are scanned from the corresponding evidence snapshot.
    """
    values: list[str] = []
    parsed = 0
    diagnostics = []
    passive_event_types = {
        "thread.started",
        "turn.started",
        "turn.completed",
    }
    passive_item_types = {"agent_message", "reasoning"}
    for raw in text.splitlines():
        if not raw.strip():
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            diagnostics.append(raw)
            continue
        parsed += 1
        if not isinstance(event, dict):
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
            values.append(json.dumps(item, sort_keys=True))
        elif item_type not in passive_item_types:
            # Unknown structured records are not trusted merely because they
            # parse as JSON. Scan them whole so a future/malformed command
            # schema cannot become a taint bypass.
            values.append(raw)
    # Long/concatenated Codex turns may contain many CLI diagnostics. Scan those
    # lines themselves, but never use their count as a reason to fall back to
    # JSON-escaped tool output (which can contain private field names from a
    # public clone traceback).
    if parsed:
        return "\n".join(values + diagnostics)
    return None


def scan_file(path: Path) -> list[str]:
    try:
        if path.stat().st_size > MAX_TAINT_SCAN_BYTES:
            return ["oversized_unscanned_evidence"]
        text = path.read_text(encoding="utf-8", errors="ignore")
        protocol_violation = (
            path.name == "proposer_last.log" or path.suffix == ".jsonl"
        ) and PUBLIC_ACTION_PROTOCOL_VIOLATION_RE.search(text) is not None
        surface = (
            codex_execution_surface(text)
            if path.name == "proposer_last.log" or path.suffix == ".jsonl"
            else None
        )
        if surface is not None:
            hits = scan_text(
                surface,
                strip_inline_code=False,
                execution_surface=True,
            )
        else:
            hits = scan_text(text)
        if protocol_violation:
            hits.insert(0, "public_action_protocol_violation")
        return hits
    except OSError:
        return []


def _file_change_is_workspace_local(value: object) -> bool:
    """Accept relative paths and absolute paths rooted in one clean workspace."""
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    if ".." in path.parts:
        return False
    if not path.is_absolute():
        return True
    return any(part.startswith("gkm_legs_ws_") for part in path.parts)


def audit_transcript_containment(root: Path) -> dict:
    """Audit proposer commands separately from gameplay/source taint.

    This catches containment defects such as Git walking into a parent
    repository. Such metadata exposure is not mislabeled as helpful game-source
    taint, but it is still a release-blocking acquisition-boundary incident.
    """
    result = {
        "transcript_paths": 0,
        "unique_transcripts": 0,
        "completed_commands": 0,
        "completed_file_changes": 0,
        "web_search_events": 0,
        "public_action_protocol_violations": 0,
        "incidents": [],
    }
    paths = sorted({
        *root.rglob("*.jsonl"),
        *root.rglob("proposer_last.log"),
    })
    result["transcript_paths"] = len(paths)
    seen_hashes: set[str] = set()
    for path in paths:
        try:
            payload = path.read_bytes()
        except OSError:
            continue
        digest = hashlib.sha256(payload).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        result["unique_transcripts"] += 1
        for line_number, raw in enumerate(
            payload.decode("utf-8", errors="replace").splitlines(), 1
        ):
            if PUBLIC_ACTION_PROTOCOL_VIOLATION_RE.search(raw):
                result["public_action_protocol_violations"] += 1
                result["incidents"].append({
                    "kind": "public_action_protocol_violation",
                    "path": str(path),
                    "line": line_number,
                    "event_sha256": hashlib.sha256(
                        raw.encode("utf-8")
                    ).hexdigest(),
                })
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            item = event.get("item")
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "web_search":
                result["web_search_events"] += 1
                result["incidents"].append({
                    "kind": "web_search_event",
                    "path": str(path),
                    "line": line_number,
                })
                continue
            if event.get("type") != "item.completed":
                continue
            if item_type == "file_change":
                changes = item.get("changes", [])
                if not isinstance(changes, list):
                    result["incidents"].append({
                        "kind": "malformed_file_change",
                        "path": str(path),
                        "line": line_number,
                    })
                    continue
                for change in changes:
                    result["completed_file_changes"] += 1
                    changed_path = (
                        change.get("path")
                        if isinstance(change, dict)
                        else None
                    )
                    if not _file_change_is_workspace_local(changed_path):
                        result["incidents"].append({
                            "kind": "file_change_outside_clean_workspace",
                            "path": str(path),
                            "line": line_number,
                            "changed_path": changed_path,
                        })
                continue
            if item_type != "command_execution":
                continue
            command = item.get("command")
            if not isinstance(command, str):
                result["incidents"].append({
                    "kind": "malformed_command",
                    "path": str(path),
                    "line": line_number,
                })
                continue
            result["completed_commands"] += 1
            command_excerpt = command[:800]
            for kind, pattern in (
                ("parent_path_command", PARENT_REFERENCE_RE),
                ("host_filesystem_enumeration", HOST_FILESYSTEM_ENUMERATION_RE),
                ("supervisor_input_command", SUPERVISOR_INPUT_COMMAND_RE),
            ):
                if pattern.search(command):
                    result["incidents"].append({
                        "kind": kind,
                        "path": str(path),
                        "line": line_number,
                        "command": command_excerpt,
                    })
            output = item.get("aggregated_output", "")
            if (
                isinstance(output, str)
                and re.search(r"(?:^|[\s;&|])git\s", command)
                and PARENT_GIT_METADATA_OUTPUT_RE.search(output)
            ):
                result["incidents"].append({
                    "kind": "parent_git_metadata_exposure",
                    "path": str(path),
                    "line": line_number,
                    "command": command_excerpt,
                    "output_sha256": hashlib.sha256(
                        output.encode("utf-8")
                    ).hexdigest(),
                })
    result["verdict"] = "clean" if not result["incidents"] else "incident"
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_evidence_path(root: Path, value: object) -> Path | None:
    """Resolve one declared evidence path without permitting root escape."""
    if not isinstance(value, str) or not value:
        return None
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    return root / relative


def audit_promotion_chain(artifact: Path) -> dict:
    manifests = sorted((artifact / "promotion_evidence").glob("level_*/manifest.json"))
    checkpoint_path = artifact / "checkpoint.json"
    expected_reached = None
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        reached = checkpoint.get("reached")
        if isinstance(reached, int) and not isinstance(reached, bool) and reached >= 0:
            expected_reached = reached
    except (OSError, json.JSONDecodeError):
        pass
    result = {
        "manifests": len(manifests),
        "expected_reached": expected_reached,
        "manifest_levels": [],
        "missing_levels": [],
        "unexpected_levels": [],
        "complete": False,
        "integrity_errors": [],
        "taint_hits": [],
        "informational_harness_introspection": [],
        "informational_operational_monitoring": [],
    }
    previous = None
    for manifest_path in manifests:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            result["integrity_errors"].append(f"{manifest_path}: {exc}")
            continue
        evidence_dir = manifest_path.parent
        try:
            path_level = int(evidence_dir.name.removeprefix("level_"))
        except ValueError:
            path_level = None
        manifest_level = manifest.get("level")
        if (
            not isinstance(manifest_level, int)
            or isinstance(manifest_level, bool)
            or manifest_level != path_level
        ):
            result["integrity_errors"].append(
                f"manifest/path level mismatch: {manifest_path}"
            )
        elif manifest_level in result["manifest_levels"]:
            result["integrity_errors"].append(
                f"duplicate manifest level {manifest_level}: {manifest_path}"
            )
        else:
            result["manifest_levels"].append(manifest_level)
        expected_game = artifact.name.removesuffix("_legs")
        if manifest.get("game") != expected_game:
            result["integrity_errors"].append(
                f"manifest game mismatch: {manifest_path}"
            )
        # The oldest schema-v1 manifests predate the explicit ``schema`` key.
        # Absence therefore means v1; every other unknown value fails closed.
        schema = manifest.get("schema", 1)
        transcript_entries = []
        if schema == 1:
            transcript_entries.append({
                "path": manifest.get("transcript", "proposer_last.log"),
                "sha256": manifest.get("transcript_sha256"),
            })
            codex_transcripts = manifest.get("codex_transcripts", [])
            if not isinstance(codex_transcripts, list):
                result["integrity_errors"].append(
                    f"codex_transcripts must be a list: {manifest_path}"
                )
            else:
                transcript_entries.extend(codex_transcripts)
        elif schema == 2:
            transcripts = manifest.get("transcripts")
            if not isinstance(transcripts, list) or not transcripts:
                result["integrity_errors"].append(
                    f"schema-v2 transcripts must be a nonempty list: "
                    f"{manifest_path}"
                )
            else:
                transcript_entries.extend(transcripts)
            audits = manifest.get("audits")
            if not isinstance(audits, dict) or not audits:
                result["integrity_errors"].append(
                    f"schema-v2 audits must be a nonempty object: "
                    f"{manifest_path}"
                )
            else:
                for audit_name, audit_entry in sorted(audits.items()):
                    if not isinstance(audit_entry, dict):
                        result["integrity_errors"].append(
                            f"invalid schema-v2 audit entry {audit_name}: "
                            f"{manifest_path}"
                        )
                        continue
                    audit_path = safe_evidence_path(
                        evidence_dir, audit_entry.get("path")
                    )
                    if audit_path is None:
                        result["integrity_errors"].append(
                            f"invalid schema-v2 audit path {audit_name}: "
                            f"{manifest_path}"
                        )
                    elif audit_path.is_symlink() or not audit_path.is_file():
                        result["integrity_errors"].append(
                            f"missing schema-v2 audit: {audit_path}"
                        )
                    elif sha256_file(audit_path) != audit_entry.get("sha256"):
                        result["integrity_errors"].append(
                            f"schema-v2 audit hash mismatch: {audit_path}"
                        )
        else:
            result["integrity_errors"].append(
                f"unsupported promotion manifest schema: {manifest_path}"
            )

        for item in transcript_entries:
            if not isinstance(item, dict):
                result["integrity_errors"].append(
                    f"invalid transcript entry: {manifest_path}"
                )
                continue
            transcript = safe_evidence_path(
                evidence_dir, item.get("path")
            )
            if transcript is None:
                result["integrity_errors"].append(
                    f"invalid transcript path: {manifest_path}"
                )
                continue
            if transcript.is_symlink() or not transcript.is_file():
                result["integrity_errors"].append(
                    f"missing transcript: {transcript}"
                )
                continue
            if sha256_file(transcript) != item.get("sha256"):
                result["integrity_errors"].append(
                    f"transcript hash mismatch: {transcript}"
                )
                continue
            hits = scan_file(transcript)
            forbidden = [hit for hit in hits if hit not in INFORMATIONAL_HITS]
            if forbidden:
                result["taint_hits"].append({
                    "path": str(transcript), "kinds": forbidden,
                })
            if "harness_introspection" in hits:
                result["informational_harness_introspection"].append(
                    str(transcript)
                )
            if "operational_process_monitoring" in hits:
                result["informational_operational_monitoring"].append(
                    str(transcript)
                )

        promoted_files = manifest.get("promoted_files_sha256", {})
        if not isinstance(promoted_files, dict):
            result["integrity_errors"].append(
                f"promoted_files_sha256 must be an object: {manifest_path}"
            )
            promoted_files = {}
        files_root = evidence_dir / "files"
        for name, expected in promoted_files.items():
            evidence_file = safe_evidence_path(files_root, name)
            if evidence_file is None:
                result["integrity_errors"].append(
                    f"invalid promoted evidence path: {manifest_path}"
                )
            elif evidence_file.is_symlink() or not evidence_file.is_file():
                result["integrity_errors"].append(f"missing promoted evidence: {evidence_file}")
            elif sha256_file(evidence_file) != expected:
                result["integrity_errors"].append(f"promoted-file hash mismatch: {evidence_file}")
            else:
                hits = scan_file(evidence_file)
                forbidden = [
                    hit for hit in hits if hit not in INFORMATIONAL_HITS
                ]
                if forbidden:
                    result["taint_hits"].append({
                        "path": str(evidence_file), "kinds": forbidden,
                    })
                if "harness_introspection" in hits:
                    result["informational_harness_introspection"].append(
                        str(evidence_file)
                    )
                if "operational_process_monitoring" in hits:
                    result["informational_operational_monitoring"].append(
                        str(evidence_file)
                    )

        if schema == 2:
            parent = manifest.get("parent_manifest")
            if parent is None:
                parent_rel = None
                parent_hash = None
            elif isinstance(parent, dict):
                parent_rel = parent.get("path")
                parent_hash = parent.get("sha256")
            else:
                result["integrity_errors"].append(
                    f"invalid schema-v2 parent manifest: {manifest_path}"
                )
                parent_rel = None
                parent_hash = None
        else:
            parent_rel = manifest.get("parent_manifest")
            parent_hash = manifest.get("parent_manifest_sha256")
        if previous is None:
            if parent_rel is not None or parent_hash is not None:
                result["integrity_errors"].append(f"unexpected parent on first manifest: {manifest_path}")
        else:
            expected_rel = str(previous.relative_to(artifact))
            if parent_rel != expected_rel or parent_hash != sha256_file(previous):
                result["integrity_errors"].append(f"broken parent chain: {manifest_path}")
        previous = manifest_path
    if expected_reached is not None:
        expected_levels = set(range(1, expected_reached + 1))
        actual_levels = set(result["manifest_levels"])
        result["missing_levels"] = sorted(expected_levels - actual_levels)
        result["unexpected_levels"] = sorted(actual_levels - expected_levels)
        result["complete"] = (
            not result["missing_levels"]
            and not result["unexpected_levels"]
            and len(result["manifest_levels"]) == expected_reached
        )
    result["verdict"] = (
        "clean" if not result["integrity_errors"] and not result["taint_hits"]
        else "tainted_or_invalid"
    )
    return result


def wip_category(metadata: dict) -> str:
    try:
        reached = int(metadata.get("reached", -1))
        level = int(metadata.get("level", 10**9))
    except (TypeError, ValueError):
        return "discarded_wip"
    phase = str(metadata.get("phase", ""))
    success_phase = phase in {
        "reached_before_debrief", "after_debrief", "after_auto_solve_debrief",
        "recovered_existing_path_artifact", "recovered_after_credit_out",
        "recovered_path_artifact", "debrief_credit_out",
    }
    return "successful_candidate_wip" if success_phase and reached >= level else "discarded_wip"


def audit(root: Path) -> dict:
    report = {
        "canonical": {"files": 0, "hits": []},
        "successful_candidate_wip": {"snapshots": 0, "files": 0, "hits": []},
        "discarded_wip": {"snapshots": 0, "files": 0, "hits": []},
        "frontier_scaffolds": {"files": 0, "hits": [], "verdict": "clean"},
        "proposer_containment": {},
        "promotion_chains": {},
        "ancestry_note": (
            "WIP adjacency is not promotion ancestry. Historical ancestry needs an "
            "explicit manifest or an independently reconstructed file-hash chain."
        ),
    }
    for artifact in sorted(root.glob("*_legs")):
        if (artifact / "checkpoint.json").is_file():
            report["promotion_chains"][artifact.name] = audit_promotion_chain(artifact)
            for name in sorted(PROMOTED_FILES):
                path = artifact / name
                if not path.is_file():
                    continue
                report["canonical"]["files"] += 1
                hits = scan_file(path)
                if hits:
                    report["canonical"]["hits"].append({
                        "path": str(path), "kinds": hits,
                    })

        # Cold-start artifacts have no canonical checkpoint yet, but their WIP
        # and reviewed scaffold will enter a future clean room. Audit them too.
        for path in sorted(
            (artifact / "wip_context").glob("level_*/frontier_scaffold.json")
        ):
            report["frontier_scaffolds"]["files"] += 1
            hits = scan_file(path)
            if hits:
                report["frontier_scaffolds"]["hits"].append({
                    "path": str(path), "kinds": hits,
                })

        for metadata_path in sorted((artifact / "wip_context").glob("level_*/*/metadata.json")):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            category = wip_category(metadata)
            report[category]["snapshots"] += 1
            # A forensic snapshot contains copies of templates and probes from
            # older attempts.  Those are context, not proof of execution.  The
            # proposer transcript is the action ledger, so WIP attempt verdicts
            # are based on that file alone.  Canonical source is scanned above.
            path = metadata_path.parent / "files" / "proposer_last.log"
            snapshot_hits = []
            if path.is_file():
                report[category]["files"] += 1
                hits = scan_file(path)
                if hits:
                    snapshot_hits.append({"path": str(path), "kinds": hits})
            if snapshot_hits:
                report[category]["hits"].append({
                    "attempt": metadata.get("attempt"),
                    "game": metadata.get("game"),
                    "level": metadata.get("level"),
                    "phase": metadata.get("phase"),
                    "files": snapshot_hits,
                })
    report["canonical"]["verdict"] = "clean" if not report["canonical"]["hits"] else "tainted"
    report["frontier_scaffolds"]["verdict"] = (
        "clean" if not report["frontier_scaffolds"]["hits"] else "tainted"
    )
    report["proposer_containment"] = audit_transcript_containment(root)
    return report


def automated_failed(
    report: dict, *, require_complete_lineage: bool = False
) -> bool:
    failed = report["canonical"]["verdict"] != "clean"
    failed = failed or report["frontier_scaffolds"]["verdict"] != "clean"
    failed = failed or report["proposer_containment"]["verdict"] != "clean"
    for chain in report["promotion_chains"].values():
        failed = failed or chain["verdict"] != "clean"
        if require_complete_lineage:
            failed = failed or chain["complete"] is not True
    return failed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", nargs="?",
        default=str(Path(__file__).parent / "crack_lab" / "agent_solutions"),
    )
    parser.add_argument(
        "--require-complete-lineage", action="store_true",
        help=(
            "fail unless every canonical checkpoint level 1..reached has "
            "exactly one sequential promotion manifest"
        ),
    )
    parser.add_argument(
        "--json", type=Path,
        help="also write the complete machine-readable audit to this path",
    )
    args = parser.parse_args()
    report = audit(Path(args.root))
    failed = automated_failed(
        report,
        require_complete_lineage=args.require_complete_lineage,
    )
    report["automated_verdict"] = "FAIL" if failed else "PASS"
    rendered = json.dumps(report, indent=2) + "\n"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
