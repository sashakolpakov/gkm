#!/usr/bin/env python3
"""Fail-closed gate for the final GKM Community Leaderboard v3 payload.

The gate does not edit GitHub, open scorecards, or mutate release artifacts. It
joins four independently produced objects after the 183/183 freeze:

* the immutable v2 leaderboard YAML and proposed v3 YAML/README;
* the complete schema-v2 release receipt;
* the ONLINE and Competition run receipts emitted by ``replay_scorecard.py``;
* the two public, closed scorecard records served by ARC Prize.

Only a single v3 version may be appended. Historical v1/v2 entries must remain
value-identical, ARC-AGI-3 may not carry a self-reported numeric score, and both
remote runs must bind the same 25-game/183-level release and public Git
revision. The CLI reopens public scorecards through the unauthenticated ARC
Prize endpoint and emits an operator summary; it never writes a payload.

An ambiguous ``close_scorecard`` exception remains FAIL in the run receipt. It
can be discharged only by this command's later public-endpoint verification of
the same card ID, exact 25-game endpoints, Competition flag, totals, score, and
publication timestamp. The recovery/final gate command is::

    python3 arc/crack_lab/arc_agi3_leaderboard_v3_gate.py \
      --baseline-yaml <pr37-v2-submission.yaml> \
      --candidate-yaml <leaderboard-checkout>/submissions/gkm/submission.yaml \
      --candidate-readme <leaderboard-checkout>/submissions/gkm/README.md \
      --release-receipt <v3-release-receipt.json> \
      --canonical-release-root <v3-release-artifacts> \
      --online-run-receipt <online-run.json> \
      --competition-run-receipt <competition-run.json> \
      --online-journal-snapshot <sha256>.jsonl \
      --competition-journal-snapshot <sha256>.jsonl
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import stat
import sys
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import yaml

from verify_frozen_release import (
    FrozenReleaseError,
    load_receipt,
    verify_frozen_release,
)


EXPECTED_GAMES = 25
EXPECTED_LEVELS = 183
AUDITED_TOOLKIT_VERSION = "0.9.9"
PUBLICATION_CLOCK_SKEW = dt.timedelta(minutes=5)
MAX_RECORD_BYTES = 32 * 1024 * 1024
MAX_PUBLIC_ARTIFACT_TEXT_BYTES = 4 * 1024 * 1024
MAX_PUBLIC_ARTIFACT_NODES = 100_000
PUBLIC_SCORECARD_ENDPOINT = "https://arcprize.org/api/v3/scorecards/{card_id}"
PUBLIC_COMMIT_ENDPOINT = (
    "https://api.github.com/repos/sashakolpakov/gkm/commits/{revision}"
)
SCORECARD_URL_PREFIX = "https://arcprize.org/scorecards/"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
OPERATION_ID_RE = re.compile(r"^[0-9a-f]{64}$")
ARC_SCORECARD_URL_RE = re.compile(
    r"https?://(?:www\.)?arcprize\.org/scorecards/[^\s<>\])]+"
)
OFFICIAL_SCORE_RE = re.compile(
    r"\bofficial\s+score\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)%?",
    re.IGNORECASE,
)
ACTION_FACT_RE = re.compile(
    r"\b(?:(frozen\s+stored|official\s+competition)\s+)?actions?"
    r"\s*[:=]?\s*([0-9]+)\b",
    re.IGNORECASE,
)
RESET_FACT_RE = re.compile(
    r"\b(?:(official\s+competition)\s+)?resets?\s*[:=]?\s*([0-9]+)\b",
    re.IGNORECASE,
)
TOOLKIT_FACT_RE = re.compile(
    r"\b(?:ARC(?:-AGI)?\s+)?toolkit(?:\s+version)?\s*[:=]?\s*"
    r"([0-9]+(?:\.[0-9]+)+)\b",
    re.IGNORECASE,
)
COMPLETE_GAMES_RE = re.compile(
    r"\bcomplete\s+games\s*[:=]?\s*([0-9]+)\b", re.IGNORECASE
)
RAW_COVERAGE_RE = re.compile(
    r"\braw\s+coverage\s*[:=]?\s*([0-9]+)\s*/\s*([0-9]+)"
    r"(?:\s*\(\s*([0-9]+(?:\.[0-9]+)?)\s*%\s*\))?",
    re.IGNORECASE,
)
CLOSE_RECOVERY_RE = re.compile(
    r"\bclose\s+recovery\s*[:=]\s*([^\n\r]+)", re.IGNORECASE
)
REVERSE_ACCOUNTING_FACT_RE = re.compile(
    r"\b[0-9]+[ \t]+(?:(?:frozen[ \t]+stored|official[ \t]+competition)"
    r"[ \t]+)?(?:actions?|resets?)\b",
    re.IGNORECASE,
)
ABSOLUTE_POSIX_PATH_RE = re.compile(
    r"(?:^|[\s\"'`=(:,;])/(?!/)(?:[^\s\"'`<>\x00]+)"
)
ABSOLUTE_WINDOWS_PATH_RE = re.compile(
    r"(?:^|[\s\"'`=(:,;])(?:[A-Za-z]:\\|\\\\)"
    r"[^\s\"'`<>\x00]*"
)
SECRET_ASSIGNMENT_RE = re.compile(
    r"\b(?:[A-Za-z][A-Za-z0-9_-]*[-_])?"
    r"(?:api[-_ ]?key|secret|token|password)\b[ \t]*[:=][ \t]*\S+",
    re.IGNORECASE,
)
ANY_COVERAGE_RE = re.compile(r"\b([0-9]+)\s*/\s*([0-9]+)\b")
AMBIGUOUS_CLOSE_ERROR_TYPES = {
    "ChunkedEncodingError",
    "ConnectionError",
    "ConnectTimeout",
    "HTTPError",
    "JSONDecodeError",
    "ProtocolError",
    "ReadTimeout",
    "RemoteDisconnected",
    "Timeout",
    "TimeoutError",
    "ValidationError",
    "ValueError",
}
REQUIRED_AUTHOR_NAMES = ("Alexander Kolpakov", "OpenAI GPT-5.6")
REQUIRED_V3_MODELS = {
    "OpenAI GPT-5.6-sol (expanded campaign)",
    "Claude Code (preserved legacy proposer lineages)",
}
V2_BASELINE_HEAD = "4d6dbaa9d1555c4093ead66d7f4ed6cc35a2b6e4"
V2_BASELINE_YAML_SHA256 = (
    "ff94774bf964c7ebaccf495f72b7d01a22f1020c4332fcb488de42d1755c2266"
)
PROVENANCE_COLUMNS = (
    "component",
    "origin/authoring agent",
    "admitted inputs",
    "transcript or source boundary",
    "verifier receipt",
    "promoted artifact",
)
GKM = Path(__file__).resolve().parents[2]
COMPLETE_VERIFICATION_FIELDS = {
    "status",
    "games",
    "levels",
    "inventory_sha256",
    "canonical_tree_sha256",
    "evidence_sha256",
    "verifier_sha256",
    "control_contract_sha256",
    "receipt_sha256",
    "verification_context_source_revision",
}
JOURNAL_BINDING_FIELDS = {
    "schema",
    "journal_id",
    "live_journal",
    "snapshot",
    "snapshot_sha256",
    "operation_id",
    "opaque_sha256",
    "intent_sequence",
    "intent_event_sha256",
    "opened_sequence",
    "opened_event_sha256",
    "terminal_sequence",
    "terminal_event_sha256",
    "terminal_outcome",
    "receipt_core_sha256",
}
JOURNAL_EVENT_FIELDS = {
    "schema",
    "journal_id",
    "sequence",
    "previous_event_sha256",
    "event",
    "operation_id",
    "timestamp_utc",
    "payload",
}
JOURNAL_INTENT_FIELDS = {
    "mode",
    "source_url",
    "source_revision",
    "arc_agi_toolkit_version",
    "release_receipt_sha256",
    "canonical_tree_sha256",
    "checkpoint_sha256_digest",
    "command_sha256",
    "output_receipt",
    "tags",
    "opaque",
}
JOURNAL_OPENED_FIELDS = {"card_id", "scorecard_url"}
JOURNAL_TERMINAL_FIELDS = {
    "outcome",
    "card_id",
    "receipt_core_sha256",
}
RUN_RECEIPT_FIELDS = {
    "schema",
    "mode",
    "status",
    "scorecard_id",
    "scorecard_url",
    "scorecard_open",
    "scorecard_close",
    "scorecard_tags",
    "scorecard_opaque",
    "source_url",
    "source_revision",
    "arc_agi_toolkit_version",
    "started_at_utc",
    "closed_at_utc",
    "scorecard_close_started_at_utc",
    "scorecard_close_finished_at_utc",
    "command",
    "artifact_root",
    "release_receipt",
    "release_verification",
    "release_binding",
    "checkpoint_sha256",
    "claimed_levels",
    "authoritative_levels",
    "stored_actions",
    "results",
    "aggregate",
    "run_journal",
}
RUN_COMMAND_FIELDS = {
    "entrypoint",
    "mode",
    "games",
    "artifact_root",
    "release_receipt",
    "expected_claimed_levels",
    "preflight_only",
    "source_url",
    "source_revision",
    "tags",
}
RELEASE_BINDING_FIELDS = {
    "binding_scope",
    "receipt_sha256",
    "canonical_tree_sha256",
    "release_identity_source_revision",
    "claimed_inventory",
    "claimed_level_count",
    "authoritative_level_count",
}
RESULT_FIELDS = {"remote", "claimed"}
AGGREGATE_FIELDS = {
    "source_url",
    "tags",
    "opaque",
    "card_id",
    "score",
    "environments",
    "tags_scores",
    "competition_mode",
    "total_environments_completed",
    "total_environments",
    "total_levels_completed",
    "total_levels",
    "total_actions",
}
ENVIRONMENT_SCORE_LIST_FIELDS = {
    "id",
    "runs",
    "score",
    "actions",
    "levels_completed",
    "completed",
    "level_count",
    "resets",
}
ENVIRONMENT_SCORE_FIELDS = {
    "id",
    "guid",
    "score",
    "levels_completed",
    "actions",
    "resets",
    "state",
    "completed",
    "level_scores",
    "level_actions",
    "level_baseline_actions",
    "number_of_levels",
    "number_of_environments",
    "message",
}
PUBLIC_ENVIRONMENT_SCORE_FIELDS = ENVIRONMENT_SCORE_FIELDS - {"message"}
PUBLIC_TAG_SCORE_FIELDS = ENVIRONMENT_SCORE_FIELDS - {
    "level_scores",
    "level_actions",
    "level_baseline_actions",
    "message",
}
EXPECTED_TAG_SCORE_IDS = {"click", "keyboard", "keyboard_click"}
SCORECARD_STATES = {"NOT_PLAYED", "NOT_FINISHED", "WIN", "GAME_OVER"}
PUBLIC_SCORECARD_FIELDS = {
    "source_url",
    "tags",
    "opaque",
    "card_id",
    "score",
    "published_at",
    "ai_agent",
    "environments",
    "tags_scores",
    "open_at",
    "last_update",
    "total_environments_completed",
    "total_environments",
    "total_levels_completed",
    "total_levels",
    "total_actions",
}
V3_FIELDS = {"version", "date", "changes", "models", "scores"}
MODEL_FIELDS = {"name"}
SCORE_FIELDS = {"benchmark", "scorecard_url", "set"}


class LeaderboardV3Error(RuntimeError):
    """The final leaderboard payload is incomplete or internally stale."""


class StrictSafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_strict_mapping(
    loader: StrictSafeLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_strict_mapping,
)


def _reject_nonfinite(value: object, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise LeaderboardV3Error(f"{label} contains a non-finite number")
    if isinstance(value, dict):
        for key, nested in value.items():
            _reject_nonfinite(key, label=label)
            _reject_nonfinite(nested, label=label)
    elif isinstance(value, list):
        for nested in value:
            _reject_nonfinite(nested, label=label)


def _validate_public_text(value: str, *, label: str) -> int:
    """Reject host-local paths, NULs, and credential assignments in output."""
    encoded_size = len(value.encode("utf-8"))
    if encoded_size > MAX_PUBLIC_ARTIFACT_TEXT_BYTES:
        raise LeaderboardV3Error(f"{label} text is unexpectedly large")
    if (
        "\x00" in value
        or ABSOLUTE_POSIX_PATH_RE.search(value) is not None
        or ABSOLUTE_WINDOWS_PATH_RE.search(value) is not None
        or SECRET_ASSIGNMENT_RE.search(value) is not None
    ):
        raise LeaderboardV3Error(
            f"{label} contains a host path, NUL, or secret assignment"
        )
    return encoded_size


def _validate_public_artifact_strings(
    value: object, *, label: str
) -> None:
    """Bounded recursive scan of all mapping keys and free-form strings."""
    stack: list[tuple[object, int]] = [(value, 0)]
    nodes = 0
    total_text_bytes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > MAX_PUBLIC_ARTIFACT_NODES or depth > 64:
            raise LeaderboardV3Error(f"{label} is too deeply nested or large")
        if isinstance(current, str):
            total_text_bytes += _validate_public_text(current, label=label)
            if total_text_bytes > MAX_PUBLIC_ARTIFACT_TEXT_BYTES:
                raise LeaderboardV3Error(f"{label} text is unexpectedly large")
        elif isinstance(current, Mapping):
            for key, nested in current.items():
                stack.append((key, depth + 1))
                stack.append((nested, depth + 1))
        elif isinstance(current, (list, tuple)):
            stack.extend((nested, depth + 1) for nested in current)


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, nested in pairs:
        if key in value:
            raise LeaderboardV3Error(f"JSON contains duplicate key {key!r}")
        value[key] = nested
    return value


def _reject_json_constant(value: str) -> None:
    raise LeaderboardV3Error(f"JSON contains non-finite number {value}")


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LeaderboardV3Error(
            "release receipt contains noncanonical JSON"
        ) from exc


def _json_sha256(value: object) -> str:
    return _sha256_bytes(_canonical_json(value))


def _read_regular(path: Path, *, label: str) -> bytes:
    target = Path(path)
    try:
        metadata = target.lstat()
    except OSError as exc:
        raise LeaderboardV3Error(f"cannot stat {label}: {target}") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size > MAX_RECORD_BYTES
    ):
        raise LeaderboardV3Error(
            f"{label} must be a bounded, single-link regular file"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags)
    except OSError as exc:
        raise LeaderboardV3Error(f"cannot securely open {label}") from exc
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size != metadata.st_size
            or (opened.st_dev, opened.st_ino)
            != (metadata.st_dev, metadata.st_ino)
            or opened.st_mtime_ns != metadata.st_mtime_ns
            or opened.st_ctime_ns != metadata.st_ctime_ns
        ):
            raise LeaderboardV3Error(f"{label} changed during bounded read")
        raw = handle.read(MAX_RECORD_BYTES + 1)
        after = os.fstat(handle.fileno())
        if (
            (after.st_dev, after.st_ino, after.st_size)
            != (opened.st_dev, opened.st_ino, opened.st_size)
            or after.st_mtime_ns != opened.st_mtime_ns
            or after.st_ctime_ns != opened.st_ctime_ns
        ):
            raise LeaderboardV3Error(f"{label} changed during bounded read")
    if len(raw) > MAX_RECORD_BYTES:
        raise LeaderboardV3Error(f"{label} is unexpectedly large")
    return raw


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    raw = _read_regular(path, label=label)
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise LeaderboardV3Error(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise LeaderboardV3Error(f"{label} must be a JSON object")
    _reject_nonfinite(value, label=label)
    return value


def _load_yaml(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular(path, label=label)
    if expected_sha256 is not None and _sha256_bytes(raw) != expected_sha256:
        raise LeaderboardV3Error(
            f"{label} is not the frozen PR #37 v2 baseline at "
            f"{V2_BASELINE_HEAD}"
        )
    try:
        value = yaml.load(raw, Loader=StrictSafeLoader)
    except (UnicodeError, yaml.YAMLError) as exc:
        raise LeaderboardV3Error(f"{label} is invalid YAML") from exc
    if not isinstance(value, dict):
        raise LeaderboardV3Error(f"{label} must be a YAML mapping")
    _reject_nonfinite(value, label=label)
    return value, raw


def _parse_utc(value: object, *, label: str) -> dt.datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise LeaderboardV3Error(f"{label} must be a UTC timestamp ending in Z")
    try:
        parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise LeaderboardV3Error(f"{label} is not an ISO-8601 timestamp") from exc
    if parsed.utcoffset() != dt.timedelta(0):
        raise LeaderboardV3Error(f"{label} is not UTC")
    return parsed


def _scorecard_id_from_url(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.startswith(SCORECARD_URL_PREFIX):
        raise LeaderboardV3Error(f"{label} is not an ARC Prize scorecard URL")
    card_id = value.removeprefix(SCORECARD_URL_PREFIX)
    if UUID_RE.fullmatch(card_id) is None:
        raise LeaderboardV3Error(f"{label} has an invalid scorecard ID")
    return card_id


def _immutable_gkm_url(value: object, revision: str) -> bool:
    return (
        isinstance(value, str)
        and REVISION_RE.fullmatch(revision) is not None
        and value == f"https://github.com/sashakolpakov/gkm/tree/{revision}"
    )


def verify_public_revision(revision: str) -> dict[str, str]:
    """Reopen the exact immutable public GKM commit through GitHub's API."""
    if REVISION_RE.fullmatch(revision) is None:
        raise LeaderboardV3Error("public source revision is invalid")
    request = urllib.request.Request(
        PUBLIC_COMMIT_ENDPOINT.format(revision=revision),
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "gkm-v3-gate/1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            raw = response.read(1024 * 1024 + 1)
    except (OSError, urllib.error.URLError) as exc:
        raise LeaderboardV3Error(
            "scored source revision is not publicly reachable on GitHub"
        ) from exc
    if len(raw) > 1024 * 1024:
        raise LeaderboardV3Error("GitHub commit response is unexpectedly large")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise LeaderboardV3Error("GitHub commit response is invalid JSON") from exc
    expected_html_url = (
        f"https://github.com/sashakolpakov/gkm/commit/{revision}"
    )
    if (
        not isinstance(value, dict)
        or value.get("sha") != revision
        or value.get("html_url") != expected_html_url
    ):
        raise LeaderboardV3Error("GitHub resolved a different source revision")
    return {"sha": revision, "html_url": expected_html_url}


def _logical_path_is_safe(value: object) -> bool:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\x00" in value
        or re.match(r"^[A-Za-z]:", value) is not None
        or re.fullmatch(r"(?:<external>/)?[A-Za-z0-9._/-]+", value) is None
    ):
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and "." not in path.parts
        and "//" not in value
    )


def _inventory(value: object, *, label: str) -> dict[str, int]:
    if not isinstance(value, dict) or len(value) != EXPECTED_GAMES:
        raise LeaderboardV3Error(f"{label} must contain exactly 25 games")
    result: dict[str, int] = {}
    for game, depth in value.items():
        if (
            not isinstance(game, str)
            or re.fullmatch(r"[a-z0-9]{4}", game) is None
            or not _is_int(depth)
            or depth <= 0
        ):
            raise LeaderboardV3Error(f"{label} contains an invalid frontier")
        result[game] = depth
    if sum(result.values()) != EXPECTED_LEVELS:
        raise LeaderboardV3Error(f"{label} is not a complete 183-level frontier")
    return result


def _release_stored_actions(
    release: Mapping[str, Any], inventory: Mapping[str, int]
) -> int:
    evidence = release.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != set(inventory):
        raise LeaderboardV3Error("complete release has no checkpoint evidence")
    total = 0
    for game, target in inventory.items():
        rows = evidence.get(game)
        endpoint = rows[-1] if isinstance(rows, list) and len(rows) == target else None
        action_count = (
            endpoint.get("action_count") if isinstance(endpoint, dict) else None
        )
        if not _is_int(action_count) or action_count <= 0:
            raise LeaderboardV3Error(
                f"complete release endpoint action evidence is invalid: {game}"
            )
        total += action_count
    return total


def validate_complete_release_receipt(
    path: Path,
    *,
    canonical_root: Path,
    verifier_root: Path | None = None,
    repo_root: Path = GKM,
) -> tuple[dict[str, Any], str, dict[str, int]]:
    """Independently replay the revision-bound verifier over the frozen tree."""
    try:
        body, _ = load_receipt(Path(path))
    except FrozenReleaseError as exc:
        raise LeaderboardV3Error(f"release receipt rejected: {exc}") from exc
    digest = Path(path).stem
    inventory = _inventory(body.get("inventory"), label="release inventory")
    if (
        not _is_int(body.get("schema"))
        or body.get("schema") != 1
        or not _is_int(body.get("canonical_game_count"))
        or body.get("canonical_game_count") != EXPECTED_GAMES
        or not _is_int(body.get("authoritative_level_count"))
        or body.get("authoritative_level_count") != EXPECTED_LEVELS
        or body.get("complete") is False
        or "claimed_level_count" in body
        or "unclaimed_boundaries" in body
        or "claimed_inventory" in body
    ):
        raise LeaderboardV3Error(
            "release receipt is partial or is not the complete schema-v2 freeze"
        )
    tree_hash = body.get("canonical_tree_sha256")
    if not isinstance(tree_hash, str) or SHA256_RE.fullmatch(tree_hash) is None:
        raise LeaderboardV3Error("release receipt has no canonical tree hash")
    evidence = body.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != set(inventory):
        raise LeaderboardV3Error("release receipt evidence game set is incomplete")
    for game, target in inventory.items():
        rows = evidence.get(game)
        if not isinstance(rows, list) or len(rows) != target:
            raise LeaderboardV3Error(
                f"release receipt evidence is incomplete for {game}"
            )
    _release_stored_actions(body, inventory)
    if (
        body.get("release_identity_sha256")
        != _json_sha256(body.get("release_identity"))
        or body.get("inventory_sha256") != _json_sha256(inventory)
        or body.get("evidence_sha256") != _json_sha256(evidence)
    ):
        raise LeaderboardV3Error("release receipt has an inconsistent bound hash")
    try:
        verified = verify_frozen_release(
            receipt_path=Path(path).resolve(),
            canonical_root=Path(canonical_root).resolve(),
            repo_root=Path(repo_root).resolve(),
            verifier_root=(
                Path(verifier_root).resolve()
                if verifier_root is not None
                else None
            ),
        )
    except (FrozenReleaseError, OSError, ValueError) as exc:
        raise LeaderboardV3Error(
            f"release receipt failed independent historical verification: {exc}"
        ) from exc
    release_identity = body.get("release_identity")
    revision = (
        release_identity.get("source_revision")
        if isinstance(release_identity, dict)
        else None
    )
    expected_verified = {
        "status": "PASS",
        "games": EXPECTED_GAMES,
        "levels": EXPECTED_LEVELS,
        "inventory_sha256": body.get("inventory_sha256"),
        "canonical_tree_sha256": body.get("canonical_tree_sha256"),
        "evidence_sha256": body.get("evidence_sha256"),
        "verifier_sha256": (
            body.get("verifier", {}).get("sha256")
            if isinstance(body.get("verifier"), dict)
            else None
        ),
        "control_contract_sha256": (
            body.get("control_contract", {}).get("sha256")
            if isinstance(body.get("control_contract"), dict)
            else None
        ),
        "receipt_sha256": digest,
        "verification_context_source_revision": revision,
    }
    if any(verified.get(key) != value for key, value in expected_verified.items()):
        raise LeaderboardV3Error(
            "independent historical verification summary differs from the "
            "complete receipt"
        )
    return body, digest, inventory


def expected_scorecard_tags(
    *, mode: str, receipt_sha256: str, revision: str
) -> list[str]:
    return [
        "gkm-v3",
        "replay-validated",
        "agent",
        f"mode:{mode}",
        f"release:{receipt_sha256}",
        f"revision:{revision}",
    ]


def expected_scorecard_opaque(
    *,
    operation_id: str,
    mode: str,
    receipt_sha256: str,
    canonical_tree_sha256: str,
    revision: str,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "gkm_operation_id": operation_id,
        "mode": mode,
        "release_receipt_sha256": receipt_sha256,
        "canonical_tree_sha256": canonical_tree_sha256,
        "source_revision": revision,
    }


def expected_journal_id(receipt_sha256: str) -> str:
    return hashlib.sha256(
        f"gkm-complete-remote-journal-v1:{receipt_sha256}".encode("ascii")
    ).hexdigest()


def _parse_journal_snapshot(raw: bytes) -> list[dict[str, Any]]:
    if len(raw) > MAX_RECORD_BYTES:
        raise LeaderboardV3Error("run-journal snapshot is unexpectedly large")
    if not raw or not raw.endswith(b"\n"):
        raise LeaderboardV3Error("run-journal snapshot is empty or truncated")
    records: list[dict[str, Any]] = []
    states: dict[str, str] = {}
    card_ids: set[str] = set()
    previous_digest: str | None = None
    journal_id: str | None = None
    previous_timestamp: dt.datetime | None = None
    for sequence, line in enumerate(raw.splitlines(), start=1):
        try:
            event = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise LeaderboardV3Error(
                "run-journal snapshot contains invalid JSON"
            ) from exc
        if not isinstance(event, dict) or set(event) != JOURNAL_EVENT_FIELDS:
            raise LeaderboardV3Error("run-journal event schema mismatch")
        if _canonical_json(event) != line:
            raise LeaderboardV3Error("run-journal event is not canonical JSON")
        if (
            not _is_int(event.get("schema"))
            or event.get("schema") != 1
            or not _is_int(event.get("sequence"))
            or event.get("sequence") != sequence
        ):
            raise LeaderboardV3Error("run-journal event sequence mismatch")
        if event.get("previous_event_sha256") != previous_digest:
            raise LeaderboardV3Error("run-journal hash chain mismatch")
        current_journal_id = event.get("journal_id")
        if (
            not isinstance(current_journal_id, str)
            or SHA256_RE.fullmatch(current_journal_id) is None
            or (journal_id is not None and current_journal_id != journal_id)
        ):
            raise LeaderboardV3Error("run-journal identity mismatch")
        journal_id = current_journal_id
        timestamp = _parse_utc(
            event.get("timestamp_utc"), label="run-journal event"
        )
        if previous_timestamp is not None and timestamp < previous_timestamp:
            raise LeaderboardV3Error(
                "run-journal event timestamps are not monotonic"
            )
        previous_timestamp = timestamp
        operation_id = event.get("operation_id")
        if (
            not isinstance(operation_id, str)
            or OPERATION_ID_RE.fullmatch(operation_id) is None
        ):
            raise LeaderboardV3Error("run-journal operation ID is invalid")
        kind = event.get("event")
        payload = event.get("payload")
        if not isinstance(payload, dict):
            raise LeaderboardV3Error("run-journal payload is invalid")
        state = states.get(operation_id)
        if kind == "INTENT":
            if state is not None or set(payload) != JOURNAL_INTENT_FIELDS:
                raise LeaderboardV3Error(
                    "run-journal has a duplicate or malformed intent"
                )
            if payload.get("mode") not in {"online", "competition"}:
                raise LeaderboardV3Error("run-journal intent mode is invalid")
            opaque = payload.get("opaque")
            if (
                not isinstance(opaque, dict)
                or set(opaque) != {
                    "schema",
                    "gkm_operation_id",
                    "mode",
                    "release_receipt_sha256",
                    "canonical_tree_sha256",
                    "source_revision",
                }
                or not _is_int(opaque.get("schema"))
                or opaque.get("schema") != 1
            ):
                raise LeaderboardV3Error(
                    "run-journal intent opaque schema is invalid"
                )
            states[operation_id] = "INTENT"
        elif kind == "OPENED":
            card_id = payload.get("card_id")
            if (
                state != "INTENT"
                or set(payload) != JOURNAL_OPENED_FIELDS
                or not isinstance(card_id, str)
                or UUID_RE.fullmatch(card_id) is None
                or card_id in card_ids
                or payload.get("scorecard_url")
                != SCORECARD_URL_PREFIX + card_id
            ):
                raise LeaderboardV3Error(
                    "run-journal OPENED transition is invalid"
                )
            card_ids.add(card_id)
            states[operation_id] = "OPENED"
        elif kind == "TERMINAL":
            outcome = payload.get("outcome")
            card_id = payload.get("card_id")
            if (
                set(payload) != JOURNAL_TERMINAL_FIELDS
                or outcome
                not in {
                    "CLOSED_CONFIRMED_PASS",
                    "CLOSED_CONFIRMED_FAIL",
                    "CLOSE_OUTCOME_AMBIGUOUS",
                    "OPEN_OUTCOME_AMBIGUOUS",
                }
                or not isinstance(payload.get("receipt_core_sha256"), str)
                or SHA256_RE.fullmatch(payload["receipt_core_sha256"]) is None
            ):
                raise LeaderboardV3Error(
                    "run-journal TERMINAL event is malformed"
                )
            if outcome == "OPEN_OUTCOME_AMBIGUOUS":
                if state != "INTENT" or card_id is not None:
                    raise LeaderboardV3Error(
                        "run-journal open ambiguity transition is invalid"
                    )
            elif (
                state != "OPENED"
                or not isinstance(card_id, str)
                or UUID_RE.fullmatch(card_id) is None
            ):
                raise LeaderboardV3Error(
                    "run-journal terminal card transition is invalid"
                )
            states[operation_id] = str(outcome)
        else:
            raise LeaderboardV3Error("run-journal event type is invalid")
        digest = _sha256_bytes(line)
        records.append({"event": event, "sha256": digest})
        previous_digest = digest
    return records


def _journal_states(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Mapping[str, Any]]]:
    states: dict[str, dict[str, Mapping[str, Any]]] = {}
    for record in records:
        event = record["event"]
        state = states.setdefault(event["operation_id"], {})
        state[event["event"].lower()] = record
    return states


def _validate_journal_snapshot(
    raw: bytes,
    *,
    run: Mapping[str, Any],
    mode: str,
    receipt_sha256: str,
    release: Mapping[str, Any],
) -> list[dict[str, Any]]:
    journal = run.get("run_journal")
    if not isinstance(journal, dict) or set(journal) != JOURNAL_BINDING_FIELDS:
        raise LeaderboardV3Error(f"{mode} run journal binding schema mismatch")
    if not _is_int(journal.get("schema")) or journal.get("schema") != 1:
        raise LeaderboardV3Error(f"{mode} run journal schema mismatch")
    if any(
        not _is_int(journal.get(field))
        for field in (
            "intent_sequence",
            "opened_sequence",
            "terminal_sequence",
        )
    ):
        raise LeaderboardV3Error(f"{mode} run journal sequence schema mismatch")
    snapshot_sha256 = _sha256_bytes(raw)
    if (
        journal.get("snapshot_sha256") != snapshot_sha256
        or journal.get("journal_id") != expected_journal_id(receipt_sha256)
        or journal.get("live_journal")
        != f"arc/crack_lab/run_journals/{receipt_sha256}.jsonl"
        or journal.get("snapshot")
        != f"arc/crack_lab/run_journals/snapshots/{snapshot_sha256}.jsonl"
    ):
        raise LeaderboardV3Error(f"{mode} run journal snapshot hash mismatch")
    records = _parse_journal_snapshot(raw)
    states = _journal_states(records)
    operation_id = journal.get("operation_id")
    state = states.get(str(operation_id))
    if state is None or set(state) != {"intent", "opened", "terminal"}:
        raise LeaderboardV3Error(
            f"{mode} run lacks one exact INTENT/OPENED/TERMINAL chain"
        )
    if any(
        "terminal" not in candidate
        or candidate["terminal"]["event"]["payload"]["outcome"]
        == "OPEN_OUTCOME_AMBIGUOUS"
        for candidate in states.values()
    ):
        raise LeaderboardV3Error(
            f"{mode} run journal contains an unresolved open intent"
        )
    modes = [
        candidate["intent"]["event"]["payload"]["mode"]
        for candidate in states.values()
    ]
    expected_modes = ["online"] if mode == "online" else ["online", "competition"]
    if modes != expected_modes or len(modes) != len(set(modes)):
        raise LeaderboardV3Error(
            f"{mode} run journal contains a hidden or duplicate open attempt"
        )
    intent = state["intent"]
    opened = state["opened"]
    terminal = state["terminal"]
    if (
        journal.get("journal_id") != intent["event"]["journal_id"]
        or journal.get("intent_sequence") != intent["event"]["sequence"]
        or journal.get("intent_event_sha256") != intent["sha256"]
        or journal.get("opened_sequence") != opened["event"]["sequence"]
        or journal.get("opened_event_sha256") != opened["sha256"]
        or journal.get("terminal_sequence") != terminal["event"]["sequence"]
        or journal.get("terminal_event_sha256") != terminal["sha256"]
    ):
        raise LeaderboardV3Error(f"{mode} run journal event binding mismatch")
    receipt_core = dict(run)
    receipt_core.pop("run_journal", None)
    receipt_core_sha256 = _json_sha256(receipt_core)
    terminal_payload = terminal["event"]["payload"]
    expected_terminal = (
        "CLOSED_CONFIRMED_PASS"
        if run.get("status") == "PASS"
        else "CLOSE_OUTCOME_AMBIGUOUS"
    )
    if (
        journal.get("receipt_core_sha256") != receipt_core_sha256
        or terminal_payload.get("receipt_core_sha256") != receipt_core_sha256
        or journal.get("terminal_outcome") != expected_terminal
        or terminal_payload.get("outcome") != expected_terminal
        or terminal_payload.get("card_id") != run.get("scorecard_id")
    ):
        raise LeaderboardV3Error(f"{mode} run journal terminal binding mismatch")
    revision = run.get("source_revision")
    expected_tags = expected_scorecard_tags(
        mode=mode,
        receipt_sha256=receipt_sha256,
        revision=str(revision),
    )
    expected_opaque = expected_scorecard_opaque(
        operation_id=str(operation_id),
        mode=mode,
        receipt_sha256=receipt_sha256,
        canonical_tree_sha256=str(release.get("canonical_tree_sha256")),
        revision=str(revision),
    )
    run_opaque = run.get("scorecard_opaque")
    if (
        not isinstance(run_opaque, dict)
        or not _is_int(run_opaque.get("schema"))
    ):
        raise LeaderboardV3Error(f"{mode} run scorecard opaque schema mismatch")
    intent_payload = intent["event"]["payload"]
    expected_output_receipt = (
        "arc/crack_lab/run_journals/receipts/"
        f"{receipt_sha256}/{mode}.json"
    )
    if intent_payload.get("output_receipt") != expected_output_receipt:
        raise LeaderboardV3Error(
            f"{mode} run journal output receipt path is not canonical"
        )
    if (
        run.get("scorecard_tags") != expected_tags
        or run_opaque != expected_opaque
        or journal.get("opaque_sha256") != _json_sha256(expected_opaque)
        or opened["event"]["payload"].get("card_id")
        != run.get("scorecard_id")
        or opened["event"]["payload"].get("scorecard_url")
        != run.get("scorecard_url")
        or intent_payload
        != {
            "mode": mode,
            "source_url": run.get("source_url"),
            "source_revision": revision,
            "arc_agi_toolkit_version": run.get("arc_agi_toolkit_version"),
            "release_receipt_sha256": receipt_sha256,
            "canonical_tree_sha256": release.get("canonical_tree_sha256"),
            "checkpoint_sha256_digest": _json_sha256(
                run.get("checkpoint_sha256")
            ),
            "command_sha256": _json_sha256(run.get("command")),
            "output_receipt": intent_payload.get("output_receipt"),
            "tags": expected_tags,
            "opaque": expected_opaque,
        }
    ):
        raise LeaderboardV3Error(f"{mode} run journal intent binding mismatch")
    return records


def _number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_complete_environment_score(
    environment: Mapping[str, Any],
    *,
    game: str,
    target: int,
    public: bool,
    label: str,
) -> None:
    if set(environment) != ENVIRONMENT_SCORE_LIST_FIELDS:
        raise LeaderboardV3Error(f"{label} environment schema mismatch: {game}")
    environment_id = environment.get("id")
    if (
        not isinstance(environment_id, str)
        or re.fullmatch(rf"{re.escape(game)}-[0-9a-f]{{8}}", environment_id)
        is None
    ):
        raise LeaderboardV3Error(f"{label} environment ID mismatch: {game}")
    runs = environment.get("runs")
    if not isinstance(runs, list) or not runs:
        raise LeaderboardV3Error(
            f"{label} must contain provider run history: {game}"
        )
    expected_fields = (
        PUBLIC_ENVIRONMENT_SCORE_FIELDS if public else ENVIRONMENT_SCORE_FIELDS
    )
    winners: list[int] = []
    for index, run in enumerate(runs):
        if not isinstance(run, dict) or set(run) != expected_fields:
            raise LeaderboardV3Error(
                f"{label} nested run schema mismatch: {game}"
            )
        if (
            run.get("id") != environment_id
            or not isinstance(run.get("guid"), str)
            or UUID_RE.fullmatch(run["guid"]) is None
            or not _number(run.get("score"))
            or not 0 <= float(run["score"]) <= 100
            or not _is_int(run.get("levels_completed"))
            or not 0 <= run["levels_completed"] <= target
            or not _is_int(run.get("actions"))
            or run["actions"] < 0
            or not _is_int(run.get("resets"))
            or run["resets"] < 0
            or run.get("state") not in SCORECARD_STATES
            or not isinstance(run.get("completed"), bool)
            or run.get("number_of_levels") != target
            or run.get("number_of_environments") != 1
            or (not public and run.get("message") is not None)
        ):
            raise LeaderboardV3Error(
                f"{label} nested run accounting is invalid: {game}"
            )
        for field, numeric in (
            ("level_scores", False),
            ("level_actions", True),
            ("level_baseline_actions", True),
        ):
            values = run.get(field)
            if (
                not isinstance(values, list)
                or len(values) != target
                or any(
                    (not _is_int(value) if numeric else not _number(value))
                    for value in values
                )
                or (
                    numeric
                    and any(value < 0 for value in values)
                )
            ):
                raise LeaderboardV3Error(
                    f"{label} nested run level accounting mismatch: {game}"
                )
        if (
            sum(run["level_actions"]) != run["actions"]
            or any(not 0 <= float(value) <= 115 for value in run["level_scores"])
        ):
            raise LeaderboardV3Error(
                f"{label} nested run level accounting mismatch: {game}"
            )
        won = (
            run.get("levels_completed") == target
            and run.get("state") == "WIN"
            and run.get("completed") is True
        )
        if won:
            winners.append(index)
        elif run.get("state") == "WIN" or run.get("completed") is True:
            raise LeaderboardV3Error(
                f"{label} nested run has inconsistent terminal state: {game}"
            )
    if winners != [len(runs) - 1]:
        raise LeaderboardV3Error(
            f"{label} must have exactly one terminal target WIN run: {game}"
        )
    if (
        environment.get("score") != max(run["score"] for run in runs)
        or environment.get("actions") != sum(run["actions"] for run in runs)
        or environment.get("levels_completed")
        != max(run["levels_completed"] for run in runs)
        or environment.get("completed")
        is not any(run["completed"] for run in runs)
        or environment.get("level_count")
        != max(len(run["level_scores"]) for run in runs)
        or environment.get("resets") != sum(run["resets"] for run in runs)
    ):
        raise LeaderboardV3Error(
            f"{label} environment aggregate mismatch: {game}"
        )


def _validate_tag_scores(
    value: object, *, public: bool, label: str
) -> dict[str, Mapping[str, Any]]:
    expected_fields = (
        PUBLIC_TAG_SCORE_FIELDS if public else ENVIRONMENT_SCORE_FIELDS
    )
    if not isinstance(value, list):
        raise LeaderboardV3Error(f"{label} tag scores are missing")
    by_id: dict[str, Mapping[str, Any]] = {}
    for score in value:
        score_id = score.get("id") if isinstance(score, dict) else None
        if (
            not isinstance(score, dict)
            or set(score) != expected_fields
            or score_id not in EXPECTED_TAG_SCORE_IDS
            or score_id in by_id
            or not isinstance(score.get("guid"), str)
            or UUID_RE.fullmatch(score["guid"]) is None
            or not _number(score.get("score"))
            or not 0 <= float(score["score"]) <= 100
            or not _is_int(score.get("levels_completed"))
            or score["levels_completed"] < 0
            or not _is_int(score.get("actions"))
            or score["actions"] < 0
            or not _is_int(score.get("resets"))
            or score["resets"] < 0
            or score.get("state") not in SCORECARD_STATES
            or not isinstance(score.get("completed"), bool)
            or not _is_int(score.get("number_of_levels"))
            or score["number_of_levels"] < 0
            or not _is_int(score.get("number_of_environments"))
            or score["number_of_environments"] <= 0
            or (
                not public
                and (
                    score.get("level_scores") is not None
                    or score.get("level_actions") is not None
                    or score.get("level_baseline_actions") is not None
                    or score.get("message") is not None
                )
            )
        ):
            raise LeaderboardV3Error(f"{label} tag score schema mismatch")
        by_id[str(score_id)] = score
    if set(by_id) != EXPECTED_TAG_SCORE_IDS:
        raise LeaderboardV3Error(f"{label} tag score set mismatch")
    return by_id


def _public_run_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    return {field: value[field] for field in PUBLIC_ENVIRONMENT_SCORE_FIELDS}


def _public_tag_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    return {field: value[field] for field in PUBLIC_TAG_SCORE_FIELDS}


def _validate_closed_aggregate(
    aggregate: object,
    *,
    run: Mapping[str, Any],
    mode: str,
    inventory: Mapping[str, int],
    allow_absent: bool,
) -> None:
    if allow_absent and aggregate is None:
        return
    if not isinstance(aggregate, dict) or set(aggregate) != AGGREGATE_FIELDS:
        raise LeaderboardV3Error(f"{mode} closed aggregate schema mismatch")
    if (
        aggregate.get("source_url") != run.get("source_url")
        or aggregate.get("tags") != run.get("scorecard_tags")
        or aggregate.get("opaque") != run.get("scorecard_opaque")
        or aggregate.get("card_id") != run.get("scorecard_id")
        or aggregate.get("competition_mode") is not (mode == "competition")
        or not _number(aggregate.get("score"))
    ):
        raise LeaderboardV3Error(f"{mode} closed aggregate provenance mismatch")
    environments = aggregate.get("environments")
    if not isinstance(environments, list) or len(environments) != EXPECTED_GAMES:
        raise LeaderboardV3Error(f"{mode} closed aggregate game set mismatch")
    by_game: dict[str, Mapping[str, Any]] = {}
    for environment in environments:
        if not isinstance(environment, dict):
            raise LeaderboardV3Error(f"{mode} closed aggregate is malformed")
        environment_id = environment.get("id")
        matches = [
            game
            for game in inventory
            if isinstance(environment_id, str)
            and re.fullmatch(
                rf"{re.escape(game)}-[0-9a-f]{{8}}", environment_id
            )
            is not None
        ]
        if len(matches) != 1 or matches[0] in by_game:
            raise LeaderboardV3Error(
                f"{mode} closed aggregate game accounting is ambiguous"
            )
        by_game[matches[0]] = environment
    if set(by_game) != set(inventory):
        raise LeaderboardV3Error(f"{mode} closed aggregate game set mismatch")
    for game, target in inventory.items():
        _validate_complete_environment_score(
            by_game[game],
            game=game,
            target=target,
            public=False,
            label=f"{mode} closed aggregate",
        )
    _validate_tag_scores(
        aggregate.get("tags_scores"),
        public=False,
        label=f"{mode} closed aggregate",
    )
    total_actions = sum(int(row["actions"]) for row in environments)
    if (
        aggregate.get("total_environments_completed") != EXPECTED_GAMES
        or aggregate.get("total_environments") != EXPECTED_GAMES
        or aggregate.get("total_levels_completed") != EXPECTED_LEVELS
        or aggregate.get("total_levels") != EXPECTED_LEVELS
        or aggregate.get("total_actions") != total_actions
    ):
        raise LeaderboardV3Error(f"{mode} closed aggregate totals mismatch")


def _run_binding(
    run: Mapping[str, Any],
    *,
    mode: str,
    receipt_sha256: str,
    release: Mapping[str, Any],
    inventory: Mapping[str, int],
) -> tuple[dt.datetime, dt.datetime, bool]:
    if set(run) != RUN_RECEIPT_FIELDS:
        raise LeaderboardV3Error(f"{mode} run receipt field schema mismatch")
    if (
        not _is_int(run.get("schema"))
        or run.get("schema") != 2
        or run.get("mode") != mode
    ):
        raise LeaderboardV3Error(f"{mode} run receipt schema/mode mismatch")
    close = run.get("scorecard_close")
    if not isinstance(close, dict) or set(close) != {"status", "error_type"}:
        raise LeaderboardV3Error(f"{mode} run has no close outcome")
    remotely_recoverable = (
        run.get("status") == "FAIL"
        and close.get("status") == "ambiguous"
        and close.get("error_type") in AMBIGUOUS_CLOSE_ERROR_TYPES
    )
    if run.get("status") == "PASS":
        if close != {"status": "confirmed", "error_type": None}:
            raise LeaderboardV3Error(
                f"{mode} run claims PASS without a confirmed close"
            )
    elif not remotely_recoverable:
        raise LeaderboardV3Error(f"{mode} run receipt did not pass")
    open_outcome = run.get("scorecard_open")
    if not isinstance(open_outcome, dict) or open_outcome != {
        "status": "confirmed",
        "error_type": None,
    }:
        raise LeaderboardV3Error(
            f"{mode} run has no durably confirmed open outcome"
        )
    card_id = run.get("scorecard_id")
    if not isinstance(card_id, str) or UUID_RE.fullmatch(card_id) is None:
        raise LeaderboardV3Error(f"{mode} run receipt has no scorecard ID")
    if run.get("scorecard_url") != SCORECARD_URL_PREFIX + card_id:
        raise LeaderboardV3Error(f"{mode} run receipt scorecard URL mismatch")
    revision = run.get("source_revision")
    if not isinstance(revision, str) or REVISION_RE.fullmatch(revision) is None:
        raise LeaderboardV3Error(f"{mode} run has no immutable source revision")
    if not _immutable_gkm_url(run.get("source_url"), revision):
        raise LeaderboardV3Error(f"{mode} run source URL is not immutable")
    toolkit = run.get("arc_agi_toolkit_version")
    if toolkit != AUDITED_TOOLKIT_VERSION:
        raise LeaderboardV3Error(
            f"{mode} run did not use audited arc-agi {AUDITED_TOOLKIT_VERSION}"
        )
    started = _parse_utc(run.get("started_at_utc"), label=f"{mode} start")
    closed = _parse_utc(run.get("closed_at_utc"), label=f"{mode} close")
    close_started = _parse_utc(
        run.get("scorecard_close_started_at_utc"),
        label=f"{mode} close attempt start",
    )
    close_finished = _parse_utc(
        run.get("scorecard_close_finished_at_utc"),
        label=f"{mode} close response",
    )
    if not started <= close_started <= close_finished <= closed:
        raise LeaderboardV3Error(f"{mode} run closes before it starts")

    binding = run.get("release_binding")
    verification = run.get("release_verification")
    if not isinstance(binding, dict) or not isinstance(verification, dict):
        raise LeaderboardV3Error(f"{mode} run has no release verification")
    if set(binding) != RELEASE_BINDING_FIELDS:
        raise LeaderboardV3Error(
            f"{mode} run release binding field schema mismatch"
        )
    release_identity = release.get("release_identity")
    release_revision = (
        release_identity.get("source_revision")
        if isinstance(release_identity, dict)
        else None
    )
    if (
        not isinstance(release_revision, str)
        or REVISION_RE.fullmatch(release_revision) is None
        or binding.get("release_identity_source_revision")
        != release_revision
        or revision != release_revision
    ):
        raise LeaderboardV3Error(
            f"{mode} source revision differs from the complete release receipt"
        )
    expected_verification = {
        "status": "PASS",
        "games": EXPECTED_GAMES,
        "levels": EXPECTED_LEVELS,
        "inventory_sha256": release.get("inventory_sha256"),
        "canonical_tree_sha256": release.get("canonical_tree_sha256"),
        "evidence_sha256": release.get("evidence_sha256"),
        "verifier_sha256": (
            release.get("verifier", {}).get("sha256")
            if isinstance(release.get("verifier"), dict)
            else None
        ),
        "control_contract_sha256": (
            release.get("control_contract", {}).get("sha256")
            if isinstance(release.get("control_contract"), dict)
            else None
        ),
        "receipt_sha256": receipt_sha256,
        "verification_context_source_revision": release_revision,
    }
    if set(verification) != COMPLETE_VERIFICATION_FIELDS:
        raise LeaderboardV3Error(
            f"{mode} run release verification schema is not path-safe"
        )
    if (
        binding.get("receipt_sha256") != receipt_sha256
        or binding.get("binding_scope")
        != "endpoint_checkpoint_bytes_only_after_full_gate"
        or verification != expected_verification
        or binding.get("canonical_tree_sha256")
        != release.get("canonical_tree_sha256")
        or binding.get("claimed_inventory") != inventory
        or binding.get("claimed_level_count") != EXPECTED_LEVELS
        or binding.get("authoritative_level_count") != EXPECTED_LEVELS
        or run.get("claimed_levels") != EXPECTED_LEVELS
        or run.get("authoritative_levels") != EXPECTED_LEVELS
    ):
        raise LeaderboardV3Error(
            f"{mode} run is not bound to the complete release receipt"
        )
    receipt_path = run.get("release_receipt")
    if (
        not _logical_path_is_safe(receipt_path)
        or Path(receipt_path).stem != receipt_sha256
    ):
        raise LeaderboardV3Error(f"{mode} run names a different release receipt")

    command = run.get("command")
    artifact_root = run.get("artifact_root")
    if (
        not isinstance(command, dict)
        or set(command) != RUN_COMMAND_FIELDS
        or command.get("entrypoint") != "arc/crack_lab/replay_scorecard.py"
        or command.get("mode") != mode
        or not isinstance(command.get("games"), list)
        or len(command["games"]) != EXPECTED_GAMES
        or set(command["games"]) != set(inventory)
        or command.get("artifact_root") != artifact_root
        or command.get("release_receipt") != receipt_path
        or command.get("expected_claimed_levels") != EXPECTED_LEVELS
        or command.get("preflight_only") is not False
        or command.get("source_url") != run.get("source_url")
        or command.get("source_revision") != revision
        or command.get("tags") != run.get("scorecard_tags")
        or not _logical_path_is_safe(artifact_root)
    ):
        raise LeaderboardV3Error(
            f"{mode} run command identity is incomplete or host-specific"
        )

    checkpoints = run.get("checkpoint_sha256")
    evidence = release.get("evidence")
    expected_checkpoints: dict[str, str] = {}
    if not isinstance(evidence, dict):
        raise LeaderboardV3Error("complete release has no checkpoint evidence")
    for game, target in inventory.items():
        rows = evidence.get(game)
        endpoint = rows[-1] if isinstance(rows, list) and len(rows) == target else None
        checkpoint_sha256 = (
            endpoint.get("checkpoint_sha256")
            if isinstance(endpoint, dict)
            else None
        )
        if (
            not isinstance(checkpoint_sha256, str)
            or SHA256_RE.fullmatch(checkpoint_sha256) is None
        ):
            raise LeaderboardV3Error(
                f"complete release endpoint checkpoint evidence is invalid: {game}"
            )
        expected_checkpoints[game] = checkpoint_sha256
    if not isinstance(checkpoints, dict) or set(checkpoints) != set(inventory):
        raise LeaderboardV3Error(f"{mode} run checkpoint set is incomplete")
    if any(
        not isinstance(value, str) or SHA256_RE.fullmatch(value) is None
        for value in checkpoints.values()
    ):
        raise LeaderboardV3Error(f"{mode} run has an invalid checkpoint hash")
    if checkpoints != expected_checkpoints:
        raise LeaderboardV3Error(
            f"{mode} run checkpoint bytes differ from release evidence"
        )
    results = run.get("results")
    if not isinstance(results, dict) or set(results) != set(inventory):
        raise LeaderboardV3Error(f"{mode} run result set is incomplete")
    for game, target in inventory.items():
        result = results.get(game)
        if (
            not isinstance(result, dict)
            or set(result) != RESULT_FIELDS
            or result.get("remote") != target
            or result.get("claimed") != target
        ):
            raise LeaderboardV3Error(
                f"{mode} run endpoint differs from the release: {game}"
            )
    stored_actions = run.get("stored_actions")
    expected_stored_actions = _release_stored_actions(release, inventory)
    if stored_actions != expected_stored_actions:
        raise LeaderboardV3Error(
            f"{mode} run stored action count differs from release evidence"
        )
    _validate_closed_aggregate(
        run.get("aggregate"),
        run=run,
        mode=mode,
        inventory=inventory,
        allow_absent=remotely_recoverable,
    )
    return started, closed, remotely_recoverable


def _environment_depth(value: object) -> int | None:
    if not isinstance(value, dict):
        return None
    depth = value.get("levels_completed")
    if _is_int(depth):
        return depth
    runs = value.get("runs")
    if not isinstance(runs, list) or not runs:
        return None
    depths = [
        run.get("levels_completed")
        for run in runs
        if isinstance(run, dict) and _is_int(run.get("levels_completed"))
    ]
    return max(depths) if depths else None


def _scorecard_accounting(
    card: Mapping[str, Any],
    *,
    mode: str,
    run: Mapping[str, Any],
    inventory: Mapping[str, int],
    require_published: bool,
    allow_missing_closed_aggregate: bool,
) -> dict[str, Any]:
    expected_fields = set(PUBLIC_SCORECARD_FIELDS)
    if mode == "competition":
        expected_fields.add("competition_mode")
    if set(card) != expected_fields:
        raise LeaderboardV3Error(
            f"{mode} public scorecard top-level schema mismatch"
        )
    card_id = run["scorecard_id"]
    if card.get("card_id") != card_id:
        raise LeaderboardV3Error(f"{mode} public scorecard ID mismatch")
    public_competition = card.get("competition_mode", False)
    if not isinstance(public_competition, bool) or public_competition is not (
        mode == "competition"
    ):
        raise LeaderboardV3Error(f"{mode} public scorecard mode mismatch")
    if (
        card.get("source_url") != run.get("source_url")
        or card.get("tags") != run.get("scorecard_tags")
        or card.get("opaque") != run.get("scorecard_opaque")
        or card.get("ai_agent") is not True
    ):
        raise LeaderboardV3Error(
            f"{mode} public scorecard provenance metadata mismatch"
        )
    published = _parse_utc(
        card.get("published_at"), label=f"{mode} publication"
    )
    opened = _parse_utc(card.get("open_at"), label=f"{mode} public open")
    last_update = _parse_utc(
        card.get("last_update"), label=f"{mode} public last update"
    )
    close_started = _parse_utc(
        run.get("scorecard_close_started_at_utc"),
        label=f"{mode} close attempt start",
    )
    close_finished = _parse_utc(
        run.get("scorecard_close_finished_at_utc"),
        label=f"{mode} close response",
    )
    if require_published and not (
        close_started - PUBLICATION_CLOCK_SKEW
        <= published
        <= close_finished + PUBLICATION_CLOCK_SKEW
    ):
        raise LeaderboardV3Error(
            f"{mode} public publication is outside the bound close interval"
        )
    if opened > last_update or last_update > published + PUBLICATION_CLOCK_SKEW:
        raise LeaderboardV3Error(
            f"{mode} public scorecard timestamps are inconsistent"
        )
    environments = card.get("environments")
    if not isinstance(environments, list):
        raise LeaderboardV3Error(f"{mode} public scorecard has no environments")
    by_game: dict[str, Mapping[str, Any]] = {}
    for environment in environments:
        if not isinstance(environment, dict):
            raise LeaderboardV3Error(
                f"{mode} public scorecard has malformed accounting"
            )
        environment_id = environment.get("id")
        if not isinstance(environment_id, str):
            raise LeaderboardV3Error(
                f"{mode} public scorecard environment has no ID"
            )
        matches = [
            game
            for game in inventory
            if re.fullmatch(
                rf"{re.escape(game)}-[0-9a-f]{{8}}", environment_id
            )
            is not None
        ]
        if len(matches) != 1 or matches[0] in by_game:
            raise LeaderboardV3Error(
                f"{mode} public scorecard game accounting is ambiguous"
            )
        by_game[matches[0]] = environment
    if set(by_game) != set(inventory):
        raise LeaderboardV3Error(f"{mode} public scorecard game set mismatch")
    for game, target in inventory.items():
        _validate_complete_environment_score(
            by_game[game],
            game=game,
            target=target,
            public=True,
            label=f"{mode} public scorecard",
        )
    public_tag_scores = _validate_tag_scores(
        card.get("tags_scores"),
        public=True,
        label=f"{mode} public scorecard",
    )
    total_actions = sum(int(row["actions"]) for row in by_game.values())
    total_resets = sum(int(row["resets"]) for row in by_game.values())
    if (
        card.get("total_environments") != EXPECTED_GAMES
        or card.get("total_environments_completed") != EXPECTED_GAMES
        or card.get("total_levels_completed") != EXPECTED_LEVELS
        or card.get("total_levels") != EXPECTED_LEVELS
        or card.get("total_actions") != total_actions
    ):
        raise LeaderboardV3Error(f"{mode} public scorecard totals mismatch")
    score = card.get("score")
    if (
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not 0 <= score <= 100
    ):
        raise LeaderboardV3Error(f"{mode} public scorecard score is invalid")
    aggregate = run.get("aggregate")
    if allow_missing_closed_aggregate and aggregate is None:
        return {
            "score": float(score),
            "actions": total_actions,
            "resets": total_resets,
            "published_at": published,
        }
    if not isinstance(aggregate, dict):
        raise LeaderboardV3Error(
            f"{mode} public scorecard differs from its closed-card receipt"
        )
    aggregate_environments = aggregate.get("environments")
    aggregate_by_game: dict[str, Mapping[str, Any]] = {}
    if isinstance(aggregate_environments, list):
        for environment in aggregate_environments:
            if not isinstance(environment, dict):
                continue
            environment_id = environment.get("id")
            for game in inventory:
                if (
                    isinstance(environment_id, str)
                    and re.fullmatch(
                        rf"{re.escape(game)}-[0-9a-f]{{8}}", environment_id
                    )
                    is not None
                ):
                    aggregate_by_game[game] = environment
                    break
    for game in inventory:
        closed_environment = aggregate_by_game.get(game)
        public_environment = by_game[game]
        if not isinstance(closed_environment, dict):
            raise LeaderboardV3Error(
                f"{mode} public environment lacks a closed-card binding: {game}"
            )
        projected = dict(closed_environment)
        projected["runs"] = [
            _public_run_projection(run)
            for run in closed_environment["runs"]
        ]
        if projected != public_environment:
            raise LeaderboardV3Error(
                f"{mode} public game accounting differs from the receipt: {game}"
            )
    closed_tag_scores = _validate_tag_scores(
        aggregate.get("tags_scores"),
        public=False,
        label=f"{mode} closed aggregate",
    )
    if {
        key: _public_tag_projection(value)
        for key, value in closed_tag_scores.items()
    } != public_tag_scores:
        raise LeaderboardV3Error(
            f"{mode} public tag accounting differs from the receipt"
        )
    for field in (
        "card_id",
        "score",
        "tags",
        "opaque",
        "total_environments_completed",
        "total_environments",
        "total_levels_completed",
        "total_levels",
        "total_actions",
    ):
        if aggregate.get(field) != card.get(field):
            raise LeaderboardV3Error(
                f"{mode} public scorecard differs from receipt field {field}"
            )
    return {
        "score": float(score),
        "actions": total_actions,
        "resets": total_resets,
        "published_at": published,
    }


def _validate_authors(candidate: Mapping[str, Any]) -> None:
    authors = candidate.get("authors")
    if not isinstance(authors, list):
        raise LeaderboardV3Error("submission authors must be a list")
    names = tuple(
        author.get("name") if isinstance(author, dict) else None
        for author in authors
    )
    if names != REQUIRED_AUTHOR_NAMES:
        raise LeaderboardV3Error(
            "submission authors must name Alexander Kolpakov and the "
            "OpenAI GPT-5.6 model"
        )
    for author in authors:
        assert isinstance(author, dict)
        links = [
            author.get(field)
            for field in ("url", "twitter", "linkedin", "scholar", "github")
        ]
        if not any(
            isinstance(link, str) and link.startswith("https://")
            for link in links
        ):
            raise LeaderboardV3Error("each submission author needs a public link")


def _validate_candidate_yaml(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    competition_run: Mapping[str, Any],
    competition_published_at: dt.datetime,
) -> None:
    if set(candidate) != set(baseline):
        raise LeaderboardV3Error("candidate YAML top-level schema mismatch")
    if candidate.get("name") != baseline.get("name"):
        raise LeaderboardV3Error("submission name changed from the v2 baseline")
    _validate_authors(candidate)
    preserved_root_fields = set(baseline) - {"code_url", "versions"}
    if any(candidate.get(field) != baseline.get(field) for field in preserved_root_fields):
        raise LeaderboardV3Error(
            "submission root metadata changed from the v2 baseline"
        )

    baseline_versions = baseline.get("versions")
    candidate_versions = candidate.get("versions")
    if not isinstance(baseline_versions, list) or len(baseline_versions) != 2:
        raise LeaderboardV3Error("baseline must contain exactly historical v1/v2")
    if [row.get("version") for row in baseline_versions if isinstance(row, dict)] != [
        "1.0",
        "2.0",
    ]:
        raise LeaderboardV3Error("baseline historical versions are not v1.0/v2.0")
    if not isinstance(candidate_versions, list):
        raise LeaderboardV3Error("candidate versions must be a list")
    if candidate_versions[:2] != baseline_versions:
        raise LeaderboardV3Error("historical v1/v2 entries were mutated")
    if len(candidate_versions) != 3:
        raise LeaderboardV3Error(
            "candidate must append exactly one v3 entry; v3 is missing or duplicated"
        )
    v3 = candidate_versions[2]
    if not isinstance(v3, dict) or set(v3) != V3_FIELDS:
        raise LeaderboardV3Error("candidate v3 field schema mismatch")
    if v3.get("version") != "3.0":
        raise LeaderboardV3Error("candidate is missing the single v3.0 entry")
    changes = v3.get("changes")
    if not isinstance(changes, str) or "183/183" not in changes:
        raise LeaderboardV3Error(
            "v3 changes must identify the complete 183/183 release"
        )
    date_value = v3.get("date")
    if isinstance(date_value, dt.datetime):
        raise LeaderboardV3Error("v3 date must be a YYYY-MM-DD date scalar")
    elif isinstance(date_value, dt.date):
        v3_date = date_value
    elif (
        isinstance(date_value, str)
        and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", date_value)
        is not None
    ):
        try:
            v3_date = dt.date.fromisoformat(date_value)
        except ValueError as exc:
            raise LeaderboardV3Error("v3 date must be YYYY-MM-DD") from exc
    else:
        raise LeaderboardV3Error("v3 date must be YYYY-MM-DD")
    if v3_date != competition_published_at.date():
        raise LeaderboardV3Error(
            "v3 date must equal the UTC Competition publication date"
        )

    models = v3.get("models")
    if not isinstance(models, list):
        raise LeaderboardV3Error("v3 models are missing")
    model_names = {
        model.get("name") for model in models if isinstance(model, dict)
    }
    if (
        model_names != REQUIRED_V3_MODELS
        or len(models) != len(model_names)
        or any(
            not isinstance(model, dict) or set(model) != MODEL_FIELDS
            for model in models
        )
    ):
        raise LeaderboardV3Error("v3 model/proposer lineage metadata is incorrect")

    scores = v3.get("scores")
    if not isinstance(scores, list) or len(scores) != 1:
        raise LeaderboardV3Error("v3 must contain one ARC-AGI-3 scorecard entry")
    score = scores[0]
    if isinstance(score, dict) and "score" in score:
        raise LeaderboardV3Error("ARC-AGI-3 forbids a numeric score field")
    if isinstance(score, dict) and "cost" in score:
        raise LeaderboardV3Error("v3 public YAML must omit optional cost")
    if (
        not isinstance(score, dict)
        or set(score) != SCORE_FIELDS
        or score.get("benchmark") != "arc-agi-3"
    ):
        raise LeaderboardV3Error("v3 score entry must target arc-agi-3")
    if score.get("set") != "public":
        raise LeaderboardV3Error("v3 ARC-AGI-3 set must be public")
    if score.get("scorecard_url") != competition_run.get("scorecard_url"):
        raise LeaderboardV3Error(
            "v3 scorecard URL differs from the Competition run receipt"
        )
    _scorecard_id_from_url(score.get("scorecard_url"), label="v3 scorecard URL")

    revision = competition_run.get("source_revision")
    code_url = candidate.get("code_url")
    if not isinstance(revision, str) or REVISION_RE.fullmatch(revision) is None:
        raise LeaderboardV3Error("Competition source revision is invalid")
    expected_code_url = (
        f"https://github.com/sashakolpakov/gkm/tree/{revision}"
    )
    if code_url != expected_code_url:
        raise LeaderboardV3Error(
            "code_url must be the exact immutable repository-root URL for the "
            "scored public revision"
        )


def _validate_readme(
    text: str,
    *,
    release_sha256: str,
    online_run: Mapping[str, Any],
    competition_run: Mapping[str, Any],
    competition_score: float,
    accounting: Mapping[str, Any],
    close_recoveries: Sequence[str],
) -> None:
    lowered = text.lower()
    for column in PROVENANCE_COLUMNS:
        if column not in lowered:
            raise LeaderboardV3Error(
                f"candidate README provenance table is missing {column!r}"
            )
    required_fragments = (
        "183/183",
        "OpenAI GPT-5.6",
        release_sha256,
        str(competition_run["source_revision"]),
    )
    for fragment in required_fragments:
        if fragment not in text:
            raise LeaderboardV3Error(
                f"candidate README is missing release fact {fragment!r}"
            )
    scorecard_urls = [
        value.rstrip(".,;") for value in ARC_SCORECARD_URL_RE.findall(text)
    ]
    if scorecard_urls != [competition_run["scorecard_url"]]:
        raise LeaderboardV3Error(
            "candidate README must contain exactly the definitive Competition "
            "scorecard URL"
        )
    score_facts = OFFICIAL_SCORE_RE.findall(text)
    if len(score_facts) != 1 or float(score_facts[0]) != competition_score:
        raise LeaderboardV3Error(
            "candidate README has missing or conflicting official score facts"
        )
    expected_accounting_block = "\n".join(
        (
            f"Complete games: {accounting['complete_games']}",
            "Raw coverage: "
            f"{accounting['raw_levels']}/{accounting['authoritative_levels']} "
            f"({accounting['raw_coverage_percent']:g}%)",
            f"Frozen stored actions: {accounting['stored_actions']}",
            f"Official Competition actions: {accounting['official_actions']}",
            f"Official Competition resets: {accounting['total_resets']}",
            f"ARC toolkit: {accounting['arc_agi_toolkit_version']}",
            "Close recovery: "
            + (", ".join(close_recoveries) if close_recoveries else "none"),
        )
    )
    if text.count(expected_accounting_block) != 1:
        raise LeaderboardV3Error(
            "candidate README is missing the canonical accounting block"
        )
    if REVERSE_ACCOUNTING_FACT_RE.search(text):
        raise LeaderboardV3Error(
            "candidate README contains an ambiguous number-first accounting fact"
        )
    action_facts = [
        ((label or "").lower().replace("  ", " "), int(value))
        for label, value in ACTION_FACT_RE.findall(text)
    ]
    expected_action_facts = {
        "frozen stored": int(accounting["stored_actions"]),
        "official competition": int(accounting["official_actions"]),
    }
    if (
        len(action_facts) != 2
        or {label for label, _ in action_facts} != set(expected_action_facts)
        or any(expected_action_facts.get(label) != value for label, value in action_facts)
    ):
        raise LeaderboardV3Error(
            "candidate README has missing, ambiguous, or conflicting action facts"
        )
    reset_facts = [
        ((label or "").lower().replace("  ", " "), int(value))
        for label, value in RESET_FACT_RE.findall(text)
    ]
    if reset_facts != [
        ("official competition", int(accounting["total_resets"]))
    ]:
        raise LeaderboardV3Error(
            "candidate README has missing or conflicting reset facts"
        )
    toolkit_facts = TOOLKIT_FACT_RE.findall(text)
    if toolkit_facts != [str(accounting["arc_agi_toolkit_version"])]:
        raise LeaderboardV3Error(
            "candidate README has missing or conflicting toolkit facts"
        )
    complete_games = COMPLETE_GAMES_RE.findall(text)
    if complete_games != [str(accounting["complete_games"])]:
        raise LeaderboardV3Error(
            "candidate README has missing or conflicting complete-game facts"
        )
    raw_facts = RAW_COVERAGE_RE.findall(text)
    if len(raw_facts) != 1 or (
        int(raw_facts[0][0]),
        int(raw_facts[0][1]),
        float(raw_facts[0][2]) if raw_facts[0][2] else None,
    ) != (
        int(accounting["raw_levels"]),
        int(accounting["authoritative_levels"]),
        float(accounting["raw_coverage_percent"]),
    ):
        raise LeaderboardV3Error(
            "candidate README has missing or conflicting raw-coverage facts"
        )
    if ANY_COVERAGE_RE.findall(text) != [
        (
            str(accounting["raw_levels"]),
            str(accounting["authoritative_levels"]),
        )
    ]:
        raise LeaderboardV3Error(
            "candidate README has an extra or conflicting coverage claim"
        )
    recovery_facts = CLOSE_RECOVERY_RE.findall(text)
    expected_recovery = ", ".join(close_recoveries) if close_recoveries else "none"
    if (
        len(recovery_facts) != 1
        or recovery_facts[0].strip().rstrip(".\t ").lower()
        != expected_recovery.lower()
    ):
        raise LeaderboardV3Error(
            "candidate README has missing or conflicting close-recovery facts"
        )


def expected_pr_title(competition_score: float) -> str:
    """Render PR percentages to four decimals, matching the frozen v2 title."""
    return (
        f"Add GKM — {competition_score:.4f}% / 100.0000% raw: general-purpose "
        "replay-gated self-improving program synthesis"
    )


def validate_v3_payload(
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    candidate_readme: str,
    release: Mapping[str, Any],
    release_sha256: str,
    inventory: Mapping[str, int],
    online_run: Mapping[str, Any],
    competition_run: Mapping[str, Any],
    online_journal_snapshot: bytes,
    competition_journal_snapshot: bytes,
    online_public: Mapping[str, Any],
    competition_public: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_public_artifact_strings(candidate, label="candidate YAML")
    _validate_public_text(candidate_readme, label="candidate README")
    online_started, online_closed, online_recovered = _run_binding(
        online_run,
        mode="online",
        receipt_sha256=release_sha256,
        release=release,
        inventory=inventory,
    )
    competition_started, competition_closed, competition_recovered = _run_binding(
        competition_run,
        mode="competition",
        receipt_sha256=release_sha256,
        release=release,
        inventory=inventory,
    )
    if online_recovered:
        raise LeaderboardV3Error(
            "Competition admission requires an ONLINE PASS with confirmed close"
        )
    if online_run.get("stored_actions") != competition_run.get("stored_actions"):
        raise LeaderboardV3Error(
            "ONLINE/Competition stored action accounting differs"
        )
    _validate_journal_snapshot(
        online_journal_snapshot,
        run=online_run,
        mode="online",
        receipt_sha256=release_sha256,
        release=release,
    )
    _validate_journal_snapshot(
        competition_journal_snapshot,
        run=competition_run,
        mode="competition",
        receipt_sha256=release_sha256,
        release=release,
    )
    online_journal = online_run["run_journal"]
    competition_journal = competition_run["run_journal"]
    if (
        not competition_journal_snapshot.startswith(online_journal_snapshot)
        or online_journal.get("live_journal")
        != competition_journal.get("live_journal")
        or online_journal.get("journal_id")
        != competition_journal.get("journal_id")
        or online_journal.get("operation_id")
        == competition_journal.get("operation_id")
        or online_journal.get("terminal_sequence")
        >= competition_journal.get("intent_sequence")
    ):
        raise LeaderboardV3Error(
            "ONLINE/Competition journal sequence or identity mismatch"
        )
    if online_closed > competition_started:
        raise LeaderboardV3Error(
            "Competition started before the ONLINE shakedown closed"
        )
    for field in (
        "source_revision",
        "source_url",
        "arc_agi_toolkit_version",
        "release_binding",
        "checkpoint_sha256",
    ):
        if online_run.get(field) != competition_run.get(field):
            raise LeaderboardV3Error(
                f"ONLINE/Competition source-receipt mismatch at {field}"
            )
    if online_run.get("scorecard_id") == competition_run.get("scorecard_id"):
        raise LeaderboardV3Error("ONLINE and Competition reused one scorecard")
    public_revision = verify_public_revision(
        str(competition_run.get("source_revision"))
    )

    online_accounting = _scorecard_accounting(
        online_public,
        mode="online",
        run=online_run,
        inventory=inventory,
        require_published=True,
        allow_missing_closed_aggregate=online_recovered,
    )
    competition_accounting = _scorecard_accounting(
        competition_public,
        mode="competition",
        run=competition_run,
        inventory=inventory,
        require_published=True,
        allow_missing_closed_aggregate=competition_recovered,
    )
    competition_score = float(competition_accounting["score"])
    online_published = online_accounting["published_at"]
    competition_published = competition_accounting["published_at"]
    if online_published > competition_published:
        raise LeaderboardV3Error(
            "Competition was published before the ONLINE shakedown"
        )
    if competition_closed < competition_started or online_started > online_closed:
        raise LeaderboardV3Error("run receipt timestamp accounting is invalid")

    _validate_candidate_yaml(
        baseline,
        candidate,
        competition_run=competition_run,
        competition_published_at=competition_published,
    )
    close_recoveries = [
        mode
        for mode, recovered in (
            ("online", online_recovered),
            ("competition", competition_recovered),
        )
        if recovered
    ]
    accounting = {
        "complete_games": EXPECTED_GAMES,
        "raw_levels": EXPECTED_LEVELS,
        "authoritative_levels": EXPECTED_LEVELS,
        "raw_coverage_percent": 100.0,
        "stored_actions": _release_stored_actions(release, inventory),
        "official_actions": int(competition_accounting["actions"]),
        "total_resets": int(competition_accounting["resets"]),
        "arc_agi_toolkit_version": AUDITED_TOOLKIT_VERSION,
    }
    _validate_readme(
        candidate_readme,
        release_sha256=release_sha256,
        online_run=online_run,
        competition_run=competition_run,
        competition_score=competition_score,
        accounting=accounting,
        close_recoveries=close_recoveries,
    )
    return {
        "schema": 1,
        "status": "PASS",
        "release_receipt_sha256": release_sha256,
        "canonical_tree_sha256": release["canonical_tree_sha256"],
        "public_revision": competition_run["source_revision"],
        "public_revision_url": public_revision["html_url"],
        "online_scorecard_url": online_run["scorecard_url"],
        "competition_scorecard_url": competition_run["scorecard_url"],
        "competition_score": competition_score,
        **accounting,
        "expected_pr_title": expected_pr_title(competition_score),
        "remote_close_recoveries": close_recoveries,
    }


def fetch_public_scorecard(card_id: str) -> dict[str, Any]:
    if UUID_RE.fullmatch(card_id) is None:
        raise LeaderboardV3Error("cannot fetch an invalid scorecard ID")
    request = urllib.request.Request(
        PUBLIC_SCORECARD_ENDPOINT.format(card_id=card_id),
        headers={"Accept": "application/json", "User-Agent": "gkm-v3-gate/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            raw = response.read(MAX_RECORD_BYTES + 1)
    except (OSError, urllib.error.URLError) as exc:
        raise LeaderboardV3Error(
            f"cannot reopen public scorecard {card_id}"
        ) from exc
    if len(raw) > MAX_RECORD_BYTES:
        raise LeaderboardV3Error("public scorecard response is unexpectedly large")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise LeaderboardV3Error("public scorecard response is invalid JSON") from exc
    if not isinstance(value, dict):
        raise LeaderboardV3Error("public scorecard response is not an object")
    _reject_nonfinite(value, label="public scorecard response")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline-yaml", type=Path, required=True)
    parser.add_argument("--candidate-yaml", type=Path, required=True)
    parser.add_argument("--candidate-readme", type=Path, required=True)
    parser.add_argument("--release-receipt", type=Path, required=True)
    parser.add_argument("--canonical-release-root", type=Path, required=True)
    parser.add_argument("--release-verifier-root", type=Path)
    parser.add_argument("--online-run-receipt", type=Path, required=True)
    parser.add_argument("--competition-run-receipt", type=Path, required=True)
    parser.add_argument("--online-journal-snapshot", type=Path, required=True)
    parser.add_argument(
        "--competition-journal-snapshot", type=Path, required=True
    )
    args = parser.parse_args(argv)
    try:
        baseline, _baseline_bytes = _load_yaml(
            args.baseline_yaml,
            label="v2 baseline YAML",
            expected_sha256=V2_BASELINE_YAML_SHA256,
        )
        candidate, candidate_bytes = _load_yaml(
            args.candidate_yaml, label="v3 candidate YAML"
        )
        readme_bytes = _read_regular(
            args.candidate_readme, label="v3 candidate README"
        )
        candidate_readme = readme_bytes.decode("utf-8")
        release, release_sha256, inventory = validate_complete_release_receipt(
            args.release_receipt,
            canonical_root=args.canonical_release_root,
            verifier_root=args.release_verifier_root,
        )
        online_run = _load_json(
            args.online_run_receipt, label="ONLINE run receipt"
        )
        competition_run = _load_json(
            args.competition_run_receipt, label="Competition run receipt"
        )
        online_journal_snapshot = _read_regular(
            args.online_journal_snapshot,
            label="ONLINE run-journal snapshot",
        )
        competition_journal_snapshot = _read_regular(
            args.competition_journal_snapshot,
            label="Competition run-journal snapshot",
        )
        for snapshot_path, snapshot_raw, label in (
            (
                args.online_journal_snapshot,
                online_journal_snapshot,
                "ONLINE",
            ),
            (
                args.competition_journal_snapshot,
                competition_journal_snapshot,
                "Competition",
            ),
        ):
            if snapshot_path.stem != _sha256_bytes(snapshot_raw):
                raise LeaderboardV3Error(
                    f"{label} run-journal snapshot is not content-addressed"
                )
        online_public = fetch_public_scorecard(str(online_run.get("scorecard_id")))
        competition_public = fetch_public_scorecard(
            str(competition_run.get("scorecard_id"))
        )
        summary = validate_v3_payload(
            baseline=baseline,
            candidate=candidate,
            candidate_readme=candidate_readme,
            release=release,
            release_sha256=release_sha256,
            inventory=inventory,
            online_run=online_run,
            competition_run=competition_run,
            online_journal_snapshot=online_journal_snapshot,
            competition_journal_snapshot=competition_journal_snapshot,
            online_public=online_public,
            competition_public=competition_public,
        )
        summary["candidate_yaml_sha256"] = _sha256_bytes(candidate_bytes)
        summary["candidate_readme_sha256"] = _sha256_bytes(readme_bytes)
    except (
        LeaderboardV3Error,
        OSError,
        UnicodeError,
        ValueError,
        yaml.YAMLError,
    ) as exc:
        print(f"v3 leaderboard gate FAIL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
