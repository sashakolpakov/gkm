#!/usr/bin/env python3
"""Audit ARC-AGI-3 action tokens and legacy acquisition evidence.

The public ACTION6 contract is a click in the returned 64x64 observation:
``[6, x, y]`` with integer ``x`` and ``y`` in ``0..63``.  This audit has two
separate jobs:

1. deterministically validate every current and promotion checkpoint token;
2. search legacy proposer transcripts for affirmative evidence that an
   out-of-range exploratory action was attempted.

Legacy GKM Arena sessions did not record every call at a trusted host boundary.
Consequently, a clean transcript scan is not proof that no dynamic illegal call
occurred.  The report states that limitation explicitly.  The contiguous
campaign closes it with a host-authenticated RPC transcript and should be run
with ``--require-complete-call-log`` against its release evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


FRAME_SIDE = 64
MAX_TRANSCRIPT_BYTES = 50_000_000
PROTOCOL_VIOLATION_MARKER = "GKM_PUBLIC_ACTION_PROTOCOL_VIOLATION"

_STEP_CALL_RE = re.compile(
    r"\.step\s*\(\s*6\s*,\s*(-?\d+)\s*,\s*(-?\d+)",
    re.IGNORECASE,
)
_ACTION_TOKEN_RE = re.compile(
    r"\[\s*6\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]"
)
_EFFECT_PAIR_RE = re.compile(
    r"\b(?:run|click(?:ed|ing)?|step(?:ped|ping)?|action|probe|experiment)"
    r"\b\s*(?:(?:at|to|on)\s*)?[\[(]\s*(-?\d+)\s*,\s*(-?\d+)\s*[\])]",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CheckpointViolation:
    game: str
    level: int
    checkpoint: str
    action_index: int
    action: object
    reason: str


@dataclass(frozen=True)
class TranscriptFinding:
    game: str
    level: int
    transcript: str
    transcript_sha256: str
    event_line: int
    surface: str
    pattern: str
    x: int | None
    y: int | None
    snippet: str


def _plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def action_error(action: object) -> str | None:
    """Return the public-protocol error for one replay token, if any."""

    if _plain_int(action):
        return (
            None
            if 1 <= int(action) <= 7 and int(action) != 6
            else "key action is outside 1..5 or 7"
        )
    if not isinstance(action, (list, tuple)):
        return "action is neither a plain integer nor [6, x, y]"
    if len(action) != 3:
        return "coordinate action does not have three elements"
    if any(not _plain_int(value) for value in action):
        return "coordinate action contains a non-integer or boolean"
    if action[0] != 6:
        return "compound action does not start with ACTION6"
    if not all(0 <= int(value) < FRAME_SIDE for value in action[1:]):
        return "ACTION6 coordinate is outside 0..63"
    return None


def _checkpoint_files(root: Path) -> list[Path]:
    return sorted(
        set(root.glob("*_legs/checkpoint.json"))
        | set(
            root.glob(
                "*_legs/promotion_evidence/level_*/files/checkpoint.json"
            )
        )
    )


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _checkpoint_scan(
    root: Path,
) -> tuple[list[CheckpointViolation], dict[str, int], set[str], int]:
    violations: list[CheckpointViolation] = []
    reached_by_game: dict[str, int] = {}
    coordinate_games: set[str] = set()
    action_count = 0
    for path in _checkpoint_files(root):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            violations.append(
                CheckpointViolation(
                    game=path.parts[-2].removesuffix("_legs"),
                    level=-1,
                    checkpoint=_relative(path, root),
                    action_index=-1,
                    action=None,
                    reason=f"checkpoint is unreadable: {type(exc).__name__}",
                )
            )
            continue
        game = value.get("game")
        level = value.get("reached")
        actions = value.get("final_path")
        if not isinstance(game, str) or not _plain_int(level):
            violations.append(
                CheckpointViolation(
                    game=str(game),
                    level=-1,
                    checkpoint=_relative(path, root),
                    action_index=-1,
                    action=None,
                    reason="checkpoint game/reached schema is invalid",
                )
            )
            continue
        if path.parent.name.endswith("_legs"):
            reached_by_game[game] = max(reached_by_game.get(game, 0), level)
        if not isinstance(actions, list):
            violations.append(
                CheckpointViolation(
                    game=game,
                    level=level,
                    checkpoint=_relative(path, root),
                    action_index=-1,
                    action=actions,
                    reason="final_path is not an array",
                )
            )
            continue
        for index, action in enumerate(actions):
            action_count += 1
            if (
                isinstance(action, (list, tuple))
                and len(action) == 3
                and action[0] == 6
            ):
                coordinate_games.add(game)
            reason = action_error(action)
            if reason is not None:
                violations.append(
                    CheckpointViolation(
                        game=game,
                        level=level,
                        checkpoint=_relative(path, root),
                        action_index=index,
                        action=action,
                        reason=reason,
                    )
                )
    return violations, reached_by_game, coordinate_games, action_count


def _game_level(path: Path) -> tuple[str, int]:
    game = next(
        (
            part.removesuffix("_legs")
            for part in path.parts
            if part.endswith("_legs")
        ),
        "unknown",
    )
    level = next(
        (
            int(part.removeprefix("level_"))
            for part in path.parts
            if re.fullmatch(r"level_\d+", part)
        ),
        -1,
    )
    return game, level


def _out_of_range(x: int, y: int) -> bool:
    return not (0 <= x < FRAME_SIDE and 0 <= y < FRAME_SIDE)


def _text_findings(
    text: str,
    *,
    game: str,
    level: int,
    transcript: str,
    digest: str,
    event_line: int,
    surface: str,
) -> Iterable[TranscriptFinding]:
    seen: set[tuple[int, int, str, int]] = set()
    patterns = (
        ("step_call", _STEP_CALL_RE),
        ("action_token", _ACTION_TOKEN_RE),
        ("effect_pair", _EFFECT_PAIR_RE),
    )
    for field_line, line in enumerate(text.splitlines(), 1):
        if PROTOCOL_VIOLATION_MARKER in line:
            snippet = re.sub(r"\s+", " ", line).strip()
            if len(snippet) > 240:
                snippet = snippet[:237] + "..."
            yield TranscriptFinding(
                game=game,
                level=level,
                transcript=transcript,
                transcript_sha256=digest,
                event_line=event_line,
                surface=f"{surface}:{field_line}",
                pattern="latched_protocol_violation",
                x=None,
                y=None,
                snippet=snippet,
            )
        for label, pattern in patterns:
            for match in pattern.finditer(line):
                x, y = int(match.group(1)), int(match.group(2))
                if not _out_of_range(x, y):
                    continue
                key = (x, y, label, field_line)
                if key in seen:
                    continue
                seen.add(key)
                snippet = re.sub(r"\s+", " ", line).strip()
                if len(snippet) > 240:
                    snippet = snippet[:237] + "..."
                yield TranscriptFinding(
                    game=game,
                    level=level,
                    transcript=transcript,
                    transcript_sha256=digest,
                    event_line=event_line,
                    surface=f"{surface}:{field_line}",
                    pattern=label,
                    x=x,
                    y=y,
                    snippet=snippet,
                )


def _scan_transcript(
    path: Path,
    root: Path,
    digest: str,
) -> list[TranscriptFinding]:
    raw = path.read_bytes()
    if len(raw) > MAX_TRANSCRIPT_BYTES:
        raise ValueError(
            f"transcript exceeds {MAX_TRANSCRIPT_BYTES} bytes: {path}"
        )
    game, level = _game_level(path)
    relative = _relative(path, root)
    findings: list[TranscriptFinding] = []
    for event_line, raw_line in enumerate(raw.splitlines(), 1):
        try:
            event = json.loads(raw_line)
        except (UnicodeError, json.JSONDecodeError):
            continue
        if event.get("type") != "item.completed":
            continue
        item = event.get("item")
        if not isinstance(item, dict):
            continue
        if item.get("type") == "command_execution":
            fields = (
                ("authored_command", item.get("command")),
                ("command_output", item.get("aggregated_output")),
            )
        elif item.get("type") == "agent_message":
            fields = (("agent_claim", item.get("text")),)
        else:
            continue
        for surface, value in fields:
            if not isinstance(value, str):
                continue
            findings.extend(
                _text_findings(
                    value,
                    game=game,
                    level=level,
                    transcript=relative,
                    digest=digest,
                    event_line=event_line,
                    surface=surface,
                )
            )
    return findings


def _transcript_scan(root: Path) -> tuple[list[TranscriptFinding], int, int]:
    paths = sorted(
        root.glob(
            "*_legs/promotion_evidence/level_*/codex_turns/*.jsonl"
        )
    )
    by_digest: dict[str, Path] = {}
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        by_digest.setdefault(digest, path)
    findings: list[TranscriptFinding] = []
    for digest, path in sorted(by_digest.items(), key=lambda item: str(item[1])):
        findings.extend(_scan_transcript(path, root, digest))
    return findings, len(paths), len(by_digest)


def run(root: Path, *, complete_call_log: bool = False) -> dict:
    root = root.resolve()
    violations, reached, coordinate_games, action_count = _checkpoint_scan(root)
    findings, transcript_copies, unique_transcripts = _transcript_scan(root)
    affected_games = sorted(
        {finding.game for finding in findings}
        | {violation.game for violation in violations}
    )
    affected_levels = sum(reached.get(game, 0) for game in affected_games)
    legacy_incomplete = sorted(coordinate_games) if not complete_call_log else []
    evidence_verdict = "FAIL" if violations or findings else "PASS"
    strict_verdict = (
        "FAIL"
        if evidence_verdict == "FAIL" or legacy_incomplete
        else "PASS"
    )
    return {
        "schema": 1,
        "contract": {
            "simple_actions": "plain integers 1..5 or 7",
            "coordinate_action": "[6, x, y]",
            "coordinate_bounds": {"minimum": 0, "maximum": 63},
        },
        "root": str(root),
        "checkpoint_actions_scanned": action_count,
        "checkpoint_violations": [
            asdict(violation) for violation in violations
        ],
        "promotion_transcript_copies": transcript_copies,
        "unique_promotion_transcripts": unique_transcripts,
        "out_of_range_transcript_findings": [
            asdict(finding) for finding in findings
        ],
        "affected_games": affected_games,
        "affected_promoted_levels": affected_levels,
        "coordinate_action_games": sorted(coordinate_games),
        "complete_host_call_log": complete_call_log,
        "legacy_coordinate_games_without_complete_call_log": legacy_incomplete,
        "evidence_verdict": evidence_verdict,
        "strict_verdict": strict_verdict,
        "interpretation": (
            "A clean legacy scan proves token/path validity and absence of "
            "recorded affirmative incidents, not absence of every dynamic "
            "exploration call. Strict proof requires the contiguous host RPC "
            "call log."
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent
        / "crack_lab"
        / "agent_solutions",
    )
    parser.add_argument(
        "--require-complete-call-log",
        action="store_true",
        help="fail unless evidence comes from a complete trusted call log",
    )
    parser.add_argument("--json", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(
        args.root,
        complete_call_log=args.require_complete_call_log,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)
    verdict = (
        report["strict_verdict"]
        if args.require_complete_call_log
        else report["evidence_verdict"]
    )
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
