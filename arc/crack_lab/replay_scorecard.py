#!/usr/bin/env python3
"""Replay the promoted, replay-validated artifact paths against the live
ARC-AGI-3 API to produce a scorecard — WITHOUT re-running any discovery.

The expensive part of GKM (proposer-driven search) already happened and its
result is a literal action path in each promoted checkpoint
(agent_solutions/<game>_legs/checkpoint.json, replay-validated locally). This
tool only replays those paths through the official `arc_agi` toolkit.  The
exact API-action count is determined by the frozen checkpoints; replay uses
zero LLM tokens.

Modes (docs.arcprize.org/toolkit/competition_mode):
  --mode online       dry run: same remote API, no competition constraints.
                      Use this FIRST to check the recorded paths reproduce
                      remotely (a desync here costs nothing).
  --mode competition  the real thing: single scorecard, each environment may
                      be made once, scoring is against ALL environments (the
                      untouched ones count as 0), game resets become level
                      resets. The closed scorecard is what the community
                      leaderboard links as scorecard_url.

Canonical all-game invocations must also provide ``--games all``, the frozen
``--artifact-root``, its content-addressed ``--release-receipt``, and the
expected claimed-level count.  See ``REPRODUCE_ARC.md`` for the exact frozen
release commands; bare mode-only examples intentionally select only the small
demonstration subset and are not publication runs.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import os
import re
import secrets
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from verify_frozen_release import (
    FrozenReleaseError,
    load_receipt,
    verify_frozen_release,
)

LAB = Path(__file__).resolve().parent
GKM = LAB.parents[1]

DEFAULT_SOURCE_URL = "https://github.com/sashakolpakov/gkm"
DEFAULT_ARTIFACT_ROOT = LAB / "agent_solutions"
RUN_JOURNAL_ROOT = LAB / "run_journals"
MAX_CHECKPOINT_BYTES = 32 * 1024 * 1024
MAX_JOURNAL_BYTES = 32 * 1024 * 1024
AUDITED_TOOLKIT_VERSION = "0.9.9"
PUBLIC_COMMIT_ENDPOINT = (
    "https://api.github.com/repos/sashakolpakov/gkm/commits/{revision}"
)
REVISION_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RUNTIME_TRACKED_PATHS = (
    "arc/crack_lab/replay_scorecard.py",
    "arc/crack_lab/verify_frozen_release.py",
    "arc/arc_agi3_adapter.py",
    "cone/cone_foraging.py",
    "cone/cone_foraging_bound.py",
    "arc/crack_lab/arc_agi3_release_gate.py",
    "arc/crack_lab/arc_agi3_contiguous_conformance.py",
    "arc/crack_lab/arc_agi3_source_schema.py",
)
RUNTIME_ENVIRONMENT_SOURCE_SHA256 = {
    "ar25": ("environment_files/ar25/0c556536/ar25.py", "b965d2faaa4a4d9119c20eb51af0fb2b657fb52fb9884919863a48f2230c4272"),
    "bp35": ("environment_files/bp35/0a0ad940/bp35.py", "e9aecb52c629c3e742276c2db04b81f91555051d8617b6cbc58bac410824867f"),
    "cd82": ("environment_files/cd82/fb555c5d/cd82.py", "dafeabff60a3a99b5c93629f8da0d8f555b873ef16a522247d6197b1b3addbcb"),
    "cn04": ("environment_files/cn04/2fe56bfb/cn04.py", "29977e145a0374b5842d360507f045029c3f4b22373612152e713c14691e01dc"),
    "dc22": ("environment_files/dc22/fdcac232/dc22.py", "74abe3523bd6bc8a0948991caa0ca71855e38542ba2123653a03485e064f252e"),
    "ft09": ("environment_files/ft09/0d8bbf25/ft09.py", "aa5b54f48a29da9f276c3df0dc04de728002f523384ab868c85d56d819cd0783"),
    "g50t": ("environment_files/g50t/5849a774/g50t.py", "6f21f2065744e484ac38534f940158034e69f39add3b10a3d9fd258de7293aa9"),
    "ka59": ("environment_files/ka59/38d34dbb/ka59.py", "3e8341f4c317ceb9992ca54e8ad453ee4e8a04547b27a907f4a560a9a30d857e"),
    "lf52": ("environment_files/lf52/271a04aa/lf52.py", "a6b01973a0905a447f6b92d77745719ebb001f9f94ff5bb0076200ef16839030"),
    "lp85": ("environment_files/lp85/305b61c3/lp85.py", "7d10baec7dd29d07e574bd47ee1ec7912542a72632077a5446798221f41c4ae9"),
    "ls20": ("environment_files/ls20/9607627b/ls20.py", "298c810da2850d557c95d92a2cbd846df29a45d7134e20888617bedf5dafcd92"),
    "m0r0": ("environment_files/m0r0/492f87ba/m0r0.py", "fc3954236f712759c2a5baaada09bb3244e217e022086dc9ec8f6e6fe101d797"),
    "r11l": ("environment_files/r11l/495a7899/r11l.py", "95d58ebaa8c758e905821bb4f5881d47166c8f5aec158bfea7585902c01040d0"),
    "re86": ("environment_files/re86/8af5384d/re86.py", "cf3c1520e17f0b70e7a3cea0e680601a591befce306b4dd1926c735a7d3b65e8"),
    "s5i5": ("environment_files/s5i5/18d95033/s5i5.py", "341f85bac1fb5ca7a3315a367405c3c993a57ce4dd0ff01cc8aaa72bf6c6ed34"),
    "sb26": ("environment_files/sb26/7fbdac44/sb26.py", "dbb4877853a8d30f84e28d26f4a3d6ad7d2d1018602e82ccb5a62284043984f3"),
    "sc25": ("environment_files/sc25/635fd71a/sc25.py", "871a507cf340d1e4a3271205a71bb5433f20c2b0e427f7ce40da47060cdd4392"),
    "sk48": ("environment_files/sk48/d8078629/sk48.py", "b6ca05eed7ba81fe5a6a26c58410fa7daddf2039d3d1c42f600aaf9f6f0111e3"),
    "sp80": ("environment_files/sp80/589a99af/sp80.py", "2306d62311a3cc027250217d86cb4e012898edc7647a7c291ade23f36c9b9715"),
    "su15": ("environment_files/su15/1944f8ab/su15.py", "a5f91f7c963d6ca6447dae0ab21342b48a3f511601c40dfa8e972bdc59b4651e"),
    "tn36": ("environment_files/tn36/ef4dde99/tn36.py", "43c30052cdeb230017eb1ec23877050d96297756b86071417868ec61138c9fb7"),
    "tr87": ("environment_files/tr87/cd924810/tr87.py", "3274657a6499af5c7a52390d58a5c5441497ed70b84388e418fdfa6c9601e305"),
    "tu93": ("environment_files/tu93/0768757b/tu93.py", "80e41888f9f7b1a0c03e02c0aff3814e0fd68eb5b35ef22bb3649c87fc60a23f"),
    "vc33": ("environment_files/vc33/5430563c/vc33.py", "8afa9f55054b2a2460b6c44ec5c66e120f6d6f4c2289e24d50832a68d1e3fcfd"),
    "wa30": ("environment_files/wa30/ee6fef47/wa30.py", "0c8b63caee2d092d6c0750fd74e417de682d09973b549046394dadf0437b1477"),
}
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
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
JOURNAL_TERMINAL_OUTCOMES = {
    "CLOSED_CONFIRMED_PASS",
    "CLOSED_CONFIRMED_FAIL",
    "CLOSE_OUTCOME_AMBIGUOUS",
    "OPEN_OUTCOME_AMBIGUOUS",
}
SCORECARD_OPAQUE_FIELDS = {
    "schema",
    "gkm_operation_id",
    "mode",
    "release_receipt_sha256",
    "canonical_tree_sha256",
    "source_revision",
}
RUN_JOURNAL_BINDING_FIELDS = {
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
CLOSED_AGGREGATE_FIELDS = {
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
EXPECTED_TAG_SCORE_IDS = {"click", "keyboard", "keyboard_click"}
SCORECARD_STATES = {"NOT_PLAYED", "NOT_FINISHED", "WIN", "GAME_OVER"}


def read_checkpoint_bytes(path: Path, *, label: str) -> bytes:
    """Read one bounded checkpoint without accepting path/content races."""
    target = Path(path)
    metadata = target.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size > MAX_CHECKPOINT_BYTES
    ):
        raise ValueError(f"{label} is not a bounded, single-link regular file")
    descriptor = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino, opened.st_size)
            != (metadata.st_dev, metadata.st_ino, metadata.st_size)
            or opened.st_mtime_ns != metadata.st_mtime_ns
            or opened.st_ctime_ns != metadata.st_ctime_ns
        ):
            raise ValueError(f"{label} changed during secure open")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"{label} ended during secure read")
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            (after.st_dev, after.st_ino, after.st_size)
            != (opened.st_dev, opened.st_ino, opened.st_size)
            or after.st_mtime_ns != opened.st_mtime_ns
            or after.st_ctime_ns != opened.st_ctime_ns
        ):
            raise ValueError(f"{label} changed during secure read")
    finally:
        os.close(descriptor)
    return raw


def sha256_file(path: Path) -> str:
    return hashlib.sha256(
        read_checkpoint_bytes(path, label=f"checkpoint {Path(path).name}")
    ).hexdigest()


def path_lexists(path: Path) -> bool:
    try:
        Path(path).lstat()
    except FileNotFoundError:
        return False
    return True


def checkpoint_path(artifact_root: Path, game: str) -> Path:
    return artifact_root / f"{game}_legs" / "checkpoint.json"


def load_checkpoint(
    game: str, artifact_root: Path = DEFAULT_ARTIFACT_ROOT
) -> tuple[dict, str]:
    """Read and hash one regular checkpoint from the same immutable byte string."""
    path = checkpoint_path(artifact_root, game)
    raw = read_checkpoint_bytes(path, label=f"checkpoint {game}")

    def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, nested in pairs:
            if key in value:
                raise ValueError(f"checkpoint {game} contains duplicate key {key!r}")
            value[key] = nested
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"checkpoint {game} contains non-finite number {value}")

    try:
        value = json.loads(
            raw,
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"checkpoint is invalid JSON: {game}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint is not a JSON object: {game}")
    return value, hashlib.sha256(raw).hexdigest()


def checkpoint(game: str, artifact_root: Path = DEFAULT_ARTIFACT_ROOT) -> dict:
    """Backward-compatible value-only checkpoint reader."""
    return load_checkpoint(game, artifact_root)[0]


def parse_games(value: str, artifact_root: Path) -> list[str]:
    if value.strip().lower() == "all":
        games = sorted(
            path.parent.name.removesuffix("_legs")
            for path in artifact_root.glob("*_legs/checkpoint.json")
        )
    else:
        games = [game.strip() for game in value.split(",") if game.strip()]
    if not games:
        raise ValueError("no games selected")
    if len(games) != len(set(games)):
        raise ValueError("duplicate game in --games")
    if any(
        len(game) != 4
        or not game.isascii()
        or not game.isalnum()
        or game.lower() != game
        for game in games
    ):
        raise ValueError(f"invalid game set: {games!r}")
    return games


def release_binding(
    receipt_path: Path,
    games: list[str],
    checkpoints: dict[str, dict],
    checkpoint_hashes: dict[str, str],
) -> dict:
    """Bind the exact already-loaded endpoint bytes after full verification."""
    receipt, _ = load_receipt(receipt_path)
    receipt_sha256 = receipt_path.stem
    release_identity = receipt.get("release_identity")
    release_identity_revision = (
        release_identity.get("source_revision")
        if isinstance(release_identity, dict)
        else None
    )
    if (
        not isinstance(release_identity_revision, str)
        or REVISION_RE.fullmatch(release_identity_revision) is None
    ):
        raise ValueError("release receipt has no source revision")
    inventory = receipt.get("inventory")
    if not isinstance(inventory, dict):
        raise ValueError("release receipt has no authoritative inventory")
    claimed = receipt.get("claimed_inventory", inventory)
    evidence = receipt.get("evidence")
    if not isinstance(claimed, dict) or not isinstance(evidence, dict):
        raise ValueError("release receipt has no claimed inventory/evidence")
    if set(games) != set(inventory):
        raise ValueError(
            "scorecard game set differs from the receipt's authoritative set"
        )
    for game in games:
        expected = claimed.get(game)
        value = checkpoints[game]
        if value.get("game") != game or value.get("reached") != expected:
            raise ValueError(
                f"checkpoint frontier differs from release receipt: {game}"
            )
        rows = evidence.get(game)
        if (
            not isinstance(rows, list)
            or len(rows) != expected
            or not isinstance(rows[-1], dict)
        ):
            raise ValueError(f"release evidence is incomplete for {game}")
        actual = checkpoint_hashes.get(game)
        if rows[-1].get("checkpoint_sha256") != actual:
            raise ValueError(
                f"checkpoint bytes differ from release receipt: {game}"
            )
        endpoint_actions = rows[-1].get("action_count")
        if (
            not isinstance(endpoint_actions, int)
            or isinstance(endpoint_actions, bool)
            or endpoint_actions != len(value.get("final_path", []))
        ):
            raise ValueError(
                f"checkpoint action count differs from release receipt: {game}"
            )
    claimed_total = sum(int(value) for value in claimed.values())
    recorded_total = receipt.get(
        "claimed_level_count", receipt.get("authoritative_level_count")
    )
    if claimed_total != recorded_total:
        raise ValueError("release receipt claimed-level total is inconsistent")
    return {
        "binding_scope": "endpoint_checkpoint_bytes_only_after_full_gate",
        "receipt_sha256": receipt_sha256,
        "canonical_tree_sha256": receipt.get("canonical_tree_sha256"),
        "release_identity_source_revision": release_identity_revision,
        "claimed_inventory": claimed,
        "claimed_level_count": claimed_total,
        "authoritative_level_count": sum(
            int(value) for value in inventory.values()
        ),
    }


class RunJournalError(RuntimeError):
    """The durable remote-run journal is unsafe or internally inconsistent."""


def canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not canonical JSON") from exc


def strict_json_loads(raw: bytes, *, label: str) -> object:
    def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RunJournalError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise RunJournalError(f"{label} contains non-finite number {value}")

    try:
        return json.loads(
            raw,
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RunJournalError(f"{label} is invalid JSON") from exc


def verify_public_revision(revision: str) -> dict[str, str]:
    """Fail unless GitHub publicly resolves the exact scored GKM commit."""
    if REVISION_RE.fullmatch(revision) is None:
        raise ValueError("public source revision is invalid")
    request = urllib.request.Request(
        PUBLIC_COMMIT_ENDPOINT.format(revision=revision),
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "gkm-complete-replay/1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            raw = response.read(1024 * 1024 + 1)
    except (OSError, urllib.error.URLError) as exc:
        raise ValueError(
            "scored source revision is not publicly reachable on GitHub"
        ) from exc
    if len(raw) > 1024 * 1024:
        raise ValueError("GitHub commit response is unexpectedly large")
    value = strict_json_loads(raw, label="GitHub commit response")
    expected_html_url = (
        f"https://github.com/sashakolpakov/gkm/commit/{revision}"
    )
    if (
        not isinstance(value, dict)
        or value.get("sha") != revision
        or value.get("html_url") != expected_html_url
    ):
        raise ValueError("GitHub resolved a different source revision")
    return {"sha": revision, "html_url": expected_html_url}


def json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def _lexical_absolute(path: Path) -> Path:
    """Return an absolute path without resolving any symlink component."""
    return Path(os.path.abspath(os.fspath(path)))


def _open_physical_directory(path: Path, *, create: bool) -> int:
    """Open a directory by walking every component with O_NOFOLLOW."""
    target = _lexical_absolute(path)
    anchor = Path(target.anchor)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(anchor, flags)
    try:
        for component in target.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, 0o700, dir_fd=descriptor)
                os.fsync(descriptor)
                child = os.open(component, flags, dir_fd=descriptor)
            metadata = os.fstat(child)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(child)
                raise RunJournalError(
                    f"directory path component is not physical: {target}"
                )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _assert_open_directory_is_canonical(path: Path, descriptor: int) -> None:
    """Re-walk a lexical path and bind it to an already-open directory."""
    comparison = _open_physical_directory(path, create=False)
    try:
        expected = os.fstat(descriptor)
        actual = os.fstat(comparison)
        if (
            not stat.S_ISDIR(expected.st_mode)
            or not stat.S_ISDIR(actual.st_mode)
            or (actual.st_dev, actual.st_ino)
            != (expected.st_dev, expected.st_ino)
        ):
            raise RunJournalError("canonical output directory was replaced")
    finally:
        os.close(comparison)


def fsync_directory(path: Path) -> None:
    descriptor = _open_physical_directory(path, create=False)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def ensure_directory_durable(path: Path) -> None:
    descriptor = _open_physical_directory(path, create=True)
    os.close(descriptor)


def write_new_bytes(
    path: Path, payload: bytes, *, parent_descriptor: int | None = None
) -> None:
    path = _lexical_absolute(path)
    owns_parent = parent_descriptor is None
    parent = (
        _open_physical_directory(path.parent, create=True)
        if parent_descriptor is None
        else parent_descriptor
    )
    try:
        _assert_open_directory_is_canonical(path.parent, parent)
        descriptor = os.open(
            path.name,
            os.O_CREAT
            | os.O_EXCL
            | os.O_RDWR
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent,
        )
    except Exception:
        if owns_parent:
            os.close(parent)
        raise
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RunJournalError("new output is not a single-link regular file")
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise RunJournalError("new output write was incomplete")
            offset += written
        os.fsync(descriptor)
        os.fsync(parent)
        _assert_open_directory_is_canonical(path.parent, parent)
        path_metadata = os.stat(
            path.name, dir_fd=parent, follow_symlinks=False
        )
        before_read = os.fstat(descriptor)
        actual = CompleteRunJournal._pread_exact(descriptor, len(payload))
        after_read = os.fstat(descriptor)
        if (
            actual != payload
            or (path_metadata.st_dev, path_metadata.st_ino)
            != (metadata.st_dev, metadata.st_ino)
            or (before_read.st_dev, before_read.st_ino, before_read.st_size)
            != (metadata.st_dev, metadata.st_ino, len(payload))
            or (after_read.st_dev, after_read.st_ino, after_read.st_size)
            != (before_read.st_dev, before_read.st_ino, before_read.st_size)
            or after_read.st_mtime_ns != before_read.st_mtime_ns
            or after_read.st_ctime_ns != before_read.st_ctime_ns
        ):
            raise RunJournalError("new output changed during publication")
    except Exception:
        try:
            os.unlink(path.name, dir_fd=parent)
            os.fsync(parent)
        except OSError:
            pass
        raise
    finally:
        os.close(descriptor)
        if owns_parent:
            os.close(parent)


def write_new_json(path: Path, value: object) -> None:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True)
        + "\n"
    ).encode("utf-8")
    write_new_bytes(path, payload)


class ReservedReceipt:
    """O_EXCL reservation held from before INTENT through durable publication."""

    def __init__(self, path: Path):
        self.path = _lexical_absolute(path)
        self.parent_fd = _open_physical_directory(
            self.path.parent, create=True
        )
        flags = (
            os.O_CREAT
            | os.O_EXCL
            | os.O_RDWR
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(
                self.path.name, flags, 0o600, dir_fd=self.parent_fd
            )
        except Exception:
            os.close(self.parent_fd)
            raise
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise RunJournalError(
                    "reserved run receipt is not a single-link regular file"
                )
            os.fsync(descriptor)
            os.fsync(self.parent_fd)
            _assert_open_directory_is_canonical(
                self.path.parent, self.parent_fd
            )
        except Exception:
            os.close(descriptor)
            try:
                os.unlink(self.path.name, dir_fd=self.parent_fd)
                os.fsync(self.parent_fd)
            except OSError:
                pass
            os.close(self.parent_fd)
            raise
        self.fd = descriptor
        self.identity = (metadata.st_dev, metadata.st_ino)
        self.published = False

    def __enter__(self) -> "ReservedReceipt":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            os.close(self.fd)
        finally:
            os.close(self.parent_fd)

    def _assert_identity(
        self, *, expected_size: int = 0, expected_payload: bytes | None = None
    ) -> None:
        _assert_open_directory_is_canonical(self.path.parent, self.parent_fd)
        path_metadata = os.stat(
            self.path.name, dir_fd=self.parent_fd, follow_symlinks=False
        )
        fd_metadata = os.fstat(self.fd)
        if (
            not stat.S_ISREG(path_metadata.st_mode)
            or stat.S_ISLNK(path_metadata.st_mode)
            or path_metadata.st_nlink != 1
            or (path_metadata.st_dev, path_metadata.st_ino) != self.identity
            or (fd_metadata.st_dev, fd_metadata.st_ino) != self.identity
            or fd_metadata.st_size != expected_size
        ):
            raise RunJournalError("reserved run receipt path was replaced")
        if expected_payload is not None:
            actual = CompleteRunJournal._pread_exact(
                self.fd, fd_metadata.st_size
            )
            after_read = os.fstat(self.fd)
            after_path = os.stat(
                self.path.name,
                dir_fd=self.parent_fd,
                follow_symlinks=False,
            )
            if (
                actual != expected_payload
                or (after_read.st_dev, after_read.st_ino, after_read.st_size)
                != (fd_metadata.st_dev, fd_metadata.st_ino, fd_metadata.st_size)
                or after_read.st_mtime_ns != fd_metadata.st_mtime_ns
                or after_read.st_ctime_ns != fd_metadata.st_ctime_ns
                or (after_path.st_dev, after_path.st_ino, after_path.st_size)
                != (after_read.st_dev, after_read.st_ino, after_read.st_size)
                or after_path.st_mtime_ns != after_read.st_mtime_ns
                or after_path.st_ctime_ns != after_read.st_ctime_ns
            ):
                raise RunJournalError(
                    "reserved run receipt changed during publication"
                )

    def publish_json(self, value: object) -> None:
        if self.published:
            raise RunJournalError("reserved run receipt was already published")
        self._assert_identity()
        payload = (
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True)
            + "\n"
        ).encode("utf-8")
        offset = 0
        while offset < len(payload):
            written = os.write(self.fd, payload[offset:])
            if written <= 0:
                raise RunJournalError("reserved run receipt write was incomplete")
            offset += written
        os.fsync(self.fd)
        os.fsync(self.parent_fd)
        self._assert_identity(
            expected_size=len(payload), expected_payload=payload
        )
        self.published = True


def project_release_verification(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project verifier output onto a fixed, host-path-free schema."""
    complete_fields = (
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
    )
    partial_fields = (
        "status",
        "kind",
        "games",
        "claimed_levels",
        "authoritative_levels",
        "unclaimed_boundaries",
        "inventory_sha256",
        "claimed_inventory_sha256",
        "canonical_tree_sha256",
        "evidence_sha256",
        "verifier_sha256",
        "control_contract_sha256",
        "receipt_sha256",
        "verification_context_source_revision",
    )
    fields = partial_fields if "kind" in value else complete_fields
    missing = [field for field in fields if field not in value]
    if missing:
        raise ValueError(
            "release verifier output is missing safe summary fields: "
            + ", ".join(missing)
        )
    projected = {field: value[field] for field in fields}
    if projected.get("status") != "PASS":
        raise ValueError("release verifier summary did not pass")
    return projected


def complete_scorecard_tags(
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


def complete_scorecard_opaque(
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


def canonical_run_journal_path(receipt_sha256: str) -> Path:
    if SHA256_RE.fullmatch(receipt_sha256) is None:
        raise ValueError("release receipt hash cannot name a run journal")
    return RUN_JOURNAL_ROOT / f"{receipt_sha256}.jsonl"


def canonical_run_receipt_path(receipt_sha256: str, mode: str) -> Path:
    if SHA256_RE.fullmatch(receipt_sha256) is None:
        raise ValueError("release receipt hash cannot name a run receipt")
    if mode not in {"online", "competition"}:
        raise ValueError("remote-run mode cannot name a run receipt")
    return RUN_JOURNAL_ROOT / "receipts" / receipt_sha256 / f"{mode}.json"


def read_bounded_regular(path: Path, *, label: str) -> bytes:
    target = _lexical_absolute(path)
    parent = _open_physical_directory(target.parent, create=False)
    try:
        metadata = os.stat(
            target.name, dir_fd=parent, follow_symlinks=False
        )
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size > MAX_JOURNAL_BYTES
        ):
            raise RunJournalError(
                f"{label} must be a bounded, single-link regular file"
            )
        descriptor = os.open(
            target.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent,
        )
        try:
            _assert_open_directory_is_canonical(target.parent, parent)
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or opened.st_size != metadata.st_size
                or (opened.st_dev, opened.st_ino)
                != (metadata.st_dev, metadata.st_ino)
                or opened.st_mtime_ns != metadata.st_mtime_ns
                or opened.st_ctime_ns != metadata.st_ctime_ns
            ):
                raise RunJournalError(f"{label} changed during secure read")
            chunks: list[bytes] = []
            remaining = opened.st_size
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    raise RunJournalError(f"{label} ended during secure read")
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            after = os.fstat(descriptor)
            if (
                (after.st_dev, after.st_ino, after.st_size)
                != (opened.st_dev, opened.st_ino, opened.st_size)
                or after.st_mtime_ns != opened.st_mtime_ns
                or after.st_ctime_ns != opened.st_ctime_ns
            ):
                raise RunJournalError(f"{label} changed during secure read")
        finally:
            os.close(descriptor)
    finally:
        os.close(parent)
    if len(raw) > MAX_JOURNAL_BYTES:
        raise RunJournalError(f"{label} is unexpectedly large")
    return raw


def _git_output(arguments: list[str], *, max_bytes: int) -> bytes:
    environment = {
        "PATH": os.environ.get("PATH", os.defpath),
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
    }
    try:
        completed = subprocess.run(
            ["git", "-C", os.fspath(GKM), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("cannot inspect the scored Git revision") from exc
    if completed.returncode != 0 or len(completed.stdout) > max_bytes:
        raise ValueError("scored Git revision lacks a bounded runtime file")
    return completed.stdout


def verify_runtime_revision(
    revision: str, games: list[str]
) -> dict[str, Any]:
    """Bind every repo-local executable byte before local or remote replay."""
    if REVISION_RE.fullmatch(revision) is None:
        raise ValueError("runtime source revision is invalid")
    head = _git_output(
        ["rev-parse", "--verify", "HEAD^{commit}"], max_bytes=128
    ).decode("ascii", errors="strict").strip()
    if head != revision:
        raise ValueError("Git HEAD differs from the scored source revision")

    unknown_games = set(games) - set(RUNTIME_ENVIRONMENT_SOURCE_SHA256)
    if unknown_games:
        raise ValueError("runtime manifest does not cover every scored game")
    files_sha256: dict[str, str] = {}
    for relative in RUNTIME_TRACKED_PATHS:
        object_name = f"{revision}:{relative}"
        size_raw = _git_output(
            ["cat-file", "-s", object_name], max_bytes=64
        )
        try:
            size = int(size_raw.strip())
        except ValueError as exc:
            raise ValueError("Git returned an invalid runtime blob size") from exc
        if not 0 <= size <= MAX_JOURNAL_BYTES:
            raise ValueError("scored runtime source is unexpectedly large")
        expected = _git_output(["show", object_name], max_bytes=size)
        actual = read_bounded_regular(
            GKM / relative, label=f"runtime source {relative}"
        )
        if len(expected) != size or actual != expected:
            raise ValueError(
                f"working runtime source differs from {revision}: {relative}"
            )
        files_sha256[relative] = hashlib.sha256(actual).hexdigest()

    for game in games:
        relative, expected_sha256 = RUNTIME_ENVIRONMENT_SOURCE_SHA256[game]
        actual = read_bounded_regular(
            GKM / relative, label=f"runtime environment source {game}"
        )
        actual_sha256 = hashlib.sha256(actual).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"runtime environment source hash mismatch: {game}"
            )
        files_sha256[relative] = actual_sha256
    return {
        "source_revision": revision,
        "files_sha256": files_sha256,
        "manifest_sha256": json_sha256(files_sha256),
    }


def load_arc_api_key() -> str | None:
    """Read only ARC_API_KEY, after local segmentation has completed."""
    inherited = os.environ.get("ARC_API_KEY")
    if inherited:
        return inherited
    env_path = GKM / ".env"
    if not path_lexists(env_path):
        return None
    raw = read_checkpoint_bytes(env_path, label="ARC_API_KEY environment file")
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError("repo .env is not UTF-8") from exc
    values: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        if key.strip() == "ARC_API_KEY":
            values.append(value.strip())
    if len(values) > 1:
        raise ValueError("repo .env contains duplicate ARC_API_KEY entries")
    return values[0] if values and values[0] else None


def journal_id_for_release(receipt_sha256: str) -> str:
    return hashlib.sha256(
        f"gkm-complete-remote-journal-v1:{receipt_sha256}".encode("ascii")
    ).hexdigest()


def _parse_journal_timestamp(value: object) -> dt.datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise RunJournalError("journal timestamp is not UTC")
    try:
        parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise RunJournalError("journal timestamp is invalid") from exc
    if parsed.utcoffset() != dt.timedelta(0):
        raise RunJournalError("journal timestamp is not UTC")
    return parsed


def parse_run_journal(
    raw: bytes, *, expected_journal_id: str | None = None
) -> list[dict[str, Any]]:
    """Validate canonical JSONL, its global hash chain, and state transitions."""
    if len(raw) > MAX_JOURNAL_BYTES:
        raise RunJournalError("run journal is unexpectedly large")
    if raw and not raw.endswith(b"\n"):
        raise RunJournalError("run journal has a truncated terminal line")
    records: list[dict[str, Any]] = []
    states: dict[str, str] = {}
    card_ids: set[str] = set()
    previous_digest: str | None = None
    previous_timestamp: dt.datetime | None = None
    for index, line in enumerate(raw.splitlines(), start=1):
        event = strict_json_loads(line, label="run journal event")
        if not isinstance(event, dict) or set(event) != JOURNAL_EVENT_FIELDS:
            raise RunJournalError("run journal event schema mismatch")
        if canonical_json(event) != line:
            raise RunJournalError("run journal event is not canonical JSON")
        journal_id = event.get("journal_id")
        if (
            not isinstance(journal_id, str)
            or SHA256_RE.fullmatch(journal_id) is None
            or (
                expected_journal_id is not None
                and journal_id != expected_journal_id
            )
        ):
            raise RunJournalError("run journal identity mismatch")
        if (
            not isinstance(event.get("schema"), int)
            or isinstance(event.get("schema"), bool)
            or event.get("schema") != 1
            or not isinstance(event.get("sequence"), int)
            or isinstance(event.get("sequence"), bool)
            or event.get("sequence") != index
        ):
            raise RunJournalError("run journal sequence mismatch")
        if event.get("previous_event_sha256") != previous_digest:
            raise RunJournalError("run journal hash chain mismatch")
        timestamp = _parse_journal_timestamp(event.get("timestamp_utc"))
        if previous_timestamp is not None and timestamp < previous_timestamp:
            raise RunJournalError("run journal timestamps are not monotonic")
        previous_timestamp = timestamp
        operation_id = event.get("operation_id")
        if (
            not isinstance(operation_id, str)
            or SHA256_RE.fullmatch(operation_id) is None
        ):
            raise RunJournalError("run journal operation ID is invalid")
        kind = event.get("event")
        payload = event.get("payload")
        if not isinstance(payload, dict):
            raise RunJournalError("run journal event payload is invalid")
        state = states.get(operation_id)
        if kind == "INTENT":
            if state is not None or set(payload) != JOURNAL_INTENT_FIELDS:
                raise RunJournalError("run journal has a duplicate/invalid intent")
            if payload.get("mode") not in {"online", "competition"}:
                raise RunJournalError("run journal intent mode is invalid")
            opaque = payload.get("opaque")
            if (
                not isinstance(opaque, dict)
                or set(opaque) != SCORECARD_OPAQUE_FIELDS
                or not isinstance(opaque.get("schema"), int)
                or isinstance(opaque.get("schema"), bool)
                or opaque.get("schema") != 1
            ):
                raise RunJournalError("run journal intent opaque schema is invalid")
            states[operation_id] = "INTENT"
        elif kind == "OPENED":
            if state != "INTENT" or set(payload) != JOURNAL_OPENED_FIELDS:
                raise RunJournalError("run journal OPENED transition is invalid")
            card_id = payload.get("card_id")
            if (
                not isinstance(card_id, str)
                or UUID_RE.fullmatch(card_id) is None
                or card_id in card_ids
                or payload.get("scorecard_url")
                != f"https://arcprize.org/scorecards/{card_id}"
            ):
                raise RunJournalError("run journal OPENED card is invalid")
            card_ids.add(card_id)
            states[operation_id] = "OPENED"
        elif kind == "TERMINAL":
            if set(payload) != JOURNAL_TERMINAL_FIELDS:
                raise RunJournalError("run journal TERMINAL schema is invalid")
            outcome = payload.get("outcome")
            card_id = payload.get("card_id")
            if (
                outcome not in JOURNAL_TERMINAL_OUTCOMES
                or not isinstance(payload.get("receipt_core_sha256"), str)
                or SHA256_RE.fullmatch(payload["receipt_core_sha256"]) is None
            ):
                raise RunJournalError("run journal TERMINAL outcome is invalid")
            if outcome == "OPEN_OUTCOME_AMBIGUOUS":
                if state != "INTENT" or card_id is not None:
                    raise RunJournalError(
                        "open ambiguity must terminate an unmaterialized intent"
                    )
            elif (
                state != "OPENED"
                or not isinstance(card_id, str)
                or UUID_RE.fullmatch(card_id) is None
            ):
                raise RunJournalError("run journal TERMINAL card is invalid")
            states[operation_id] = str(outcome)
        else:
            raise RunJournalError("run journal event kind is invalid")
        digest = hashlib.sha256(line).hexdigest()
        records.append({"event": event, "sha256": digest})
        previous_digest = digest
    return records


def journal_operation_states(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for record in records:
        event = record["event"]
        operation_id = event["operation_id"]
        state = states.setdefault(operation_id, {})
        state[event["event"].lower()] = record
    return states


class CompleteRunJournal:
    """Locked append-only journal for one complete release/API identity."""

    def __init__(
        self,
        path: Path,
        *,
        journal_id: str,
        snapshot_directory: Path | None = None,
    ):
        self.path = _lexical_absolute(path)
        self.snapshot_directory = (
            _lexical_absolute(snapshot_directory)
            if snapshot_directory is not None
            else None
        )
        self.journal_id = journal_id
        self.fd: int | None = None
        self.parent_fd: int | None = None
        self.snapshot_parent_fd: int | None = None
        self.records: list[dict[str, Any]] = []
        self.raw = b""
        self.identity: tuple[int, int] | None = None

    @staticmethod
    def _pread_exact(descriptor: int, size: int) -> bytes:
        chunks: list[bytes] = []
        offset = 0
        while offset < size:
            chunk = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
            if not chunk:
                raise RunJournalError("run journal ended during locked read")
            chunks.append(chunk)
            offset += len(chunk)
        return b"".join(chunks)

    def __enter__(self) -> "CompleteRunJournal":
        parent = _open_physical_directory(self.path.parent, create=True)
        try:
            existing = os.stat(
                self.path.name, dir_fd=parent, follow_symlinks=False
            )
        except FileNotFoundError:
            existed = False
        else:
            existed = True
            if stat.S_ISLNK(existing.st_mode):
                os.close(parent)
                raise RunJournalError("run journal path is a symlink")
        flags = (
            os.O_RDWR
            | os.O_APPEND
            | os.O_CREAT
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(
                self.path.name, flags, 0o600, dir_fd=parent
            )
        except Exception:
            os.close(parent)
            raise
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RunJournalError(
                    "another complete remote run holds the journal lock"
                ) from exc
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_size > MAX_JOURNAL_BYTES
            ):
                raise RunJournalError(
                    "run journal must be a bounded, single-link regular file"
                )
            if not existed:
                os.fsync(parent)
            _assert_open_directory_is_canonical(self.path.parent, parent)
            self.raw = self._pread_exact(descriptor, metadata.st_size)
            after_read = os.fstat(descriptor)
            if (
                (after_read.st_dev, after_read.st_ino, after_read.st_size)
                != (metadata.st_dev, metadata.st_ino, metadata.st_size)
                or after_read.st_mtime_ns != metadata.st_mtime_ns
                or after_read.st_ctime_ns != metadata.st_ctime_ns
            ):
                raise RunJournalError("run journal changed during locked read")
            self.records = parse_run_journal(
                self.raw, expected_journal_id=self.journal_id
            )
            self.identity = (metadata.st_dev, metadata.st_ino)
            if self.snapshot_directory is not None:
                self.snapshot_parent_fd = _open_physical_directory(
                    self.snapshot_directory, create=True
                )
                _assert_open_directory_is_canonical(
                    self.snapshot_directory, self.snapshot_parent_fd
                )
        except Exception:
            if self.snapshot_parent_fd is not None:
                os.close(self.snapshot_parent_fd)
                self.snapshot_parent_fd = None
            os.close(descriptor)
            os.close(parent)
            raise
        self.fd = descriptor
        self.parent_fd = parent
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.fd is not None:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            finally:
                os.close(self.fd)
                self.fd = None
        if self.parent_fd is not None:
            os.close(self.parent_fd)
            self.parent_fd = None
        if self.snapshot_parent_fd is not None:
            os.close(self.snapshot_parent_fd)
            self.snapshot_parent_fd = None

    def _assert_path_identity(self, *, expected_raw: bytes | None = None) -> None:
        if self.fd is None or self.parent_fd is None or self.identity is None:
            raise RunJournalError("run journal is not locked")
        try:
            _assert_open_directory_is_canonical(
                self.path.parent, self.parent_fd
            )
            path_metadata = os.stat(
                self.path.name,
                dir_fd=self.parent_fd,
                follow_symlinks=False,
            )
            fd_metadata = os.fstat(self.fd)
        except OSError as exc:
            raise RunJournalError("run journal path disappeared") from exc
        expected = self.raw if expected_raw is None else expected_raw
        if (
            stat.S_ISLNK(path_metadata.st_mode)
            or not stat.S_ISREG(path_metadata.st_mode)
            or path_metadata.st_nlink != 1
            or (path_metadata.st_dev, path_metadata.st_ino) != self.identity
            or (fd_metadata.st_dev, fd_metadata.st_ino) != self.identity
            or fd_metadata.st_size != len(expected)
        ):
            raise RunJournalError("run journal path was replaced")
        actual = self._pread_exact(self.fd, fd_metadata.st_size)
        after_read = os.fstat(self.fd)
        if (
            actual != expected
            or (after_read.st_dev, after_read.st_ino, after_read.st_size)
            != (fd_metadata.st_dev, fd_metadata.st_ino, fd_metadata.st_size)
            or after_read.st_mtime_ns != fd_metadata.st_mtime_ns
            or after_read.st_ctime_ns != fd_metadata.st_ctime_ns
        ):
            raise RunJournalError("run journal changed outside the locked writer")

    def assert_can_start(self, mode: str) -> None:
        self._assert_path_identity()
        states = journal_operation_states(self.records)
        for operation_id, state in states.items():
            intent = state["intent"]["event"]
            terminal = state.get("terminal")
            if terminal is None or terminal["event"]["payload"]["outcome"] == (
                "OPEN_OUTCOME_AMBIGUOUS"
            ):
                phase = (
                    "OPEN_OUTCOME_AMBIGUOUS"
                    if "opened" not in state
                    else "OPENED_OUTCOME_UNRESOLVED"
                )
                raise RunJournalError(
                    f"unresolved remote-run intent {operation_id}: {phase}; "
                    "human/provider reconciliation is required"
                )
            if intent["payload"]["mode"] == mode:
                raise RunJournalError(
                    f"journal already contains a {mode} open attempt; "
                    "automatic retry is forbidden"
                )
        if mode == "competition":
            online_states = [
                state
                for state in states.values()
                if state["intent"]["event"]["payload"]["mode"] == "online"
            ]
            if len(online_states) != 1:
                raise RunJournalError(
                    "Competition requires exactly one durably published ONLINE "
                    "shakedown receipt"
                )
            self._validate_prior_online_receipt(online_states[0])

    def _validate_prior_online_receipt(
        self, state: Mapping[str, Mapping[str, Any]]
    ) -> None:
        intent = state.get("intent")
        opened = state.get("opened")
        terminal = state.get("terminal")
        if intent is None or opened is None or terminal is None:
            raise RunJournalError("ONLINE shakedown journal is not terminal")
        receipt_sha256 = intent["event"]["payload"].get(
            "release_receipt_sha256"
        )
        if (
            not isinstance(receipt_sha256, str)
            or SHA256_RE.fullmatch(receipt_sha256) is None
        ):
            raise RunJournalError("ONLINE shakedown release binding is invalid")
        receipt_path = canonical_run_receipt_path(receipt_sha256, "online")
        try:
            receipt_raw = read_bounded_regular(
                receipt_path, label="canonical ONLINE run receipt"
            )
            receipt = strict_json_loads(
                receipt_raw, label="canonical ONLINE run receipt"
            )
        except OSError as exc:
            raise RunJournalError(
                "canonical ONLINE run receipt is absent or invalid; "
                "Competition is blocked"
            ) from exc
        if not isinstance(receipt, dict):
            raise RunJournalError("canonical ONLINE run receipt is not an object")
        journal = receipt.get("run_journal")
        receipt_opaque = receipt.get("scorecard_opaque")
        if (
            set(receipt) != RUN_RECEIPT_FIELDS
            or not isinstance(receipt.get("schema"), int)
            or isinstance(receipt.get("schema"), bool)
            or receipt.get("schema") != 2
            or receipt.get("mode") != "online"
            or receipt.get("status") != "PASS"
            or receipt.get("scorecard_open")
            != {"status": "confirmed", "error_type": None}
            or receipt.get("scorecard_close")
            != {"status": "confirmed", "error_type": None}
            or not isinstance(journal, dict)
            or set(journal) != RUN_JOURNAL_BINDING_FIELDS
            or not isinstance(journal.get("schema"), int)
            or isinstance(journal.get("schema"), bool)
            or journal.get("schema") != 1
            or any(
                not isinstance(journal.get(field), int)
                or isinstance(journal.get(field), bool)
                for field in (
                    "intent_sequence",
                    "opened_sequence",
                    "terminal_sequence",
                )
            )
            or journal.get("journal_id") != self.journal_id
            or journal.get("live_journal") != logical_path(self.path)
            or journal.get("operation_id")
            != intent["event"]["operation_id"]
            or journal.get("intent_sequence")
            != intent["event"]["sequence"]
            or journal.get("intent_event_sha256") != intent["sha256"]
            or journal.get("opened_sequence")
            != opened["event"]["sequence"]
            or journal.get("opened_event_sha256") != opened["sha256"]
            or journal.get("terminal_sequence")
            != terminal["event"]["sequence"]
            or journal.get("terminal_event_sha256") != terminal["sha256"]
            or journal.get("terminal_outcome") != "CLOSED_CONFIRMED_PASS"
            or terminal["event"]["payload"].get("outcome")
            != "CLOSED_CONFIRMED_PASS"
            or receipt.get("scorecard_id")
            != opened["event"]["payload"].get("card_id")
            or receipt.get("scorecard_url")
            != opened["event"]["payload"].get("scorecard_url")
            or receipt.get("scorecard_id")
            != terminal["event"]["payload"].get("card_id")
            or not isinstance(receipt.get("release_binding"), dict)
            or receipt["release_binding"].get("receipt_sha256")
            != receipt_sha256
            or not isinstance(receipt.get("release_verification"), dict)
            or receipt["release_verification"].get("receipt_sha256")
            != receipt_sha256
            or receipt.get("source_revision")
            != intent["event"]["payload"].get("source_revision")
            or receipt.get("source_url")
            != intent["event"]["payload"].get("source_url")
            or receipt.get("arc_agi_toolkit_version")
            != intent["event"]["payload"].get("arc_agi_toolkit_version")
            or receipt.get("scorecard_tags")
            != intent["event"]["payload"].get("tags")
            or json_sha256(receipt.get("checkpoint_sha256"))
            != intent["event"]["payload"].get("checkpoint_sha256_digest")
            or json_sha256(receipt.get("command"))
            != intent["event"]["payload"].get("command_sha256")
            or not isinstance(receipt_opaque, dict)
            or set(receipt_opaque) != SCORECARD_OPAQUE_FIELDS
            or not isinstance(receipt_opaque.get("schema"), int)
            or isinstance(receipt_opaque.get("schema"), bool)
            or receipt_opaque.get("schema") != 1
            or json_sha256(receipt_opaque)
            != journal.get("opaque_sha256")
        ):
            raise RunJournalError(
                "canonical ONLINE run receipt does not bind a successful "
                "journal chain"
            )
        receipt_core = dict(receipt)
        receipt_core.pop("run_journal", None)
        receipt_core_sha256 = json_sha256(receipt_core)
        if (
            journal.get("receipt_core_sha256") != receipt_core_sha256
            or terminal["event"]["payload"].get("receipt_core_sha256")
            != receipt_core_sha256
        ):
            raise RunJournalError(
                "canonical ONLINE run receipt core hash is invalid"
            )
        snapshot_sha256 = journal.get("snapshot_sha256")
        if (
            not isinstance(snapshot_sha256, str)
            or SHA256_RE.fullmatch(snapshot_sha256) is None
        ):
            raise RunJournalError("ONLINE journal snapshot hash is invalid")
        snapshot_path = (
            RUN_JOURNAL_ROOT / "snapshots" / f"{snapshot_sha256}.jsonl"
        )
        try:
            snapshot_raw = read_bounded_regular(
                snapshot_path, label="canonical ONLINE journal snapshot"
            )
        except OSError as exc:
            raise RunJournalError(
                "canonical ONLINE journal snapshot is absent; Competition is "
                "blocked"
            ) from exc
        if (
            hashlib.sha256(snapshot_raw).hexdigest() != snapshot_sha256
            or journal.get("snapshot") != logical_path(snapshot_path)
            or snapshot_raw != self.raw
        ):
            raise RunJournalError(
                "canonical ONLINE journal snapshot differs from the live prefix"
            )

    def append(
        self,
        *,
        kind: str,
        operation_id: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self.fd is None:
            raise RunJournalError("run journal is not locked")
        self._assert_path_identity()
        previous = self.records[-1]["sha256"] if self.records else None
        event = {
            "schema": 1,
            "journal_id": self.journal_id,
            "sequence": len(self.records) + 1,
            "previous_event_sha256": previous,
            "event": kind,
            "operation_id": operation_id,
            "timestamp_utc": utc_now(),
            "payload": dict(payload),
        }
        line = canonical_json(event) + b"\n"
        written = os.write(self.fd, line)
        if written != len(line):
            raise RunJournalError("run journal append was incomplete")
        os.fsync(self.fd)
        assert self.parent_fd is not None
        os.fsync(self.parent_fd)
        expected_raw = self.raw + line
        self._assert_path_identity(expected_raw=expected_raw)
        self.raw = expected_raw
        parsed = parse_run_journal(
            self.raw, expected_journal_id=self.journal_id
        )
        self.records = parsed
        return parsed[-1]

    def write_snapshot(self) -> tuple[Path, str]:
        self._assert_path_identity()
        snapshot_sha256 = hashlib.sha256(self.raw).hexdigest()
        snapshot_directory = (
            self.snapshot_directory
            if self.snapshot_directory is not None
            else _lexical_absolute(RUN_JOURNAL_ROOT / "snapshots")
        )
        snapshot_path = snapshot_directory / f"{snapshot_sha256}.jsonl"
        write_new_bytes(
            snapshot_path,
            self.raw,
            parent_descriptor=self.snapshot_parent_fd,
        )
        return snapshot_path, snapshot_sha256


def toolkit_version() -> str:
    """Return the installed official ARC toolkit version."""
    try:
        return importlib.metadata.version("arc-agi")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ValueError("the official arc-agi toolkit is not installed") from exc


def logical_path(path: Path) -> str:
    """Describe a path without publishing host-specific absolute prefixes."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(GKM.resolve()).as_posix()
    except ValueError:
        return f"<external>/{resolved.name}"


def command_identity(
    *,
    args: argparse.Namespace,
    games: list[str],
    artifact_root: Path,
) -> dict:
    """Return the secret-free logical invocation recorded in schema-2."""
    return {
        "entrypoint": "arc/crack_lab/replay_scorecard.py",
        "mode": args.mode,
        "games": games,
        "artifact_root": logical_path(artifact_root),
        "release_receipt": (
            logical_path(args.release_receipt)
            if args.release_receipt is not None
            else None
        ),
        "expected_claimed_levels": args.expected_claimed_levels,
        "preflight_only": args.preflight_only,
        "source_url": args.source_url,
        "source_revision": args.source_revision,
        "tags": [tag.strip() for tag in args.tags.split(",") if tag.strip()],
    }


def finalize_journal_receipt(
    *,
    journal: CompleteRunJournal,
    receipt_core: dict[str, Any],
    operation_id: str,
    opaque: Mapping[str, Any],
    intent_record: Mapping[str, Any],
    opened_record: Mapping[str, Any] | None,
    terminal_outcome: str,
) -> dict[str, Any]:
    """Durably terminate, snapshot, and bind one remote-run receipt core."""
    receipt_core_sha256 = json_sha256(receipt_core)
    terminal_record = journal.append(
        kind="TERMINAL",
        operation_id=operation_id,
        payload={
            "outcome": terminal_outcome,
            "card_id": receipt_core.get("scorecard_id"),
            "receipt_core_sha256": receipt_core_sha256,
        },
    )
    snapshot_path, snapshot_sha256 = journal.write_snapshot()
    return {
        **receipt_core,
        "run_journal": {
            "schema": 1,
            "journal_id": journal.journal_id,
            "live_journal": logical_path(journal.path),
            "snapshot": logical_path(snapshot_path),
            "snapshot_sha256": snapshot_sha256,
            "operation_id": operation_id,
            "opaque_sha256": json_sha256(opaque),
            "intent_sequence": intent_record["event"]["sequence"],
            "intent_event_sha256": intent_record["sha256"],
            "opened_sequence": (
                opened_record["event"]["sequence"]
                if opened_record is not None
                else None
            ),
            "opened_event_sha256": (
                opened_record["sha256"]
                if opened_record is not None
                else None
            ),
            "terminal_sequence": terminal_record["event"]["sequence"],
            "terminal_event_sha256": terminal_record["sha256"],
            "terminal_outcome": terminal_outcome,
            "receipt_core_sha256": receipt_core_sha256,
        },
    }


def immutable_source_url(source_url: str, revision: str) -> bool:
    """Recognize the exact public GKM revision URL used for final cards."""
    return (
        REVISION_RE.fullmatch(revision) is not None
        and source_url
        == f"https://github.com/sashakolpakov/gkm/tree/{revision}"
    )


def _environment_depth(value: object) -> int | None:
    if not isinstance(value, dict):
        return None
    depth = value.get("levels_completed")
    if isinstance(depth, int) and not isinstance(depth, bool):
        return depth
    runs = value.get("runs")
    if not isinstance(runs, list) or not runs:
        return None
    depths = [
        run.get("levels_completed")
        for run in runs
        if isinstance(run, dict)
        and isinstance(run.get("levels_completed"), int)
        and not isinstance(run.get("levels_completed"), bool)
    ]
    return max(depths) if depths else None


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float("-inf") < float(value) < float("inf")
    )


def _validate_complete_environment_score(
    environment: Mapping[str, Any],
    *,
    game: str,
    target: int,
) -> None:
    if set(environment) != ENVIRONMENT_SCORE_LIST_FIELDS:
        raise ValueError(f"closed scorecard environment schema mismatch: {game}")
    environment_id = environment.get("id")
    if (
        not isinstance(environment_id, str)
        or re.fullmatch(rf"{re.escape(game)}-[0-9a-f]{{8}}", environment_id)
        is None
    ):
        raise ValueError(f"closed scorecard environment ID mismatch: {game}")
    runs = environment.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError(
            f"closed scorecard must contain provider run history: {game}"
        )
    winners: list[int] = []
    for index, run in enumerate(runs):
        if not isinstance(run, dict) or set(run) != ENVIRONMENT_SCORE_FIELDS:
            raise ValueError(f"closed scorecard nested run schema mismatch: {game}")
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
            or run.get("message") is not None
        ):
            raise ValueError(
                f"closed scorecard nested run accounting is invalid: {game}"
            )
        for field, integral in (
            ("level_scores", False),
            ("level_actions", True),
            ("level_baseline_actions", True),
        ):
            values = run.get(field)
            if (
                not isinstance(values, list)
                or len(values) != target
                or any(
                    (not _is_int(value) if integral else not _number(value))
                    for value in values
                )
                or (integral and any(value < 0 for value in values))
            ):
                raise ValueError(
                    f"closed scorecard level accounting mismatch: {game}"
                )
        if (
            sum(run["level_actions"]) != run["actions"]
            or any(not 0 <= float(value) <= 115 for value in run["level_scores"])
        ):
            raise ValueError(
                f"closed scorecard level accounting mismatch: {game}"
            )
        won = (
            run.get("levels_completed") == target
            and run.get("state") == "WIN"
            and run.get("completed") is True
        )
        if won:
            winners.append(index)
        elif run.get("state") == "WIN" or run.get("completed") is True:
            raise ValueError(
                f"closed scorecard terminal state is inconsistent: {game}"
            )
    if winners != [len(runs) - 1]:
        raise ValueError(
            f"closed scorecard must have one terminal target WIN run: {game}"
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
        raise ValueError(f"closed scorecard environment aggregate mismatch: {game}")


def _validate_closed_tag_scores(value: object) -> None:
    if not isinstance(value, list):
        raise ValueError("closed scorecard tag scores are missing")
    ids: set[str] = set()
    for score in value:
        score_id = score.get("id") if isinstance(score, dict) else None
        if (
            not isinstance(score, dict)
            or set(score) != ENVIRONMENT_SCORE_FIELDS
            or score_id not in EXPECTED_TAG_SCORE_IDS
            or score_id in ids
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
            or score.get("level_scores") is not None
            or score.get("level_actions") is not None
            or score.get("level_baseline_actions") is not None
            or not _is_int(score.get("number_of_levels"))
            or score["number_of_levels"] < 0
            or not _is_int(score.get("number_of_environments"))
            or score["number_of_environments"] <= 0
            or score.get("message") is not None
        ):
            raise ValueError("closed scorecard tag aggregate schema mismatch")
        ids.add(str(score_id))
    if ids != EXPECTED_TAG_SCORE_IDS:
        raise ValueError("closed scorecard tag aggregate set mismatch")


def validate_closed_scorecard(
    aggregate: object,
    *,
    mode: str,
    card_id: str,
    source_url: str,
    games: list[str],
    plan: dict[str, dict],
    scorecard_tags: list[str],
    scorecard_opaque: Mapping[str, Any] | None,
) -> dict:
    """Fail unless the returned closed card matches the frozen replay plan."""
    if aggregate is None:
        raise ValueError("closed scorecard aggregate is absent")
    if not isinstance(aggregate, dict) or set(aggregate) != CLOSED_AGGREGATE_FIELDS:
        raise ValueError("closed scorecard aggregate schema mismatch")
    expected_competition = mode == "competition"
    if (
        aggregate.get("card_id") != card_id
        or aggregate.get("source_url") != source_url
        or aggregate.get("tags") != scorecard_tags
        or aggregate.get("opaque") != scorecard_opaque
        or aggregate.get("competition_mode") is not expected_competition
    ):
        raise ValueError("closed scorecard provenance differs from the run plan")

    environments = aggregate.get("environments")
    if not isinstance(environments, list) or len(environments) != len(games):
        raise ValueError("closed scorecard has no environment accounting")
    by_game: dict[str, dict] = {}
    for environment in environments:
        if not isinstance(environment, dict):
            raise ValueError("closed scorecard has malformed environment accounting")
        environment_id = environment.get("id")
        if not isinstance(environment_id, str):
            raise ValueError("closed scorecard environment has no game ID")
        matches = [
            game
            for game in games
            if re.fullmatch(
                rf"{re.escape(game)}-[0-9a-f]{{8}}", environment_id
            )
            is not None
        ]
        if len(matches) != 1 or matches[0] in by_game:
            raise ValueError("closed scorecard game accounting is ambiguous")
        by_game[matches[0]] = environment
    if set(by_game) != set(games):
        raise ValueError("closed scorecard game set differs from the frozen plan")

    for game in games:
        _validate_complete_environment_score(
            by_game[game], game=game, target=plan[game]["reached"]
        )
    expected_levels = sum(int(plan[game]["reached"]) for game in games)
    total_actions = sum(int(row["actions"]) for row in environments)
    if (
        aggregate.get("total_levels_completed") != expected_levels
        or aggregate.get("total_levels") != expected_levels
        or aggregate.get("total_environments") != len(games)
        or aggregate.get("total_environments_completed") != len(games)
        or aggregate.get("total_actions") != total_actions
    ):
        raise ValueError("closed scorecard totals differ from the frozen plan")
    _validate_closed_tag_scores(aggregate.get("tags_scores"))
    score = aggregate.get("score")
    if not _number(score) or not 0 <= float(score) <= 100:
        raise ValueError("closed scorecard has no valid aggregate score")
    return aggregate


def decode_action(action) -> tuple[int, dict | None]:
    """Decode scalar keys and canonical ``[6, x, y]`` replay tokens."""
    if isinstance(action, (list, tuple)):
        if (
            len(action) != 3
            or action[0] != 6
            or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in action
            )
            or not all(0 <= value < 64 for value in action[1:])
        ):
            raise ValueError(f"invalid compound replay action: {action!r}")
        return 6, {"x": action[1], "y": action[2]}
    if not isinstance(action, int) or isinstance(action, bool):
        raise ValueError(f"invalid replay action: {action!r}")
    action_id = action
    if not 1 <= action_id <= 7 or action_id == 6:
        raise ValueError(f"invalid replay action: {action!r}")
    return action_id, None


def level_segments(game: str, actions) -> list:
    """Split the flat recorded path into per-level action segments by replaying
    it on the LOCAL engine (offline, ~2000 fps). Level boundaries let the remote
    replay recover from transient API failures: in competition mode RESET is a
    LEVEL reset, so a failed level can be restarted and its segment replayed
    without double-applying actions."""
    sys.path[:0] = [str(GKM / "arc"), str(GKM / "cone")]
    import arc_agi3_adapter as arc

    env = arc.LocalArcEnv(game, operation_mode="offline",
                          environments_dir=str(GKM / "environment_files"))
    snap = env.reset()
    levels = snap.levels_completed
    segments, start = [], 0
    for i, a in enumerate(actions):
        action_id, data = decode_action(a)
        snap = env.step(
            arc.GameAction(action_id),
            **({"x": data["x"], "y": data["y"]} if data else {}),
        )
        if snap.levels_completed > levels:
            segments.append(list(actions[start:i + 1]))
            start, levels = i + 1, snap.levels_completed
    if start < len(actions):  # trailing moves that close no level (not expected)
        segments.append(list(actions[start:]))
    return segments


def _reset_with_retry(env, label: str, tries: int = 5):
    for t in range(tries):
        fd = env.reset()
        if fd is not None:
            return fd
        print(f"  {label}: RESET failed (attempt {t + 1}/{tries}); retrying")
        time.sleep(3 * (t + 1))
    raise RuntimeError(f"{label}: RESET failed after {tries} attempts")


def replay(env, segments, engine_action_cls, label: str,
           level_retries: int = 4, max_recovery_cycles: int = 20,
           verbose: bool = True) -> int:
    # RemoteEnvironmentWrapper performs the authoritative initial RESET in its
    # constructor. Reuse that frame so ONLINE does not create an empty extra
    # public run before the admitted replay; reset only when no initial frame
    # exists or during bounded recovery below.
    fd = getattr(env, "observation_space", None)
    if fd is None:
        fd = _reset_with_retry(env, label)
    levels = int(fd.levels_completed or 0)
    moves = 0
    attempts_at_level: dict[int, int] = {}
    recovery_cycles = 0
    while levels < len(segments):
        k = levels + 1
        seg = segments[k - 1]
        failed_at = None
        for i, a in enumerate(seg):
            action_id, data = decode_action(a)
            fd = env.step(engine_action_cls[f"ACTION{action_id}"], data=data)
            if fd is None:  # transient API failure; remote state is uncertain
                failed_at = i
                break
            moves += 1
        now = int(fd.levels_completed or 0) if fd is not None else levels
        if failed_at is None and now >= k:
            levels = now
            attempts_at_level.pop(k, None)
            if verbose:
                print(f"  {label}: level {now} after {moves} moves")
            state = getattr(fd.state, "name", str(fd.state))
            if state == "WIN":
                break
            continue

        attempts_at_level[k] = attempts_at_level.get(k, 0) + 1
        attempt = attempts_at_level[k]
        why = (f"step {failed_at} failed" if failed_at is not None
               else f"segment ended at levels={now}")
        print(f"  {label}: level {k} attempt {attempt}/{level_retries}: {why}; "
              f"level-reset and recover")
        if attempt >= level_retries:
            raise RuntimeError(f"{label}: level {k} failed {level_retries} attempts")
        recovery_cycles += 1
        if recovery_cycles > max_recovery_cycles:
            raise RuntimeError(
                f"{label}: exceeded {max_recovery_cycles} total recovery cycles"
            )
        time.sleep(3 * attempt)
        fd = _reset_with_retry(env, label)
        recovered = int(fd.levels_completed or 0)
        if recovered < levels:
            print(f"  {label}: reset rolled back from level {levels} to "
                  f"{recovered}; rebuilding from level {recovered + 1}")
        levels = recovered
    return levels


def execute_remote_run(
    *,
    args: argparse.Namespace,
    arcade: Any,
    engine_action_cls: Any,
    games: list[str],
    plan: dict[str, dict],
    segs: dict[str, list],
    artifact_root: Path,
    checkpoint_hashes: dict[str, str],
    binding: dict[str, Any] | None,
    release_verification: dict[str, Any] | None,
    arc_agi_version: str,
    started_at_utc: str,
    claimed_levels: int,
    complete_release: bool,
) -> int:
    """Open exactly once and durably journal a complete remote replay."""
    if complete_release:
        assert binding is not None
        expected_output = _lexical_absolute(
            canonical_run_receipt_path(binding["receipt_sha256"], args.mode)
        )
        if (
            args.output_json is None
            or _lexical_absolute(args.output_json) != expected_output
        ):
            print(
                "complete remote run receipt must use the canonical path: "
                f"{expected_output}"
            )
            return 2
        if path_lexists(expected_output):
            print(
                "canonical complete run receipt already exists; remote open "
                "is forbidden"
            )
            return 2
        if not immutable_source_url(args.source_url, args.source_revision):
            print("complete remote run source URL is not the exact public revision")
            return 2
        try:
            verify_public_revision(args.source_revision)
            verify_runtime_revision(args.source_revision, games)
        except (OSError, RunJournalError, ValueError) as exc:
            print(f"complete remote run source/runtime binding failed: {exc}")
            return 2
    run_tags = (
        complete_scorecard_tags(
            mode=args.mode,
            receipt_sha256=binding["receipt_sha256"],
            revision=args.source_revision,
        )
        if complete_release and binding is not None
        else [tag.strip() for tag in args.tags.split(",") if tag.strip()]
    )
    logical_command = command_identity(
        args=args,
        games=games,
        artifact_root=artifact_root,
    )
    logical_command["tags"] = run_tags
    journal_context: Any = contextlib.nullcontext(None)
    if complete_release:
        assert binding is not None
        journal_context = CompleteRunJournal(
            canonical_run_journal_path(binding["receipt_sha256"]),
            journal_id=journal_id_for_release(binding["receipt_sha256"]),
            snapshot_directory=RUN_JOURNAL_ROOT / "snapshots",
        )

    try:
        with contextlib.ExitStack() as stack:
            journal = stack.enter_context(journal_context)
            operation_id: str | None = None
            scorecard_opaque: dict[str, Any] | None = None
            intent_record: dict[str, Any] | None = None
            opened_record: dict[str, Any] | None = None
            receipt_reservation: ReservedReceipt | None = None
            if journal is not None:
                journal.assert_can_start(args.mode)
                operation_id = secrets.token_hex(32)
                scorecard_opaque = complete_scorecard_opaque(
                    operation_id=operation_id,
                    mode=args.mode,
                    receipt_sha256=binding["receipt_sha256"],
                    canonical_tree_sha256=binding["canonical_tree_sha256"],
                    revision=args.source_revision,
                )
                intent_payload = {
                    "mode": args.mode,
                    "source_url": args.source_url,
                    "source_revision": args.source_revision,
                    "arc_agi_toolkit_version": arc_agi_version,
                    "release_receipt_sha256": binding["receipt_sha256"],
                    "canonical_tree_sha256": binding[
                        "canonical_tree_sha256"
                    ],
                    "checkpoint_sha256_digest": json_sha256(
                        checkpoint_hashes
                    ),
                    "command_sha256": json_sha256(logical_command),
                    "output_receipt": logical_path(args.output_json),
                    "tags": run_tags,
                    "opaque": scorecard_opaque,
                }
                # O_EXCL is the authority for publication admission.  Keep the
                # descriptor open from before INTENT through the final fsync so
                # no external creator can win a check-then-create race after a
                # remote operation has become possible.
                receipt_reservation = stack.enter_context(
                    ReservedReceipt(args.output_json)
                )
                intent_record = journal.append(
                    kind="INTENT",
                    operation_id=operation_id,
                    payload=intent_payload,
                )

            def receipt_core(
                *,
                status: str,
                card_id: str | None,
                open_outcome: dict[str, Any],
                close_outcome: dict[str, Any],
                results: dict[str, Any],
                aggregate: dict[str, Any] | None,
                closed_at_utc: str,
                close_started_at_utc: str | None,
                close_finished_at_utc: str | None,
            ) -> dict[str, Any]:
                return {
                    "schema": 2,
                    "mode": args.mode,
                    "status": status,
                    "scorecard_id": card_id,
                    "scorecard_url": (
                        f"https://arcprize.org/scorecards/{card_id}"
                        if card_id is not None
                        else None
                    ),
                    "scorecard_open": open_outcome,
                    "scorecard_close": close_outcome,
                    "scorecard_tags": run_tags,
                    "scorecard_opaque": scorecard_opaque,
                    "source_url": args.source_url,
                    "source_revision": args.source_revision,
                    "arc_agi_toolkit_version": arc_agi_version,
                    "started_at_utc": started_at_utc,
                    "closed_at_utc": closed_at_utc,
                    "scorecard_close_started_at_utc": close_started_at_utc,
                    "scorecard_close_finished_at_utc": close_finished_at_utc,
                    "command": logical_command,
                    "artifact_root": logical_command["artifact_root"],
                    "release_receipt": (
                        logical_path(args.release_receipt)
                        if args.release_receipt is not None
                        else None
                    ),
                    "release_verification": release_verification,
                    "release_binding": binding,
                    "checkpoint_sha256": checkpoint_hashes,
                    "claimed_levels": claimed_levels,
                    "authoritative_levels": (
                        binding["authoritative_level_count"] if binding else None
                    ),
                    "stored_actions": sum(
                        len(value["final_path"]) for value in plan.values()
                    ),
                    "results": results,
                    "aggregate": aggregate,
                }

            if journal is not None:
                journal._assert_path_identity()
            try:
                card_id = arcade.open_scorecard(
                    source_url=args.source_url,
                    tags=run_tags,
                    opaque=scorecard_opaque,
                )
                if not isinstance(card_id, str) or (
                    complete_release and UUID_RE.fullmatch(card_id) is None
                ):
                    raise ValueError("open_scorecard returned no valid card ID")
            except Exception as exc:
                print(
                    "scorecard open outcome is ambiguous; automatic retry is "
                    f"forbidden ({type(exc).__name__})."
                )
                if journal is None:
                    return 1
                assert operation_id is not None and intent_record is not None
                core = receipt_core(
                    status="FAIL",
                    card_id=None,
                    open_outcome={
                        "status": "ambiguous",
                        "error_type": type(exc).__name__,
                    },
                    close_outcome={
                        "status": "not_attempted",
                        "error_type": None,
                    },
                    results={},
                    aggregate=None,
                    closed_at_utc=utc_now(),
                    close_started_at_utc=None,
                    close_finished_at_utc=None,
                )
                try:
                    run_receipt = finalize_journal_receipt(
                        journal=journal,
                        receipt_core=core,
                        operation_id=operation_id,
                        opaque=scorecard_opaque,
                        intent_record=intent_record,
                        opened_record=None,
                        terminal_outcome="OPEN_OUTCOME_AMBIGUOUS",
                    )
                    assert receipt_reservation is not None
                    receipt_reservation.publish_json(run_receipt)
                except (OSError, RunJournalError, ValueError) as journal_exc:
                    print(
                        "cannot publish ambiguous-open evidence; the durable "
                        f"intent remains blocking: {journal_exc}"
                    )
                return 1

            if journal is not None:
                assert operation_id is not None
                try:
                    opened_record = journal.append(
                        kind="OPENED",
                        operation_id=operation_id,
                        payload={
                            "card_id": card_id,
                            "scorecard_url": (
                                f"https://arcprize.org/scorecards/{card_id}"
                            ),
                        },
                    )
                except (OSError, RunJournalError, ValueError) as exc:
                    try:
                        arcade.close_scorecard(card_id)
                    except Exception:
                        pass
                    print(
                        "card ID was not durably journaled; no environment was "
                        "opened and human/provider reconciliation is required: "
                        f"{exc}"
                    )
                    return 1
            print(f"scorecard opened: {card_id} (mode={args.mode})")

            results: dict[str, Any] = {}
            ok = True
            card = None
            close_outcome = {"status": "not_attempted", "error_type": None}
            close_started_at_utc: str | None = None
            close_finished_at_utc: str | None = None
            try:
                for game, checkpoint_value in plan.items():
                    try:
                        env = arcade.make(game, scorecard_id=card_id)
                    except Exception as exc:
                        print(
                            f"{game}: make() aborted: {type(exc).__name__}: {exc}"
                        )
                        results[game] = {
                            "remote": -1,
                            "claimed": checkpoint_value["reached"],
                        }
                        ok = False
                        continue
                    if env is None:
                        print(f"{game}: make() failed")
                        results[game] = {
                            "remote": -1,
                            "claimed": checkpoint_value["reached"],
                        }
                        ok = False
                        continue
                    try:
                        reached = replay(
                            env, segs[game], engine_action_cls, game
                        )
                    except Exception as exc:
                        print(
                            f"{game}: replay aborted: {type(exc).__name__}"
                        )
                        reached, ok = -1, False
                    results[game] = {
                        "remote": reached,
                        "claimed": checkpoint_value["reached"],
                    }
                    endpoint_matches = (
                        reached == checkpoint_value["reached"]
                        if binding is not None
                        else reached >= checkpoint_value["reached"]
                    )
                    status = "OK" if endpoint_matches else "DESYNC"
                    if not endpoint_matches:
                        ok = False
                    print(
                        f"{game}: remote levels_completed={reached} "
                        f"vs local {checkpoint_value['reached']} -> {status}"
                    )
            finally:
                close_started_at_utc = utc_now()
                try:
                    card = arcade.close_scorecard(card_id)
                except Exception as exc:
                    close_outcome = {
                        "status": "ambiguous",
                        "error_type": type(exc).__name__,
                    }
                    ok = False
                    print(
                        "scorecard close outcome is ambiguous; a FAIL receipt "
                        "will be written for remote recovery verification "
                        f"({type(exc).__name__})."
                    )
                else:
                    if card is None:
                        close_outcome = {
                            "status": "absent",
                            "error_type": None,
                        }
                        ok = False
                    else:
                        close_outcome = {
                            "status": "confirmed",
                            "error_type": None,
                        }
                close_finished_at_utc = utc_now()

            if close_outcome["status"] == "confirmed":
                print(f"scorecard closed: {card_id}")
            else:
                print(f"scorecard close not confirmed locally: {card_id}")
            print(f"scorecard_url: https://arcprize.org/scorecards/{card_id}")
            after_hashes = {
                game: sha256_file(checkpoint_path(artifact_root, game))
                for game in games
            }
            if after_hashes != checkpoint_hashes:
                print("frozen checkpoint bytes changed during scorecard replay.")
                ok = False
            aggregate = None
            if card is not None and hasattr(card, "model_dump"):
                try:
                    aggregate = card.model_dump(
                        mode="json", exclude={"api_key"}
                    )
                except Exception as exc:
                    print(
                        "closed scorecard serialization failed: "
                        f"{type(exc).__name__}"
                    )
                    ok = False
                else:
                    if isinstance(aggregate, dict):
                        print(
                            "aggregate summary:",
                            {
                                "card_id": aggregate.get("card_id"),
                                "score": aggregate.get("score"),
                                "total_environments": aggregate.get(
                                    "total_environments"
                                ),
                                "total_levels_completed": aggregate.get(
                                    "total_levels_completed"
                                ),
                                "total_actions": aggregate.get(
                                    "total_actions"
                                ),
                            },
                        )
            try:
                aggregate = validate_closed_scorecard(
                    aggregate,
                    mode=args.mode,
                    card_id=card_id,
                    source_url=args.source_url,
                    games=games,
                    plan=plan,
                    scorecard_tags=run_tags,
                    scorecard_opaque=scorecard_opaque,
                )
            except ValueError as exc:
                print(f"closed scorecard rejected: {exc}")
                ok = False
                # Invalid provider payloads are not safe receipt material.  A
                # FAIL receipt records their rejection without copying unknown
                # fields or free-form messages into a durable artifact.
                aggregate = None
            core = receipt_core(
                status="PASS" if ok else "FAIL",
                card_id=card_id,
                open_outcome={"status": "confirmed", "error_type": None},
                close_outcome=close_outcome,
                results=results,
                aggregate=aggregate,
                closed_at_utc=utc_now(),
                close_started_at_utc=close_started_at_utc,
                close_finished_at_utc=close_finished_at_utc,
            )
            run_receipt = core
            if journal is not None:
                assert (
                    operation_id is not None
                    and scorecard_opaque is not None
                    and intent_record is not None
                    and opened_record is not None
                )
                if close_outcome["status"] == "confirmed":
                    terminal_outcome = (
                        "CLOSED_CONFIRMED_PASS"
                        if ok
                        else "CLOSED_CONFIRMED_FAIL"
                    )
                else:
                    terminal_outcome = "CLOSE_OUTCOME_AMBIGUOUS"
                try:
                    run_receipt = finalize_journal_receipt(
                        journal=journal,
                        receipt_core=core,
                        operation_id=operation_id,
                        opaque=scorecard_opaque,
                        intent_record=intent_record,
                        opened_record=opened_record,
                        terminal_outcome=terminal_outcome,
                    )
                except (OSError, RunJournalError, ValueError) as exc:
                    print(
                        "cannot durably terminate/snapshot the run journal; "
                        f"the final receipt was not published: {exc}"
                    )
                    return 1
            if args.output_json is not None:
                try:
                    if receipt_reservation is not None:
                        receipt_reservation.publish_json(run_receipt)
                    else:
                        write_new_json(args.output_json, run_receipt)
                except (OSError, RunJournalError, ValueError) as exc:
                    print(f"cannot write run receipt: {exc}")
                    return 1
            return 0 if ok else 1
    except (OSError, RunJournalError, ValueError) as exc:
        print(f"complete remote-run journal rejected the start: {exc}")
        return 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("online", "competition"), default="online",
                    help="online = remote dry run; competition = the single real scorecard run")
    ap.add_argument("--games", default="wa30,ls20",
                    help="comma-separated games or 'all' (default: wa30,ls20)")
    ap.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="frozen *_legs tree containing the scored checkpoints",
    )
    ap.add_argument(
        "--release-receipt",
        type=Path,
        help=(
            "content-addressed schema-v2 receipt; triggers full historical "
            "release-gate verification before endpoint binding"
        ),
    )
    ap.add_argument(
        "--release-verifier-root",
        type=Path,
        help=(
            "already extracted receipt-bound verifier tree; by default the "
            "exact source revision is read from local Git history"
        ),
    )
    ap.add_argument(
        "--expected-claimed-levels",
        type=int,
        help="fail unless the receipt and checkpoints claim exactly this depth",
    )
    ap.add_argument(
        "--output-json",
        type=Path,
        help="write a new machine-readable run receipt; never overwrite",
    )
    ap.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate the frozen receipt, checkpoints, and local level segments without network access",
    )
    ap.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    ap.add_argument(
        "--source-revision",
        help=(
            "immutable public GKM Git revision; required for a complete "
            "183-level release and recorded in the run receipt"
        ),
    )
    ap.add_argument("--tags", default="gkm,replay-validated")
    args = ap.parse_args()

    started_at_utc = dt.datetime.now(dt.timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    try:
        arc_agi_version = toolkit_version()
    except ValueError as exc:
        print(f"cannot identify ARC toolkit: {exc}")
        return 2
    if args.source_revision is not None:
        if REVISION_RE.fullmatch(args.source_revision) is None:
            print("--source-revision must be a 40- or 64-hex Git revision.")
            return 2
        if args.source_url == DEFAULT_SOURCE_URL:
            args.source_url = (
                f"{DEFAULT_SOURCE_URL}/tree/{args.source_revision}"
            )
        if not immutable_source_url(args.source_url, args.source_revision):
            print("--source-url does not bind the requested public revision.")
            return 2

    artifact_root = args.artifact_root.resolve()
    if not artifact_root.is_dir() or artifact_root.is_symlink():
        print("artifact root must be a non-symlink directory.")
        return 2
    try:
        games = parse_games(args.games, artifact_root)
    except ValueError as exc:
        print(f"invalid scorecard plan: {exc}")
        return 2
    if args.mode == "competition" and args.release_receipt is None:
        print("competition mode requires --release-receipt.")
        return 2
    if args.release_receipt is not None and args.source_revision is not None:
        try:
            verify_runtime_revision(args.source_revision, games)
        except (OSError, RunJournalError, UnicodeError, ValueError) as exc:
            print(f"release verifier runtime manifest rejected: {exc}")
            return 2

    binding = None
    release_verification = None
    if args.release_receipt is not None:
        try:
            raw_release_verification = verify_frozen_release(
                receipt_path=args.release_receipt.resolve(),
                canonical_root=artifact_root,
                repo_root=GKM,
                verifier_root=(
                    args.release_verifier_root.resolve()
                    if args.release_verifier_root is not None
                    else None
                ),
            )
            release_verification = project_release_verification(
                raw_release_verification
            )
        except (
            FrozenReleaseError,
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            print(f"release receipt rejected: {exc}")
            return 2
    try:
        loaded = {game: load_checkpoint(game, artifact_root) for game in games}
        plan = {game: value for game, (value, _) in loaded.items()}
        checkpoint_hashes = {game: digest for game, (_, digest) in loaded.items()}
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"cannot load frozen checkpoint plan: {exc}")
        return 2
    if args.release_receipt is not None:
        try:
            binding = release_binding(
                args.release_receipt.resolve(),
                games,
                plan,
                checkpoint_hashes,
            )
            if (
                release_verification is None
                or binding["receipt_sha256"]
                != release_verification.get("receipt_sha256")
                or binding["release_identity_source_revision"]
                != release_verification.get(
                    "verification_context_source_revision"
                )
            ):
                raise ValueError("release receipt changed after full verification")
        except (
            FrozenReleaseError,
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            print(f"release endpoint binding rejected: {exc}")
            return 2
    if args.mode == "competition" and (
        binding is None
        or len(games) != 25
        or binding["authoritative_level_count"] != 183
    ):
        print(
            "competition mode requires one receipt-bound 25-game/183-level "
            "authoritative plan."
        )
        return 2
    claimed_levels = sum(
        int(value.get("reached", -1)) for value in plan.values()
    )
    if (
        args.expected_claimed_levels is not None
        and claimed_levels != args.expected_claimed_levels
    ):
        print(
            "claimed-level mismatch: "
            f"expected {args.expected_claimed_levels}, found {claimed_levels}"
        )
        return 2
    if (
        binding is not None
        and claimed_levels != binding["claimed_level_count"]
    ):
        print("checkpoint total differs from release receipt.")
        return 2
    complete_release = (
        binding is not None
        and binding["claimed_level_count"] == 183
        and binding["authoritative_level_count"] == 183
    )
    if complete_release and args.expected_claimed_levels != 183:
        print(
            "a complete 183-level release requires "
            "--expected-claimed-levels 183."
        )
        return 2
    if complete_release and args.source_revision is None:
        print(
            "a complete 183-level release requires --source-revision so the "
            "scorecard binds immutable public source."
        )
        return 2
    if complete_release and (
        args.source_revision
        != binding.get("release_identity_source_revision")
    ):
        print(
            "--source-revision differs from the complete release receipt's "
            "source revision."
        )
        return 2
    if complete_release and arc_agi_version != AUDITED_TOOLKIT_VERSION:
        print(
            "a complete remote release requires the audited arc-agi toolkit "
            f"version {AUDITED_TOOLKIT_VERSION}; found {arc_agi_version}."
        )
        return 2
    if complete_release and not args.preflight_only and args.output_json is None:
        print(
            "a complete remote release run requires --output-json for its "
            "non-overwriting schema-2 receipt."
        )
        return 2
    if args.output_json is not None:
        output_path = args.output_json.resolve()
        if output_path == artifact_root or artifact_root in output_path.parents:
            print("run receipt output must be outside the frozen artifact root.")
            return 2
        if path_lexists(args.output_json):
            print("run receipt output already exists; remote open is forbidden.")
            return 2

    if complete_release:
        try:
            assert args.source_revision is not None
            verify_runtime_revision(args.source_revision, games)
        except (OSError, RunJournalError, UnicodeError, ValueError) as exc:
            print(f"complete runtime manifest rejected: {exc}")
            return 2

    segs = {}
    for g, ck in plan.items():
        segs[g] = level_segments(g, ck["final_path"])
        if len(segs[g]) != ck["reached"]:
            print(
                f"{g}: local segmentation reached {len(segs[g])} "
                f"but checkpoint claims {ck['reached']}"
            )
            return 2
        print(f"{g}: replaying {len(ck['final_path'])} recorded actions in "
              f"{len(segs[g])} level segments (locally validated reached={ck['reached']})")

    if args.preflight_only:
        logical_command = command_identity(
            args=args,
            games=games,
            artifact_root=artifact_root,
        )
        run_receipt = {
            "schema": 2,
            "mode": "preflight",
            "status": "PASS",
            "scorecard_id": None,
            "scorecard_url": None,
            "source_url": args.source_url,
            "source_revision": args.source_revision,
            "arc_agi_toolkit_version": arc_agi_version,
            "started_at_utc": started_at_utc,
            "closed_at_utc": None,
            "command": logical_command,
            "artifact_root": logical_command["artifact_root"],
            "release_receipt": (
                logical_path(args.release_receipt)
                if args.release_receipt is not None
                else None
            ),
            "release_verification": release_verification,
            "release_binding": binding,
            "checkpoint_sha256": checkpoint_hashes,
            "claimed_levels": claimed_levels,
            "authoritative_levels": (
                binding["authoritative_level_count"] if binding else None
            ),
            "stored_actions": sum(
                len(value["final_path"]) for value in plan.values()
            ),
            "results": {
                game: {
                    "locally_segmented": len(segs[game]),
                    "claimed": plan[game]["reached"],
                }
                for game in games
            },
            "aggregate": None,
        }
        if args.output_json is not None:
            try:
                write_new_json(args.output_json, run_receipt)
            except OSError as exc:
                print(f"cannot write run receipt: {exc}")
                return 1
        print(
            "preflight PASS: "
            f"{len(games)} games, {claimed_levels} claimed levels"
        )
        return 0

    try:
        arc_api_key = load_arc_api_key()
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"cannot read ARC_API_KEY: {exc}")
        return 2
    if not arc_api_key:
        print("ARC_API_KEY required (repo .env or environment).")
        return 2
    from arc_agi import Arcade, OperationMode  # network toolkit; import late
    from arcengine import GameAction as EngineAction

    arcade = Arcade(arc_api_key=arc_api_key,
                    operation_mode=OperationMode(args.mode))
    return execute_remote_run(
        args=args,
        games=games,
        arcade=arcade,
        engine_action_cls=EngineAction,
        plan=plan,
        segs=segs,
        artifact_root=artifact_root,
        checkpoint_hashes=checkpoint_hashes,
        binding=binding,
        release_verification=release_verification,
        arc_agi_version=arc_agi_version,
        started_at_utc=started_at_utc,
        claimed_levels=claimed_levels,
        complete_release=complete_release,
    )


if __name__ == "__main__":
    sys.exit(main())
