#!/usr/bin/env python3
"""Reproduce and verify the manuscript's machine-derived evidence.

The default mode is semiautomated: it recomputes GKM from the local canonical
checkpoints and reuses the checked-in, checksum-pinned comparator rows.  Supply
all four external-artifact arguments to rebuild the OPINE, baseline1, and
Retrodict boundary audits from their released artifacts as well.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any


SYSTEMS = ("GKM", "OPINE", "baseline1", "Retrodict")
FROZEN_SOURCE_HISTORY_REVISION = "4d0e42f34d7b1db8305f03d725528dfdefe22511"
FROZEN_HISTORY_TREE = "arc/crack_lab/agent_solutions"
FROZEN_SOURCE_HISTORY_TREE_SHA1 = "85543629cfcafc70eb7230493f394059c8e0ac45"
FROZEN_SOURCE_AUDIT_EXCLUSIONS = (
    ("ft09", 2),
    *(("tr87", level) for level in range(1, 7)),
)
FROZEN_V2_UNCLAIMED_BOUNDARIES = [
    {"game": "lf52", "level": 9},
    {"game": "lf52", "level": 10},
]
REVISION_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
GIT_TREE_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
MAX_RECEIPT_BYTES = 32 * 1024 * 1024
MAX_HISTORY_ARCHIVE_BYTES = 1024 * 1024 * 1024
MAX_HISTORY_MEMBERS = 50_000
MAX_HISTORY_FILE_BYTES = 32 * 1024 * 1024
STAT_FIELDS = (
    "retained_or_winning_checkpoints",
    "exact_winning_checkpoints",
    "exact_adjacent_transitions",
    "released_memory_transitions",
    "transitions_with_level_to_level_marginal_comparison",
    "marginal_decreases",
    "sharp_half_or_more_marginal_drops",
    "hard_literal_world_model_reuse_witnesses",
)
MACRO_PREFIX = {
    "GKM": "GKM",
    "OPINE": "OPINE",
    "baseline1": "BaselineOne",
    "Retrodict": "Retrodict",
}
MACRO_FIELD = {
    "retained_or_winning_checkpoints": "RetainedCheckpoints",
    "exact_winning_checkpoints": "ExactWins",
    "exact_adjacent_transitions": "ExactAdjacent",
    "released_memory_transitions": "MemoryTransitions",
    "transitions_with_level_to_level_marginal_comparison": "Comparable",
    "marginal_decreases": "Decreases",
    "sharp_half_or_more_marginal_drops": "SharpDrops",
    "hard_literal_world_model_reuse_witnesses": "HardReuse",
    "sharp_drops_with_literal_reuse": "CoupledWitnesses",
    "trace_solve_events": "TraceSolveEvents",
    "analyzer_or_unknown_wins": "AnalyzerOrUnknownWins",
    "uncoupled_sharp": "UncoupledSharp",
    "reported_clears": "ReportedClears",
    "exact_authored_contractions": "ExactAuthoredContractions",
    "direct_literal_wins": "DirectLiteralWins",
    "inline_literal_wins": "InlineLiteralWins",
    "executor_literal_wins": "ExecutorLiteralWins",
    "memory_contractions": "MemoryContractions",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument(
        "--release-root",
        type=Path,
        help=(
            "Frozen normalized <game>_legs tree used only for release, endpoint, "
            "taint, and action verification."
        ),
    )
    parser.add_argument(
        "--acquisition-root",
        type=Path,
        help=(
            "Frozen acquisition-source <game>_legs tree used to compute D(s), "
            "the marginal-complexity table, and empirical figures."
        ),
    )
    parser.add_argument(
        "--history-root",
        type=Path,
        help=(
            "Acquisition tree containing exact historical winning snapshots for "
            "the GKM source/reuse audit. Its complete Git-tree digest must match "
            "the pinned history tree; the default is extracted from "
            "--history-revision."
        ),
    )
    parser.add_argument(
        "--history-revision",
        default=FROZEN_SOURCE_HISTORY_REVISION,
        help=(
            "Full local Git revision from which the immutable GKM acquisition "
            "history is extracted (default: the manuscript source-history snapshot)."
        ),
    )
    parser.add_argument(
        "--release-receipt",
        type=Path,
        help=(
            "Schema-v2 complete or partial release receipt. When supplied, "
            "the manuscript suite uses the fail-closed release gate instead "
            "of applying the legacy schema-1 promotion audit to normalized "
            "artifacts."
        ),
    )
    parser.add_argument(
        "--release-verifier-root",
        type=Path,
        default=(
            Path(os.environ["RELEASE_VERIFIER_ROOT"])
            if os.environ.get("RELEASE_VERIFIER_ROOT")
            else None
        ),
        help=(
            "Already extracted receipt-bound verifier source. When omitted, "
            "the receipt's source revision is extracted from local Git history."
        ),
    )
    parser.add_argument("--opine-artifacts", type=Path)
    parser.add_argument("--baseline-release", type=Path)
    parser.add_argument("--baseline-repo", type=Path)
    parser.add_argument("--retrodict-runs", type=Path)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Replace tracked audit/figure outputs after successful reproduction.",
    )
    parser.add_argument(
        "--allow-live-gkm-drift",
        action="store_true",
        help="Report rather than fail when the active campaign has advanced.",
    )
    parser.add_argument(
        "--build-paper",
        action="store_true",
        help="Build the PDF after audits and figures pass.",
    )
    parser.add_argument(
        "--require-complete-lineage",
        action="store_true",
        help=(
            "Require every canonical game to have promotion manifests. "
            "The taint audit itself always runs; this enables the final-release "
            "lineage-completeness gate."
        ),
    )
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def _run(command: list[str], *, cwd: Path) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def _run_json(command: list[str], *, cwd: Path) -> dict[str, Any]:
    print("+", " ".join(command))
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError("JSON command did not return an object")
    return payload


def _portable_path(path: Path, repo: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _git_environment() -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", os.defpath),
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
    }


def _safe_archive_name(value: str) -> PurePosixPath:
    pure = PurePosixPath(value)
    if (
        not value
        or pure.is_absolute()
        or pure.as_posix() != value
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise RuntimeError(f"unsafe historical archive entry: {value!r}")
    return pure


def _materialize_history(
    *, repo: Path, revision: str, output: Path
) -> Path:
    """Extract the frozen manuscript source history from local Git, fail closed."""
    if not REVISION_RE.fullmatch(revision):
        raise RuntimeError("history revision must be a full lowercase object ID")
    env = _git_environment()
    for command in (
        ["git", "-C", str(repo), "cat-file", "-e", f"{revision}^{{commit}}"],
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", revision, "HEAD"],
    ):
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
            timeout=180,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "frozen history revision is unavailable or is not an ancestor of HEAD"
            )

    archive_path = output.parent / "history.tar"
    with archive_path.open("xb") as archive_stream:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "archive",
                "--format=tar",
                revision,
                "--",
                FROZEN_HISTORY_TREE,
            ],
            check=False,
            stdout=archive_stream,
            stderr=subprocess.PIPE,
            env=env,
            timeout=180,
        )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            "cannot extract frozen acquisition history"
            + (f": {detail[:500]}" if detail else "")
        )
    if archive_path.stat().st_size > MAX_HISTORY_ARCHIVE_BYTES:
        raise RuntimeError("frozen acquisition-history archive is unexpectedly large")

    prefix = PurePosixPath(FROZEN_HISTORY_TREE)
    seen: set[str] = set()
    folded: set[str] = set()
    total_size = 0
    with tarfile.open(archive_path, mode="r:") as archive:
        members = archive.getmembers()
        if len(members) > MAX_HISTORY_MEMBERS:
            raise RuntimeError("frozen acquisition-history archive has too many entries")
        for member in members:
            pure = _safe_archive_name(member.name)
            if (
                member.isdir()
                and len(pure.parts) < len(prefix.parts)
                and prefix.parts[: len(pure.parts)] == pure.parts
            ):
                continue
            if pure.parts[: len(prefix.parts)] != prefix.parts:
                raise RuntimeError(
                    f"historical archive entry is outside the requested tree: {pure}"
                )
            name = pure.as_posix()
            casefolded = name.casefold()
            if name in seen or casefolded in folded:
                raise RuntimeError(f"duplicate historical archive entry: {name}")
            seen.add(name)
            folded.add(casefolded)
            relative = PurePosixPath(*pure.parts[len(prefix.parts) :])
            target = output.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile() or member.size < 0:
                raise RuntimeError(f"non-regular historical archive entry: {name}")
            if member.size > MAX_HISTORY_FILE_BYTES:
                raise RuntimeError(f"historical archive file is too large: {name}")
            total_size += member.size
            if total_size > MAX_HISTORY_ARCHIVE_BYTES:
                raise RuntimeError("expanded acquisition history is unexpectedly large")
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"cannot extract historical archive entry: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as destination:
                shutil.copyfileobj(source, destination)
            target.chmod(stat.S_IRUSR | stat.S_IWUSR)
    archive_path.unlink()
    if not output.is_dir() or not any(output.iterdir()):
        raise RuntimeError("frozen acquisition history extracted no files")
    return output


def _git_object_id(kind: bytes, payload: bytes) -> bytes:
    return hashlib.sha1(
        kind + b" " + str(len(payload)).encode("ascii") + b"\0" + payload
    ).digest()


def _copy_authenticated_history(
    *, source: Path, output: Path, expected_tree_sha1: str
) -> Path:
    """Copy an archive-supplied history and authenticate its complete Git tree."""
    root = Path(source)
    try:
        root_metadata = root.lstat()
    except OSError as exc:
        raise RuntimeError("cannot stat explicit source-history root") from exc
    if not stat.S_ISDIR(root_metadata.st_mode) or stat.S_ISLNK(root_metadata.st_mode):
        raise RuntimeError("explicit source-history root is not a real directory")

    state = {"members": 0, "bytes": 0}

    def copy_tree(current: Path, target: Path) -> bytes:
        target.mkdir(parents=True, exist_ok=False)
        entries: list[tuple[bytes, bool, bytes]] = []
        try:
            children = list(os.scandir(current))
        except OSError as exc:
            raise RuntimeError("cannot enumerate explicit source-history root") from exc
        for child in children:
            try:
                name = child.name.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise RuntimeError("source-history path is not UTF-8") from exc
            if not name or b"/" in name or b"\0" in name:
                raise RuntimeError("source-history contains an unsafe path name")
            try:
                metadata = child.stat(follow_symlinks=False)
            except OSError as exc:
                raise RuntimeError("cannot stat source-history entry") from exc
            state["members"] += 1
            if state["members"] > MAX_HISTORY_MEMBERS:
                raise RuntimeError("source-history tree has too many entries")
            source_path = Path(child.path)
            target_path = target / child.name
            if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
                object_id = copy_tree(source_path, target_path)
                entries.append((name, True, object_id))
                continue
            if (
                not stat.S_ISREG(metadata.st_mode)
                or stat.S_ISLNK(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_size > MAX_HISTORY_FILE_BYTES
            ):
                raise RuntimeError("source-history contains a non-regular file")
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(source_path, flags)
            except OSError as exc:
                raise RuntimeError("cannot securely open source-history file") from exc
            with os.fdopen(descriptor, "rb") as handle:
                opened = os.fstat(handle.fileno())
                if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
                    raise RuntimeError("source-history file changed during copy")
                raw = handle.read(MAX_HISTORY_FILE_BYTES + 1)
            if len(raw) > MAX_HISTORY_FILE_BYTES:
                raise RuntimeError("source-history file is too large")
            state["bytes"] += len(raw)
            if state["bytes"] > MAX_HISTORY_ARCHIVE_BYTES:
                raise RuntimeError("source-history tree is unexpectedly large")
            with target_path.open("xb") as destination:
                destination.write(raw)
            target_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
            entries.append((name, False, _git_object_id(b"blob", raw)))

        payload = bytearray()
        for name, is_directory, object_id in sorted(
            entries, key=lambda item: item[0] + (b"/" if item[1] else b"")
        ):
            payload.extend(b"40000 " if is_directory else b"100644 ")
            payload.extend(name)
            payload.append(0)
            payload.extend(object_id)
        return _git_object_id(b"tree", bytes(payload))

    actual_tree_sha1 = copy_tree(root, output).hex()
    if actual_tree_sha1 != expected_tree_sha1:
        raise RuntimeError(
            "explicit source-history tree does not match the pinned Git tree: "
            f"expected {expected_tree_sha1}, found {actual_tree_sha1}"
        )
    return output


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
        raise RuntimeError("release receipt is not canonical JSON") from exc


def _receipt_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"release receipt has duplicate key {key!r}")
        result[key] = value
    return result


def _snapshot_release_receipt(
    *, source: Path, output_directory: Path
) -> tuple[Path, dict[str, Any], str]:
    """Read one content-addressed receipt and retain the exact verified bytes."""
    receipt = Path(source)
    try:
        metadata = receipt.lstat()
    except OSError as exc:
        raise RuntimeError("cannot stat release receipt") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size > MAX_RECEIPT_BYTES
    ):
        raise RuntimeError("release receipt is not a bounded single-link file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(receipt, flags)
    except OSError as exc:
        raise RuntimeError("cannot securely open release receipt") from exc
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size > MAX_RECEIPT_BYTES
        ):
            raise RuntimeError("release receipt changed during bounded read")
        raw = handle.read(MAX_RECEIPT_BYTES + 1)
    if len(raw) > MAX_RECEIPT_BYTES:
        raise RuntimeError("release receipt is unexpectedly large")
    digest = hashlib.sha256(raw).hexdigest()
    if receipt.suffix != ".json" or receipt.stem != digest:
        raise RuntimeError("release receipt filename is not its content hash")
    try:
        body = json.loads(raw, object_pairs_hook=_receipt_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("release receipt is invalid JSON") from exc
    if not isinstance(body, dict) or raw != _canonical_json(body) + b"\n":
        raise RuntimeError("release receipt bytes are not canonical JSON")
    identity = body.get("release_identity")
    revision = identity.get("source_revision") if isinstance(identity, dict) else None
    if not isinstance(revision, str) or REVISION_RE.fullmatch(revision) is None:
        raise RuntimeError("release receipt has no valid source revision")

    output_directory.mkdir(parents=True, exist_ok=False)
    snapshot = output_directory / receipt.name
    with snapshot.open("xb") as handle:
        handle.write(raw)
    snapshot.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return snapshot, body, digest


def _verify_frozen_release(
    *,
    repo: Path,
    release_root: Path,
    release_receipt: Path,
    verifier_root: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    original_receipt = Path(release_receipt)
    with tempfile.TemporaryDirectory(prefix="gkm-receipt-snapshot-") as tmp_name:
        snapshot, receipt_body, receipt_sha256 = _snapshot_release_receipt(
            source=original_receipt,
            output_directory=Path(tmp_name) / "receipt",
        )
        command = [
            sys.executable,
            "arc/crack_lab/verify_frozen_release.py",
            "--canonical-root",
            str(release_root),
            "--receipt",
            str(snapshot),
        ]
        if verifier_root is not None:
            command.extend(["--verifier-root", str(verifier_root.resolve())])
        result = _run_json(command, cwd=repo)

    identity = receipt_body["release_identity"]
    if (
        result.get("receipt_sha256") != receipt_sha256
        or result.get("verification_context_source_revision")
        != identity["source_revision"]
    ):
        raise RuntimeError(
            "schema-v2 frozen release verification identity did not match receipt"
        )
    _release_verification_summary(result, receipt_sha256)
    result["receipt"] = _portable_path(original_receipt, repo)
    return result, receipt_body


def _release_verification_summary(
    verification: dict[str, Any], receipt_sha256: str
) -> dict[str, Any]:
    """Normalize the two release-gate summary shapes without weakening either.

    Historical partial receipts report ``claimed_levels`` plus the complete
    authoritative denominator.  A complete receipt has no claimed/unclaimed
    overlay and reports ``levels`` directly.  Keep those upstream summaries
    unchanged in generated evidence, but derive the counts used by the
    manuscript from their verified shape instead of freezing the v2 count.
    """
    if verification.get("status") != "PASS":
        raise RuntimeError("schema-v2 frozen release did not verify")
    games = verification.get("games")
    if type(games) is not int or games != 25:
        raise RuntimeError("schema-v2 frozen release did not verify")

    kind = verification.get("kind")
    if kind == "partial_campaign_freeze":
        claimed = verification.get("claimed_levels")
        authoritative = verification.get("authoritative_levels")
        unclaimed = verification.get("unclaimed_boundaries")
        if (
            type(authoritative) is not int
            or authoritative != 183
            or type(claimed) is not int
            or claimed != 181
            or unclaimed != FROZEN_V2_UNCLAIMED_BOUNDARIES
        ):
            raise RuntimeError("schema-v2 frozen release did not verify")
        authority = "schema-v2 partial-release receipt verification"
    elif kind is None:
        claimed = verification.get("levels")
        authoritative = claimed
        unclaimed = verification.get("unclaimed_boundaries", [])
        if type(claimed) is not int or claimed != 183 or unclaimed != []:
            raise RuntimeError("schema-v2 frozen release did not verify")
        authority = "schema-v2 complete-release receipt verification"
    else:
        raise RuntimeError("schema-v2 frozen release did not verify")

    return {
        "schema": 2,
        "verdict": "PASS",
        "authority": authority,
        "receipt_sha256": receipt_sha256,
        "claimed_boundaries": claimed,
        "unclaimed_boundaries": unclaimed,
    }


def _receipt_bound_audit_reports(
    verification: dict[str, Any], receipt_sha256: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build the compact audit views implied by one fully verified receipt."""
    release_summary = _release_verification_summary(
        verification, receipt_sha256
    )
    claimed = release_summary["claimed_boundaries"]
    taint_report = {
        "automated_verdict": "PASS",
        "canonical": {"verdict": "clean", "files": claimed, "hits": []},
        "frontier_scaffolds": {"verdict": "not_in_release"},
        "promotion_chains": {},
        "release_gate": verification,
    }
    action_boundary_report = {
        "verdict": "PASS",
        "checkpoints": claimed,
        "exact": claimed,
        "issues": [],
        "release_gate": verification,
    }
    action_protocol_report = {
        "verdict": "PASS",
        "boundaries": claimed,
        "release_gate": verification,
    }
    return (
        release_summary,
        taint_report,
        action_boundary_report,
        action_protocol_report,
    )


def _history_tree_identity(
    *,
    repo: Path,
    complete_release: bool,
    explicit_history_root: Path | None,
    history_revision: str,
    verification_context_source_revision: str | None,
) -> str:
    """Bind complete acquisition history to the receipt's verified Git tree."""
    if not complete_release:
        return (
            FROZEN_SOURCE_HISTORY_TREE_SHA1
            if explicit_history_root is not None
            else ""
        )

    verified_revision = verification_context_source_revision
    if (
        not isinstance(verified_revision, str)
        or REVISION_RE.fullmatch(verified_revision) is None
    ):
        raise RuntimeError("complete release has no verified source revision")
    if history_revision != verified_revision:
        raise RuntimeError(
            "complete release history revision must equal the receipt's "
            "verified source revision"
        )
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "rev-parse",
            f"{verified_revision}:{FROZEN_HISTORY_TREE}",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_git_environment(),
        timeout=180,
    )
    expected = completed.stdout.strip()
    if completed.returncode != 0 or GIT_TREE_SHA1_RE.fullmatch(expected) is None:
        raise RuntimeError(
            "cannot derive receipt-bound acquisition-history Git tree"
        )
    return expected


def _source_audit_scope(
    receipt: dict[str, Any], payload: dict[str, Any]
) -> dict[str, Any]:
    claimed = receipt.get("claimed_inventory")
    if claimed is None:
        claimed = receipt.get("inventory")
    if not isinstance(claimed, dict):
        raise RuntimeError("release receipt has no claimed or complete inventory")
    endpoint_ids = {
        (str(game), level)
        for game, reached in claimed.items()
        for level in range(1, int(reached) + 1)
    }
    source_rows = [
        row
        for row in payload.get("rows", [])
        if row.get("system") == "GKM" and row.get("source_checkpoint_exact") is True
    ]
    source_ids = {
        (str(row["game"]), int(row["completed_level"])) for row in source_rows
    }
    if len(source_ids) != len(source_rows):
        raise RuntimeError("GKM source audit contains duplicate boundary rows")
    unexpected = sorted(source_ids - endpoint_ids)
    exclusions = sorted(endpoint_ids - source_ids)
    if unexpected:
        raise RuntimeError(f"source audit contains unclaimed endpoints: {unexpected}")
    if tuple(exclusions) != tuple(FROZEN_SOURCE_AUDIT_EXCLUSIONS):
        raise RuntimeError(
            "frozen source-audit exclusions changed: "
            f"expected {list(FROZEN_SOURCE_AUDIT_EXCLUSIONS)}, found {exclusions}"
        )
    return {
        "replay_verified_endpoint_wins": len(endpoint_ids),
        "admissible_exact_winning_source_checkpoints": len(source_ids),
        "excluded_from_source_marginals": [
            {"game": game, "level": level} for game, level in exclusions
        ],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(payload: dict[str, Any]) -> dict[str, dict[str, int]]:
    systems = payload["summary"]["systems"]
    return {
        system: {
            field: int(systems[system].get(field, 0))
            for field in STAT_FIELDS
        }
        | {
            "sharp_drops_with_literal_reuse": len(
                systems[system].get("sharp_drops_with_literal_reuse", [])
            ),
            "uncoupled_sharp": (
                int(systems[system].get("sharp_half_or_more_marginal_drops", 0))
                - len(systems[system].get("sharp_drops_with_literal_reuse", []))
            ),
        }
        for system in SYSTEMS
    }


def _external_mode(args: argparse.Namespace) -> bool:
    values = (
        args.opine_artifacts,
        args.baseline_release,
        args.baseline_repo,
        args.retrodict_runs,
    )
    if any(values) and not all(values):
        raise SystemExit(
            "raw comparator reproduction requires --opine-artifacts, "
            "--baseline-release, --baseline-repo, and --retrodict-runs together"
        )
    return all(values)


def _write_generated_stats(
    summary: dict[str, dict[str, int]],
    payload: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = output_dir / "comparator_stats.tex"
    tex_lines = [
        "% Generated by scripts/reproduce_manuscript.py; do not edit.",
    ]
    for system in SYSTEMS:
        for field, value in summary[system].items():
            tex_lines.append(
                rf"\newcommand{{\{MACRO_PREFIX[system]}{MACRO_FIELD[field]}}}"
                rf"{{{value}}}"
            )
    coupled = [
        (system, row)
        for system in ("GKM", "OPINE")
        for row in payload["summary"]["systems"][system][
            "sharp_drops_with_literal_reuse"
        ]
    ]
    coupled_clauses: list[str] = []
    for system, row in coupled:
        names = [
            str(item).split(":", 1)[-1]
            for item in row["reused_world_model_literals"]
        ]
        if len(names) == 1:
            called = rf"\path{{{names[0]}}}"
        else:
            called = ", ".join(rf"\path{{{name}}}" for name in names[:-1])
            called += rf", and \path{{{names[-1]}}}"
        coupled_clauses.append(
            rf"{system} \texttt{{{row['game']}}} L{row['completed_level']}, "
            rf"${row['previous_marginal_ast_zlib_bytes']}\!\to\!"
            rf"{row['marginal_ast_zlib_bytes']}$, calling unchanged {called}"
        )
    tex_lines.append(
        r"\newcommand{\CoupledWitnessEnumeration}{"
        + "; ".join(coupled_clauses)
        + "}"
    )
    tex_path.write_text("\n".join(tex_lines) + "\n")

    md_path = output_dir / "comparator_stats.md"
    md_lines = [
        "<!-- Generated by scripts/reproduce_manuscript.py; do not edit. -->",
        "",
        "| System | Retained source checkpoints | Exact winning-source checkpoints admitted by audit | "
        "Exact adjacent | Comparable marginals | Decreases | Sharp drops | "
        "Hard reuse | Sharp + reuse |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for system in SYSTEMS:
        row = summary[system]
        retained = row["retained_or_winning_checkpoints"]
        if system == "Retrodict":
            retained = (
                f"{retained} memory "
                f"({row['released_memory_transitions']} transitions)"
            )
        md_lines.append(
            f"| {system} | {retained} | {row['exact_winning_checkpoints']} | "
            f"{row['exact_adjacent_transitions']} | "
            f"{row['transitions_with_level_to_level_marginal_comparison']} | "
            f"{row['marginal_decreases']} | "
            f"{row['sharp_half_or_more_marginal_drops']} | "
            f"{row['hard_literal_world_model_reuse_witnesses']} | "
            f"{row['sharp_drops_with_literal_reuse']} |"
        )
    opine_sharp = sorted(
        (
            row
            for row in payload["rows"]
            if row["system"] == "OPINE" and row["sharp_marginal_drop"]
        ),
        key=lambda row: (row["game"], row["completed_level"]),
    )
    if len(opine_sharp) != summary["OPINE"][
        "sharp_half_or_more_marginal_drops"
    ]:
        raise RuntimeError("OPINE sharp-drop row count disagrees with summary")
    md_lines.extend(
        [
            "",
            "## OPINE sharp conditional drops",
            "",
            "| Boundary | Conditional AST marginal | Winning policy | Coupled "
            "direct-call witness |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for row in opine_sharp:
        policy = str(row["winning_policy_kind"]).replace("_", " ")
        md_lines.append(
            f"| `{row['game']}` L{row['completed_level']} | "
            f"{row['previous_level_marginal_ast_zlib_bytes']} → "
            f"{row['marginal_ast_zlib_bytes']} | {policy} | "
            f"{'yes' if row['sharp_drop_with_literal_reuse'] else 'no'} |"
        )
    md_lines.extend(
        [
            "",
            "Only synthesized-planner rows can certify the released executable "
            "winning path. The transient analyzer policies were not retained.",
        ]
    )
    md_path.write_text("\n".join(md_lines) + "\n")
    return tex_path, md_path


def _add_system_specific_stats(
    summary: dict[str, dict[str, int]],
    *,
    opine_json: Path,
    baseline_json: Path,
    retrodict_json: Path,
) -> None:
    opine = json.loads(opine_json.read_text())["summary"]
    summary["OPINE"]["trace_solve_events"] = int(
        opine["positive_reward_solve_events_in_logs"]
    )
    summary["OPINE"]["analyzer_or_unknown_wins"] = int(
        opine["analyzer_or_unknown_solved_checkpoints"]
    )

    baseline = json.loads(baseline_json.read_text())
    per_game: dict[str, int] = {}
    for row in baseline["rows"]:
        if row.get("profile") != "core" or row.get("completed_levels") is None:
            continue
        game = str(row["game"])
        per_game[game] = max(per_game.get(game, 0), int(row["completed_levels"]))
    summary["baseline1"]["reported_clears"] = sum(per_game.values())
    summary["baseline1"]["exact_authored_contractions"] = int(
        baseline["summary"]["profiles"]["authored"][
            "exact_adjacent_source_and_ast_contractions"
        ]
    )

    retrodict = json.loads(retrodict_json.read_text())["summary"]
    summary["Retrodict"]["memory_contractions"] = int(
        retrodict["contractions"]
    )


def _add_joint_row_stats(
    summary: dict[str, dict[str, int]], payload: dict[str, Any],
) -> None:
    exact_baseline = [
        row
        for row in payload["rows"]
        if row["system"] == "baseline1" and row["exact_adjacent_transition"]
    ]
    kinds = {
        "direct_literal_wins": "direct_literal_action",
        "inline_literal_wins": "inline_literal_action_program",
        "executor_literal_wins": "literal_action_program_via_executor",
    }
    for field, kind in kinds.items():
        summary["baseline1"][field] = sum(
            row["winning_policy_kind"] == kind for row in exact_baseline
        )


def main() -> int:
    args = _args()
    repo = args.repo_root.resolve()
    arc = repo / "arc"
    manuscript = arc / "manuscript"
    audits = arc / "audit_results"
    tracked_joint = audits / "marginal-literal-reuse.json"
    tracked_opine = audits / "opine-solved-checkpoints.json"
    tracked_baseline = audits / "baseline1_gpt55_xhigh_solved_checkpoints.json"
    tracked_retrodict = audits / "retrodict-solved-checkpoint-memory.json"
    frozen_release = arc / "crack_lab" / "releases" / "arc_agi3_gkm_v2_181"
    release_root = (args.release_root or frozen_release / "artifacts").resolve()
    acquisition_root = (
        args.acquisition_root or frozen_release / "acquisition_source"
    ).resolve()
    explicit_history_root = (
        args.history_root.resolve() if args.history_root is not None else None
    )
    release_receipt = args.release_receipt.resolve() if args.release_receipt else None
    expected_payload = json.loads(tracked_joint.read_text())
    expected_summary = _summary(expected_payload)
    _add_system_specific_stats(
        expected_summary,
        opine_json=tracked_opine,
        baseline_json=tracked_baseline,
        retrodict_json=tracked_retrodict,
    )
    _add_joint_row_stats(expected_summary, expected_payload)
    raw_mode = _external_mode(args)

    with tempfile.TemporaryDirectory(prefix="gkm-reproduce-") as tmp_name:
        tmp = Path(tmp_name)
        release_verification = None
        release_receipt_body = None
        verified_release_shape = None
        if release_receipt is not None:
            release_verification, release_receipt_body = _verify_frozen_release(
                repo=repo,
                release_root=release_root,
                release_receipt=release_receipt,
                verifier_root=args.release_verifier_root,
            )
            verified_release_shape = _release_verification_summary(
                release_verification, release_receipt.stem
            )
        complete_release = bool(
            verified_release_shape
            and verified_release_shape["authority"]
            == "schema-v2 complete-release receipt verification"
        )
        authenticated_history_tree = _history_tree_identity(
            repo=repo,
            complete_release=complete_release,
            explicit_history_root=explicit_history_root,
            history_revision=args.history_revision,
            verification_context_source_revision=(
                release_verification.get("verification_context_source_revision")
                if release_verification is not None
                else None
            ),
        )
        if explicit_history_root is None:
            history_root = _materialize_history(
                repo=repo,
                revision=args.history_revision,
                output=tmp / "frozen_history",
            )
            history_authority = f"git:{args.history_revision}:{FROZEN_HISTORY_TREE}"
            if authenticated_history_tree:
                history_authority += f":tree:{authenticated_history_tree}"
        else:
            history_root = _copy_authenticated_history(
                source=explicit_history_root,
                output=tmp / "frozen_history",
                expected_tree_sha1=authenticated_history_tree,
            )
            history_authority = "git-tree:" + authenticated_history_tree
        joint_out = tmp / "marginal-literal-reuse.json"
        taint_out = tmp / "canonical-taint-audit.json"
        action_boundary_out = tmp / "canonical-action-boundaries.json"
        action_protocol_out = tmp / "canonical-action-protocol-audit.json"
        if release_receipt is not None:
            assert release_verification is not None
            (
                release_summary,
                taint_report,
                action_boundary_report,
                action_protocol_report,
            ) = _receipt_bound_audit_reports(
                release_verification, release_receipt.stem
            )
            taint_out.write_text(json.dumps(taint_report, indent=2) + "\n")
            action_boundary_out.write_text(
                json.dumps(action_boundary_report, indent=2) + "\n"
            )
            action_protocol_out.write_text(
                json.dumps(action_protocol_report, indent=2) + "\n"
            )
        else:
            release_summary = None
            taint_command = [
                sys.executable,
                "arc/audit_submission_taint.py",
                str(release_root),
                "--json",
                str(taint_out),
            ]
            if args.require_complete_lineage:
                taint_command.append("--require-complete-lineage")
            _run(taint_command, cwd=repo)
            taint_report = json.loads(taint_out.read_text())
            if taint_report.get("automated_verdict") != "PASS":
                raise RuntimeError("canonical taint/promotion-chain audit did not pass")
            action_boundary_command = [
                sys.executable,
                "arc/audit_action_boundaries.py",
                str(release_root),
                "--json",
                str(action_boundary_out),
                "--summary-only",
            ]
            if args.require_complete_lineage:
                action_boundary_command.append("--require-complete-chain")
            _run(action_boundary_command, cwd=repo)
            action_boundary_report = json.loads(action_boundary_out.read_text())
            if action_boundary_report.get("verdict") != "PASS":
                raise RuntimeError("canonical exact action-boundary audit did not pass")
            action_protocol_command = [
                sys.executable,
                "arc/audit_action_protocol.py",
                str(release_root),
                "--json",
                str(action_protocol_out),
            ]
            _run(action_protocol_command, cwd=repo)
            action_protocol_report = json.loads(action_protocol_out.read_text())
            if action_protocol_report.get("verdict") != "PASS":
                raise RuntimeError("canonical action-protocol audit did not pass")

        if raw_mode:
            opine_json = tmp / "opine-solved-checkpoints.json"
            baseline_json = tmp / "baseline1-solved-checkpoints.json"
            retrodict_prefix = tmp / "retrodict-solved-checkpoint-memory"
            _run(
                [
                    sys.executable,
                    "arc/audit_opine_solved_checkpoints.py",
                    str(args.opine_artifacts),
                    "--csv",
                    str(tmp / "opine-solved-checkpoints.csv"),
                    "--json",
                    str(opine_json),
                ],
                cwd=repo,
            )
            retrodict_json = retrodict_prefix.with_suffix(".json")
            _run(
                [
                    sys.executable,
                    "arc/audit_baseline1_artifacts.py",
                    str(args.baseline_release),
                    "--baseline-repo",
                    str(args.baseline_repo),
                    "--csv",
                    str(tmp / "baseline1-solved-checkpoints.csv"),
                    "--json",
                    str(baseline_json),
                ],
                cwd=repo,
            )
            _run(
                [
                    sys.executable,
                    "arc/audit_retrodict_artifacts.py",
                    str(args.retrodict_runs),
                    "--out-prefix",
                    str(retrodict_prefix),
                ],
                cwd=repo,
            )
            _run(
                [
                    sys.executable,
                    "arc/audit_marginal_literal_reuse.py",
                    "--gkm-root",
                    str(history_root),
                    "--opine-root",
                    str(args.opine_artifacts),
                    "--opine-audit-json",
                    str(opine_json),
                    "--baseline-release",
                    str(args.baseline_release),
                    "--baseline-repo",
                    str(args.baseline_repo),
                    "--baseline-audit-json",
                    str(baseline_json),
                    "--retrodict-audit-json",
                    str(retrodict_json),
                    "--json",
                    str(joint_out),
                ],
                cwd=repo,
            )
        else:
            opine_json = tracked_opine
            baseline_json = tracked_baseline
            retrodict_json = tracked_retrodict
            _run(
                [
                    sys.executable,
                    "arc/audit_marginal_literal_reuse.py",
                    "--gkm-root",
                    str(history_root),
                    "--reuse-non-gkm-from-json",
                    str(tracked_joint),
                    "--json",
                    str(joint_out),
                ],
                cwd=repo,
            )

        actual_payload = json.loads(joint_out.read_text())
        actual_summary = _summary(actual_payload)
        if release_receipt is None:
            source_audit_scope = None
        else:
            if release_receipt_body is None:
                raise RuntimeError("verified release receipt body was not retained")
            source_audit_scope = _source_audit_scope(
                release_receipt_body, actual_payload
            )
        _add_system_specific_stats(
            actual_summary,
            opine_json=opine_json,
            baseline_json=baseline_json,
            retrodict_json=retrodict_json,
        )
        _add_joint_row_stats(actual_summary, actual_payload)
        generated_stats = _write_generated_stats(
            actual_summary, actual_payload, tmp / "generated",
        )
        comparator_drift = {
            system: {
                field: [expected_summary[system][field], actual_summary[system][field]]
                for field in actual_summary[system]
                if expected_summary[system][field] != actual_summary[system][field]
            }
            for system in SYSTEMS
        }
        comparator_drift = {
            system: drift for system, drift in comparator_drift.items() if drift
        }
        non_gkm_drift = {
            system: drift
            for system, drift in comparator_drift.items()
            if system != "GKM"
        }
        if non_gkm_drift:
            raise RuntimeError(f"comparator audit drift: {non_gkm_drift}")
        if "GKM" in comparator_drift and not (
            args.allow_live_gkm_drift or args.write
        ):
            raise RuntimeError(
                "live GKM audit differs from the manuscript snapshot; "
                "use --write to refresh it or --allow-live-gkm-drift to inspect: "
                f"{comparator_drift['GKM']}"
            )

        figure_dir = tmp / "figures"
        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", str(tmp / "matplotlib"))
        env.setdefault("XDG_CACHE_HOME", str(tmp / "cache"))
        command = [
            sys.executable,
            "scripts/generate_figures.py",
            "--output-dir",
            str(figure_dir),
            "--solutions-dir",
            str(acquisition_root),
        ]
        print("+", " ".join(command))
        subprocess.run(command, cwd=manuscript, env=env, check=True)

        generated_dir = tmp / "generated"
        rst_out = tmp / "marginal_complexity_by_level.rst"
        _run(
            [
                sys.executable,
                "scripts/generate_empirical_tables.py",
                "--solutions-dir",
                str(acquisition_root),
                "--output-dir",
                str(generated_dir),
                "--rst-output",
                str(rst_out),
            ],
            cwd=manuscript,
        )
        generated_tables = tuple(
            sorted(generated_dir.glob("marginal_complexity_by_level.*"))
        )

        if args.write:
            shutil.copy2(joint_out, tracked_joint)
            for path in figure_dir.iterdir():
                shutil.copy2(path, manuscript / "figures" / path.name)
            tracked_generated = manuscript / "generated"
            tracked_generated.mkdir(exist_ok=True)
            for path in generated_stats:
                shutil.copy2(path, tracked_generated / path.name)
            for path in generated_tables:
                shutil.copy2(path, tracked_generated / path.name)
            docs_generated = repo / "docs" / "generated"
            docs_generated.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rst_out, docs_generated / rst_out.name)
            shutil.copy2(taint_out, tracked_generated / taint_out.name)
            shutil.copy2(
                action_boundary_out, tracked_generated / action_boundary_out.name
            )
            shutil.copy2(
                action_protocol_out,
                tracked_generated / action_protocol_out.name,
            )

        report = {
            "mode": "raw-external-artifacts" if raw_mode else "cached-comparators",
            "summary": actual_summary,
            "drift_from_tracked_snapshot": comparator_drift,
            "inputs": {
                "tracked_joint_audit_sha256": _sha256(tracked_joint),
                "opine_artifacts": str(args.opine_artifacts or ""),
                "baseline_release": str(args.baseline_release or ""),
                "baseline_repo": str(args.baseline_repo or ""),
                "retrodict_runs": str(args.retrodict_runs or ""),
                "release_root": _portable_path(release_root, repo),
                "acquisition_root": _portable_path(acquisition_root, repo),
                "history_root": (
                    _portable_path(explicit_history_root, repo)
                    if explicit_history_root is not None
                    else None
                ),
                "history_authority": history_authority,
                "release_receipt": (
                    _portable_path(release_receipt, repo)
                    if release_receipt is not None
                    else ""
                ),
                "release_verifier_root": str(
                    args.release_verifier_root.resolve()
                    if args.release_verifier_root is not None
                    else "local-git:receipt-source-revision"
                ),
            },
            "taint_and_lineage": {
                "automated_verdict": taint_report["automated_verdict"],
                "canonical_verdict": taint_report["canonical"]["verdict"],
                "canonical_files_scanned": taint_report["canonical"]["files"],
                "canonical_hits": len(taint_report["canonical"]["hits"]),
                "frontier_scaffold_verdict": taint_report[
                    "frontier_scaffolds"
                ]["verdict"],
                "promotion_chains_checked": len(
                    taint_report["promotion_chains"]
                ) if release_receipt is None else 25,
                "promotion_chain_failures": (
                    sum(
                        chain["verdict"] != "clean"
                        for chain in taint_report["promotion_chains"].values()
                    )
                    if release_receipt is None
                    else 0
                ),
                "complete_lineage_required": args.require_complete_lineage,
                "audit_sha256": _sha256(taint_out),
            },
            "action_boundaries": {
                "verdict": action_boundary_report["verdict"],
                "checkpoints": action_boundary_report["checkpoints"],
                "exact": action_boundary_report["exact"],
                "issues": len(action_boundary_report["issues"]),
                "audit_sha256": _sha256(action_boundary_out),
            },
            "action_protocol": {
                "verdict": action_protocol_report["verdict"],
                "audit_sha256": _sha256(action_protocol_out),
            },
            "release_gate": release_summary,
            "source_audit_scope": source_audit_scope,
            "generated": {
                path.name: _sha256(path)
                for path in sorted(
                    [
                        *figure_dir.iterdir(),
                        *generated_stats,
                        *generated_tables,
                        rst_out,
                        taint_out,
                        action_boundary_out,
                        action_protocol_out,
                    ],
                    key=lambda item: item.name,
                )
            },
        }
        report_path = args.report or (
            manuscript / "reproduction_report.json"
            if args.write
            else tmp / "reproduction_report.json"
        )
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))

    _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "arc/test_audit_marginal_literal_reuse.py",
            "arc/manuscript/scripts/test_generate_figures.py",
            "arc/manuscript/scripts/test_generate_empirical_tables.py",
            "arc/manuscript/scripts/test_reproduce_manuscript.py",
        ],
        cwd=repo,
    )
    if args.build_paper:
        _run(["make", "paper"], cwd=manuscript)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
