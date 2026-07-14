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
import importlib.util
import ast
import fcntl
import glob
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import gkm_arena as A
from gkm_solve_agent import discovered_context

# Working-directory root for per-game leg workspaces. Defaults to a repo-relative
# ``runs/scratch`` dir; override with the ``GKM_SCRATCH`` environment variable.
from pathlib import Path as _Path
SCRATCH = os.environ.get(
    "GKM_SCRATCH",
    str(_Path(__file__).resolve().parent / "runs" / "scratch"),
)


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


def free_energy(levels: int, marginal_C_total: int, lam: float = 0.02) -> float:
    """F = R + lambda*C with R = -levels_reached and C = total marginal novelty."""
    return -float(levels) + lam * float(marginal_C_total)


CHECKPOINT_FILE = "checkpoint.json"
"""Filename for per-level marginal-C checkpoint (enables cross-run resume)."""

AUTO_SOLVE_LOG = "auto_solve_attempts.json"
"""Per-level record of failed auto-solve attempts, keyed by (level, legs-hash), so a
relaunch does not re-pay a long BFS that already failed against the same legs."""

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
) + ARC_GAME_SOURCE_NAMES

PRIVATE_RUNTIME_RE = re.compile(
    r"\.\s*_(?:game|env|fd|budget)\b"
    r"|__dict__|inspect\.(?:getsource|getmembers)"
    r"|\b(?:vars|dir)\s*\("
    r"|object\.__getattribute__"
    r"|\b(?:getattr|hasattr)\s*\([^,\n]+,\s*['\"]_(?:game|env|fd|budget)\b"
)
"""Strings that make a proposer workspace inadmissible.

The arena may execute the hidden game implementation internally, but the
proposer must not inspect source files or earlier solution history. If any
agent-authored workspace file records such an access, the harness refuses to
verify or promote the attempt.
"""

PROMOTED_FILES = ("legs.py", "players.py", "solve.py", "legs_log.md", CHECKPOINT_FILE,
                  AUTO_SOLVE_LOG)
"""Files that define a verified leg-library state and should survive scratch loss."""

SNAPSHOT_SKIP_DIRS = {"__pycache__", ".pytest_cache"}
SNAPSHOT_SKIP_FILES = {".orchestrate.lock"}
BLOCKED_ATTEMPTS_LOG = "blocked_attempts.log"


class WorkspaceTainted(RuntimeError):
    """The proposer workspace contains evidence of forbidden source/history use."""


def _file_taint_reason(path: str, display_name: str) -> Optional[str]:
    try:
        if os.path.getsize(path) > 2_000_000:
            return None
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().lower()
    except OSError:
        return None
    if os.path.basename(path) == "proposer_last.log":
        # Debriefs may quote a blocked command as Markdown inline code. The
        # blocked-attempt ledger is the execution record; a quoted mention in
        # prose is not evidence that introspection ran.
        text = re.sub(r"`[^`\n]*`", "", text)
    if PRIVATE_RUNTIME_RE.search(text):
        return f"private game/runtime introspection in {display_name}"
    for marker in SOURCE_TAINT_MARKERS:
        if marker in text:
            return f"{marker} in {display_name}"
    return None


def _workspace_taint_reason(ws: str) -> Optional[str]:
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


def promoted_artifact_taint_reason(art: str) -> Optional[str]:
    """Scan canonical promoted evidence without reclassifying forensic WIP."""
    for name in PROMOTED_FILES:
        reason = _file_taint_reason(os.path.join(art, name), name)
        if reason:
            return reason
    return None


def assert_workspace_not_tainted(ws: str) -> None:
    reason = _workspace_taint_reason(ws)
    if reason:
        raise WorkspaceTainted(
            f"forbidden source/history access tainted proposer workspace: {reason}"
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
        rep.total_marginal_C = max(0, rep.total_marginal_C - removed_cost)
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
    with open(os.path.join(ws, CHECKPOINT_FILE), "w") as f:
        json.dump(data, f)


def _load_checkpoint(ws: str) -> Optional[Report]:
    """Restore a checkpoint from a previous run, or None."""
    path = os.path.join(ws, CHECKPOINT_FILE)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    rep = Report(
        game=data["game"],
        reached=data["reached"],
        total_marginal_C=data["total_marginal_C"],
        records=[LevelRecord(**r) for r in data.get("records", [])],
        final_path=data.get("final_path", []),
        validated=data.get("validated", False),
    )
    _deduplicate_level_records(rep)
    return rep


def _acquire_workspace_lock(ws: str):
    """Hold an exclusive process lock for one orchestrator per scratch workspace."""
    path = os.path.join(ws, ".orchestrate.lock")
    lock = open(path, "a+")
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


def _release_workspace_lock(lock) -> None:
    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    lock.close()


def _adopt_workspace_checkpoint(game: str, ws: str, rep: Report,
                                verbose: bool = True) -> Report:
    """Accept a proposer-updated checkpoint only after independent replay validation."""
    ws_rep = _load_checkpoint(ws)
    if ws_rep is None or ws_rep.game != game:
        return rep
    if ws_rep.reached < rep.reached or not ws_rep.final_path:
        return rep
    if not A.validate(game, ws_rep.final_path, ws_rep.reached):
        return rep
    if (ws_rep.reached, len(ws_rep.final_path)) == (rep.reached, len(rep.final_path)):
        return rep
    if verbose:
        print(f"adopted proposer checkpoint: reached={ws_rep.reached} "
              f"path_len={len(ws_rep.final_path)}")
    return ws_rep


def artifact_dir(game: str, tag: str = "") -> str:
    """Stable, repo-visible storage for the latest verified leg-library artifact."""
    labdir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(labdir, "agent_solutions", f"{game}_legs")


def _wip_level_dir(art: str, level: int) -> str:
    return os.path.join(art, "wip_context", f"level_{level:02d}")


def _workspace_snapshot_files(ws: str) -> List[str]:
    files = []
    for name in sorted(os.listdir(ws)):
        if name in SNAPSHOT_SKIP_FILES:
            continue
        path = os.path.join(ws, name)
        if os.path.isdir(path):
            if name in SNAPSHOT_SKIP_DIRS:
                continue
            continue
        if os.path.isfile(path):
            files.append(name)
    return files


def _snapshot_digest(ws: str, phase: str, names: List[str]) -> str:
    h = hashlib.sha256()
    h.update(phase.encode("utf-8"))
    for name in names:
        path = os.path.join(ws, name)
        h.update(name.encode("utf-8"))
        with open(path, "rb") as f:
            h.update(f.read())
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
    art = artifact_dir(game, tag)
    os.makedirs(art, exist_ok=True)
    names = _workspace_snapshot_files(ws)
    digest = _snapshot_digest(ws, phase, names)
    level_dir = _wip_level_dir(art, level)
    attempt = f"{phase}_{digest}"
    attempt_dir = os.path.join(level_dir, attempt)
    files_dir = os.path.join(attempt_dir, "files")
    os.makedirs(files_dir, exist_ok=True)
    for name in names:
        shutil.copy2(os.path.join(ws, name), os.path.join(files_dir, name))
    meta = {
        "game": game,
        "level": level,
        "phase": phase,
        "reached": reached,
        "err": err,
        "attempt": attempt,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": names,
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


def _restore_wip_probes(game: str, ws: str, level: int, tag: str = "",
                        verbose: bool = True) -> int:
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
    if not os.path.exists(latest_path):
        return 0
    try:
        with open(latest_path) as f:
            latest_attempt = json.load(f).get("attempt")
    except Exception:
        return 0

    attempts = []
    for attempt in sorted(os.listdir(level_dir)):
        attempt_dir = os.path.join(level_dir, attempt)
        files_dir = os.path.join(attempt_dir, "files")
        meta_path = os.path.join(attempt_dir, "metadata.json")
        if not os.path.isdir(files_dir):
            continue
        created = ""
        try:
            with open(meta_path) as f:
                created = json.load(f).get("created_at", "")
        except Exception:
            pass
        attempts.append((attempt == latest_attempt, created, attempt, files_dir))
    if not attempts:
        return 0
    attempts.sort(key=lambda t: t[1], reverse=True)
    attempts.sort(key=lambda t: not t[0])
    skip = set(PROMOTED_FILES) | {"gkm_try.py"}
    restored = 0
    latest_done = False
    for is_latest, _, attempt, files_dir in attempts:
        for name in sorted(os.listdir(files_dir)):
            if name in skip:
                continue
            src = os.path.join(files_dir, name)
            dst = os.path.join(ws, name)
            if os.path.exists(dst):
                if latest_done:
                    continue
                if os.path.getmtime(dst) >= os.path.getmtime(src):
                    continue
            shutil.copy2(src, dst)
            restored += 1
        latest_done = True
    if verbose and restored:
        print(f"restored {restored} WIP probe file(s) for level {level} "
              f"from latest/backfill snapshots")
    return restored


def seed_workspace_from_artifact(game: str, ws: str, tag: str = "", verbose: bool = True,
                                 restore_wip: bool = True) -> Optional[Report]:
    """Overwrite scratch with the latest promoted verified state, if one exists.

    Scratch is treated as disposable and possibly contaminated by an unfinished next
    level. The repo artifact is the source of truth for resuming. Unverified probe
    context for the NEXT level can be restored alongside (fill-gaps-only), so an
    interrupted attempt's probes survive scratch loss without contaminating the
    verified files. Set restore_wip=False for a clean continuation that retains
    only the verified Kolmogorov-Schmidhuber backbone.
    """
    art = artifact_dir(game, tag)
    rep = _load_checkpoint(art)
    if rep is None or not rep.validated:
        if restore_wip:
            _restore_wip_probes(game, ws, 1, tag, verbose=verbose)
        return None
    for name in PROMOTED_FILES:
        src = os.path.join(art, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(ws, name))
    if restore_wip:
        _restore_wip_probes(game, ws, rep.reached + 1, tag, verbose=verbose)
    if verbose:
        print(f"seeded workspace from artifact: {art} (reached={rep.reached})")
    return rep


def promote_verified_artifact(game: str, ws: str, rep: Report, tag: str = "", verbose: bool = True) -> bool:
    """Idempotently publish the latest replay-validated workspace state.

    Promotion is intentionally gated on replay validation. This prevents speculative
    edits for an unfinished next level from replacing the last known-good artifact.
    """
    if not rep.validated or rep.reached <= 0:
        return False
    assert_workspace_not_tainted(ws)
    art = artifact_dir(game, tag)
    old = _load_checkpoint(art)
    if old is not None and old.validated and old.reached > rep.reached:
        if verbose:
            print(f"kept artifact at level {old.reached}; current verified level {rep.reached} is older")
        return False
    promote_files = old is None or not old.validated or old.reached < rep.reached
    os.makedirs(art, exist_ok=True)
    _save_checkpoint(ws, rep)
    if promote_files:
        for name in PROMOTED_FILES:
            src = os.path.join(ws, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(art, name))
    else:
        # Same verified level: refresh metadata only. Scratch may contain
        # speculative next-level code, so do not overwrite clean solution files.
        src = os.path.join(ws, CHECKPOINT_FILE)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(art, CHECKPOINT_FILE))
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
import gkm_arena as A
taint_reason = G._workspace_taint_reason(os.getcwd())
if taint_reason:
    raise SystemExit(f"TAINTED WORKSPACE: {{taint_reason}}")
spec = importlib.util.spec_from_file_location("solve", "solve.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
def resumed_solve(env):
    ck = None
    if os.path.exists("checkpoint.json"):
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


def run_solve_file(game: str, solve_path: str):
    """Import solve(env) from the workspace and run it on the real game. The workspace
    dir must be on sys.path so solve.py's `import players` / `from legs import *`
    resolve, and cached modules are dropped so each call re-reads the agent's edits.
    (Default verifier.)"""
    import sys
    wsdir = os.path.dirname(os.path.abspath(solve_path))
    added = wsdir not in sys.path
    if added:
        sys.path.insert(0, wsdir)
    for name in ("solve", "players", "legs"):
        sys.modules.pop(name, None)
    try:
        spec = importlib.util.spec_from_file_location("solve", solve_path)
        m = importlib.util.module_from_spec(spec)
        sys.modules["solve"] = m
        spec.loader.exec_module(m)
        ckpt = _load_checkpoint(wsdir)

        def resumed_solve(env):
            if ckpt and ckpt.game == game and ckpt.validated and ckpt.final_path:
                for act in ckpt.final_path:
                    env.step(act)
            m.solve(env)

        return A.run_program(game, resumed_solve)
    finally:
        if added and wsdir in sys.path:
            sys.path.remove(wsdir)


def _candidate_path_files(ws: str, K: int) -> List[str]:
    """Likely action-path artifacts a proposer may have left outside solve.py.

    API/tool agents sometimes discover a winning path in a probe and write it to
    /tmp (or print it) but time out before integrating it into players.py. The
    harness owns verification, so it should harvest those candidate paths instead
    of treating the level as unsolved.
    """
    patterns = [
        os.path.join(ws, "base*.json"),
        os.path.join(ws, f"seg_L{K}.json"),
        os.path.join(ws, f"seg_{K}.json"),
        os.path.join(ws, f"*path*.json"),
        os.path.join(ws, f"win{K}*.json"),
        os.path.join(ws, f"*win*{K}*.json"),
        os.path.join("/tmp", f"win{K}*.json"),
        os.path.join("/tmp", f"*win*{K}*.json"),
    ]
    out = []
    for pat in patterns:
        out.extend(glob.glob(pat))
    # Newer first, de-duped.
    seen = set()
    ordered = []
    for path in sorted(out, key=lambda p: os.path.getmtime(p), reverse=True):
        rp = os.path.realpath(path)
        if rp not in seen:
            seen.add(rp)
            ordered.append(path)
    return ordered


def _load_action_path(value) -> Optional[list]:
    """Normalize JSON/log candidates into replayable key or coordinate actions."""
    if isinstance(value, dict):
        for key in ("path", "actions", "win", "solution"):
            if key in value:
                value = value[key]
                break
    if isinstance(value, list) and value:
        normalized = []
        for action in value:
            if isinstance(action, int) and 1 <= action <= 9:
                normalized.append(action)
            elif (isinstance(action, (list, tuple)) and len(action) == 3
                  and action[0] == 6
                  and all(isinstance(v, int) for v in action[1:])):
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
    candidates = []
    for path in _candidate_path_files(ws, K):
        try:
            value = json.load(open(path))
        except (OSError, json.JSONDecodeError):
            continue
        suffix = _load_action_path(value)
        if suffix:
            candidates.append((suffix, path))
    candidates.extend((p, "proposer_last.log") for p in _candidate_paths_from_log(ws))
    candidates.extend((p, CHECKPOINT_FILE) for p in _candidate_paths_from_checkpoint(ws))

    # Some successful WIP states naturally factor the replay into a compressed
    # verified prefix plus a next-level suffix. Harvest both halves without
    # forcing the proposer to remember to rewrite checkpoint.json before timeout.
    failed_glues = 0
    prefix_ok = {}
    for prefix_path, prefix_source in candidates:
        key = _action_path_key(prefix_path)
        if key not in prefix_ok:
            prefix_ok[key] = _validated_prefix_floor(game, prefix_path, K - 1)
        if not prefix_ok[key]:
            continue
        for suffix_path, suffix_source in candidates:
            if suffix_path is prefix_path:
                continue
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
    for name in ordered:
        stub = f"\n\ndef play_level_{K}(env):\n    {name}(env)\n"
        with open(players_p, "a") as f:
            f.write(stub)
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


def action_deltas(env, actions: Sequence[int] = ACTIONS) -> Dict[int, dict]:
    base = arr(env.frame()).copy()
    out = {}
    for action in actions:
        clone = env.clone()
        clone.step(action)
        out[int(action)] = frame_delta(base, clone.frame())
    return out


def replay(env, actions: Sequence[int]):
    clone = env.clone()
    for action in actions:
        if clone.terminal():
            break
        clone.step(int(action))
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


def bounded_bfs(env, goal_fn, actions: Sequence[int] = (UP, DOWN, LEFT, RIGHT, USE),
                key_fn=None, max_states: int = 20000, max_depth: int = 80):
    """Generic clone BFS over observational keys. Use small max_states first."""
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
        for action in actions:
            child = node.clone()
            child.step(int(action))
            key = key_fn(child)
            if key in seen:
                continue
            seen.add(key)
            q.append((child, path + [int(action)]))
    return None


def level_goal(base_level: int):
    return lambda env, path: env.levels_completed > base_level
'''


def setup_workspace(game: str, tag: str = "") -> str:
    suffix = f"_{tag}" if tag else ""
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


# ---------------------------------------------------------------------------
# default proposer: the real Claude Code agent (tools) -- needs credits
# ---------------------------------------------------------------------------
# markers that mean "no credits / rate-limited" -- the whole run should abort, not
# silently churn out empty proposals against a dead API.
_CREDIT_OUT_MARKERS = ("out of usage credits", "usage limit", "credit balance", "session limit",
                       "rate limit", "insufficient", "quota", "not logged in", "please run /login")


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


class CreditOut(RuntimeError):
    """Raised when the proposer subprocess reports it is out of credits/quota, so the
    orchestrator can stop the whole sequence cleanly instead of burning the budget."""


# markers of a transient infrastructure failure (dropped connection, server error):
# the proposer never worked on the level, so the attempt is retried, not judged.
_TRANSIENT_MARKERS = ("api error", "connection closed", "connection error", "connection refused",
                      "overloaded", "internal server error", "service unavailable")

_TRANSIENT_RETRIES = 2
"""Extra proposer attempts per level when the failure looks infrastructural."""


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


def _claude_agent(ws: str, task: str, model: Optional[str], minutes: int) -> None:
    labdir = os.path.dirname(os.path.abspath(__file__))
    cmd = ["claude", "-p", task, "--allowedTools", "Bash", "Read", "Write", "Edit",
           "--dangerously-skip-permissions", "--add-dir", labdir, "--output-format", "text"]
    if model:
        cmd += ["--model", model]
    out = err = ""
    try:
        r = subprocess.run(cmd, cwd=ws, capture_output=True, text=True, timeout=minutes * 60)
        out, err = r.stdout or "", r.stderr or ""
    except subprocess.TimeoutExpired as ex:
        # Out of the per-level time budget. Whatever the agent already wrote to the
        # workspace (legs.py/players.py) persists; let the loop verify that partial
        # work instead of crashing the whole run.
        out = (ex.stdout or b"").decode("utf-8", "replace") if isinstance(ex.stdout, bytes) else (ex.stdout or "")
        print(f"[proposer hit {minutes}min budget; verifying partial work]")
    # SAFEGUARD: persist the proposer's own output so a credit-out / crash is visible
    # (previously discarded). And if it reports no-credits, abort the whole sequence.
    with open(os.path.join(ws, "proposer_last.log"), "w") as fh:
        fh.write(out + ("\n--- STDERR ---\n" + err if err else ""))
    blob = (out + " " + err).lower()
    if any(m in blob for m in _CREDIT_OUT_MARKERS):
        raise CreditOut(f"proposer reported no credits/quota (see {ws}/proposer_last.log)")


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


def _codex_agent(ws: str, task: str, model: Optional[str], minutes: int) -> None:
    """Run Codex headlessly as the proposer, writing a transcript as it streams."""
    labdir = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        "codex", "exec",
        "--cd", ws,
        "--add-dir", labdir,
        "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox",
        "--color", "never",
        task,
    ]
    if model:
        cmd[2:2] = ["--model", model]
    log_path = os.path.join(ws, "proposer_last.log")
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, cwd=ws, stdout=log, stderr=subprocess.STDOUT,
                                text=True)
        try:
            proc.wait(timeout=minutes * 60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            print(f"[codex proposer hit {minutes}min budget; verifying partial work]")
        if proc.returncode not in (0, -9):
            txt = _read(log_path).lower()
            if any(m in txt for m in _CREDIT_OUT_MARKERS):
                raise CreditOut(f"codex reported no credits/quota (see {log_path})")
            raise RuntimeError(f"codex proposer CLI failed (see {log_path})")


def _api_agent(ws: str, task: str, model: Optional[str], minutes: int) -> None:
    """Run the Messages-API agentic loop (gkm_api_agent) as the proposer.

    Bills against ANTHROPIC_API_KEY Console credits -- a separate pool from the
    Claude Code CLI subscription, so it works as a credit-out fallback."""
    import gkm_api_agent
    _ensure_anthropic_api_key()
    try:
        gkm_api_agent.run_agent(ws, task, model=model, minutes=minutes)
    except Exception as ex:
        with open(os.path.join(ws, "proposer_last.log"), "w") as fh:
            fh.write(_redact_secrets(f"API Error: {type(ex).__name__}: {ex}\n"))
    blob = _read(os.path.join(ws, "proposer_last.log")).lower()
    if any(m in blob for m in _CREDIT_OUT_MARKERS):
        raise CreditOut(f"api proposer reported no credits/quota (see {ws}/proposer_last.log)")


def _propose_task(game, K, context, legs_index):
    return (A.PRECONCEPTIONS + "\n\n" + context +
            f"\n\nYou are growing a LEG LIBRARY across the levels of {game}. Existing "
            f"legs in legs.py: {legs_index or '(none yet)'}.\n"
            f"GOAL: make solve.py reach LEVEL {K}. First run `python gkm_try.py` to see "
            "where you are; solve.py dispatches to players.play_level_K. On a clone at "
            f"level {K}, learn its structure. Use perception.py first: it is a "
            "source-free scaffold that turns frames into blobs, object candidates, "
            "action deltas, replay summaries, and bounded clone BFS keys. "
            "Inspect env.actions: key actions are integers; coordinate-only games "
            "use env.step(6, x, y), recorded in replay paths as [6, x, y]. "
            f"Build small symbolic probes on top of those observations instead of repeatedly "
            f"dumping raw pixels. Then WRITE `play_level_{K}(env)` in "
            "players.py that ONLY COMPOSES legs imported from legs.py. REUSE existing "
            "legs wherever the level is an earlier one in a new configuration; add NEW "
            "legs to legs.py ONLY when nothing fits, and keep them minimal and general. "
            "Do not put level logic inline in the player -- put reusable skills in "
            "legs.py. Iterate with `python gkm_try.py` until RESULT shows "
            f"levels>={K}. Keep clone use bounded (~300 steps/s).")


def _debrief_task(game, K):
    return (f"DEBRIEF after clearing {game} level {K}. Compare play_level_{K} to the "
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


def orchestrate(game="wa30", max_level=9, model=None, minutes_per=40,
                proposer="claude", tag="",
                seed_artifact: bool = True,
                restore_wip: bool = True,
                propose_fn: Optional[Callable] = None,
                verify_fn: Optional[Callable] = None,
                debrief_fn: Optional[Callable] = None,
                verbose=True) -> Report:
    """Per-level compose->verify->debrief with marginal-C accounting.

    propose_fn(ws,K) / verify_fn(game, solve_path)->(levels,path,err) /
    debrief_fn(ws,K) are injectable; defaults use either the Claude Code agent
    (``proposer="claude"``) or opencode (``proposer="opencode"``) as proposer,
    and the real game as verifier (credits needed for either proposer).
    """
    ws = setup_workspace(game, tag)
    run_lock = _acquire_workspace_lock(ws)
    legs_p, players_p, solve_p = (os.path.join(ws, f) for f in ("legs.py", "players.py", "solve.py"))
    if seed_artifact:
        seed_workspace_from_artifact(game, ws, tag, verbose=verbose, restore_wip=restore_wip)
    elif verbose:
        print("fresh run requested: skipping artifact seed")
    context = discovered_context(game) if propose_fn is None else ""
    if propose_fn is None:
        agents = {"claude": _claude_agent, "opencode": _opencode_agent, "codex": _codex_agent, "api": _api_agent}
        _agent = agents[proposer]
        propose_fn = lambda w, k: _agent(
            w,
            _propose_task(game, k, context, _defs(_read(legs_p))),
            model,
            minutes_per,
        )
    if debrief_fn is None:
        agents = {"claude": _claude_agent, "opencode": _opencode_agent, "codex": _codex_agent, "api": _api_agent}
        _agent = agents[proposer]
        debrief_fn = lambda w, k: _agent(w, _debrief_task(game, k), model, max(10, minutes_per // 2))
    verify_fn = verify_fn or run_solve_file

    rep = Report(game=game, reached=0)
    # resume from checkpoint (restores marginal-C history across restarts)
    ckpt = _load_checkpoint(ws)
    if ckpt is not None:
        rep = ckpt
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
    try:
        signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    except ValueError:
        pass  # not in the main thread; Ctrl-C still covered
    try:
        while rep.reached < max_level:
            K = rep.reached + 1
            legs_b, players_b = _read(legs_p), _read(players_p)

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
                if verbose:
                    print(f"level {K}: auto-solved via existing legs")
                # still debrief: the auto-solve succeeded, but legs_log.md may need an entry
                try:
                    debrief_fn(ws, K)
                    levels2, path2, _ = verify_fn(game, solve_p)
                except CreditOut as ex:
                    print(f"CREDIT-OUT during debrief after level {K}: {ex}; preserving solved level")
                    levels2, path2 = levels, path
                snapshot_wip_context(
                    game, ws, K, "after_auto_solve_debrief",
                    max(levels, levels2), None, tag, verbose=verbose,
                )
                reached = max(levels, levels2)
                path = path2 if levels2 >= levels else path
                Cm = marginal_complexity(legs_b, _read(legs_p), players_b, _read(players_p))
                _record_level(rep, K, Cm)
                rep.reached = reached
                rep.final_path = path
                rep.validated = A.validate(game, rep.final_path, rep.reached) if rep.final_path else False
                _save_checkpoint(ws, rep)
                promote_verified_artifact(game, ws, rep, tag, verbose=verbose)
                if verbose:
                    print(f"level {K}: reached={reached} marginal_C={Cm} "
                          f"total_C={rep.total_marginal_C} validated={rep.validated} F={rep.free_energy:.3f}")
                if reached <= K - 1:
                    break
                continue

            # PHASE 1: proposer (existing legs could not solve the level). A transient
            # infrastructure failure (dropped connection, logged-out CLI that slipped
            # past the credit-out check) says nothing about the level, so it is retried;
            # only a real full-budget attempt that falls short stops the run.
            credit_out = False
            pre_recovered = recover_discovered_path_artifact(
                game, ws, K, rep.final_path, verbose=verbose)
            if pre_recovered is not None:
                levels, path, err = pre_recovered
                snapshot_wip_context(game, ws, K, "recovered_existing_path_artifact",
                                     levels, None, tag, verbose=verbose)
            for attempt in range(0 if pre_recovered is not None else 1 + _TRANSIENT_RETRIES):
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
                rep = _adopt_workspace_checkpoint(game, ws, rep, verbose=verbose)
                recovered = recover_discovered_path_artifact(
                    game, ws, K, rep.final_path, verbose=verbose)
                _save_checkpoint(ws, rep)
                if recovered is not None:
                    levels, path, err = recovered
                    snapshot_wip_context(game, ws, K, "recovered_path_artifact",
                                         levels, None, tag, verbose=verbose)
                    break
                snapshot_wip_context(game, ws, K, "after_propose", rep.reached, None, tag, verbose=verbose)
                levels, path, err = verify_fn(game, solve_p)
                if levels >= K:
                    break
                code_changed = (_read(legs_p) != legs_b or _read(players_p) != players_b)
                if attempt < _TRANSIENT_RETRIES and _transient_proposer_failure(ws, code_changed):
                    if verbose:
                        print(f"level {K}: transient proposer failure (see proposer_last.log); retrying")
                    continue
                snapshot_wip_context(game, ws, K, "not_reached", levels, err, tag, verbose=verbose)
                if verbose:
                    print(f"level {K}: NOT reached (got {levels}, err={err}); stopping")
                break
            if credit_out or levels < K:
                break
            snapshot_wip_context(game, ws, K, "reached_before_debrief", levels, err, tag, verbose=verbose)
            files_before_debrief = {
                name: _read(os.path.join(ws, name))
                for name in ("legs.py", "players.py", "solve.py", "legs_log.md")
            }
            try:
                debrief_fn(ws, K)
                levels2, path2, _ = verify_fn(game, solve_p)  # behaviour preserved?
            except CreditOut as ex:
                print(f"CREDIT-OUT during debrief after level {K}: {ex}; preserving solved level")
                levels2, path2 = levels, path
                snapshot_wip_context(game, ws, K, "debrief_credit_out", levels, str(ex), tag, verbose=verbose)
            else:
                snapshot_wip_context(game, ws, K, "after_debrief", levels2, None, tag, verbose=verbose)
                if levels2 < levels:
                    for name, text in files_before_debrief.items():
                        with open(os.path.join(ws, name), "w") as f:
                            f.write(text)
                    if verbose:
                        print(f"level {K}: debrief regressed solve ({levels2} < {levels}); restored pre-debrief files")
            reached = max(levels, levels2)
            path = path2 if levels2 >= levels else path
            Cm = marginal_complexity(legs_b, _read(legs_p), players_b, _read(players_p))
            _record_level(rep, K, Cm)
            rep.reached = reached
            rep.final_path = path
            rep.validated = A.validate(game, rep.final_path, rep.reached) if rep.final_path else False
            _save_checkpoint(ws, rep)
            promote_verified_artifact(game, ws, rep, tag, verbose=verbose)
            if verbose:
                print(f"level {K}: reached={reached} marginal_C={Cm} "
                      f"total_C={rep.total_marginal_C} validated={rep.validated} F={rep.free_energy:.3f}")
            if reached <= K - 1:
                break

    except KeyboardInterrupt:
        snapshot_wip_context(game, ws, rep.reached + 1, "interrupted",
                             rep.reached, "interrupted mid-level", tag, verbose=verbose)
        interrupted = True

    rep.validated = A.validate(game, rep.final_path, rep.reached) if rep.final_path else False
    _save_checkpoint(ws, rep)
    promote_verified_artifact(game, ws, rep, tag, verbose=verbose)
    if verbose:
        print(f"\n=== {game}: reached level {rep.reached} | validated={rep.validated} | "
              f"total_marginal_C={rep.total_marginal_C} | F={rep.free_energy:.3f} ===")
        print("  per-level marginal novelty (should trend DOWN as legs are reused): "
              + ", ".join(f"L{r.level}:{r.marginal_C}" for r in rep.records))
    _release_workspace_lock(run_lock)
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
    game, model, minutes, maxl, proposer, tag = "wa30", None, 40, 9, "opencode", ""
    fresh, restore_wip = False, True
    for a in sys.argv[1:]:
        if a.startswith("--game="): game = a.split("=", 1)[1]
        elif a.startswith("--model="): model = a.split("=", 1)[1]
        elif a.startswith("--minutes="): minutes = int(a.split("=", 1)[1])
        elif a.startswith("--max-level="): maxl = int(a.split("=", 1)[1])
        elif a.startswith("--proposer="): proposer = a.split("=", 1)[1]
        elif a.startswith("--tag="): tag = a.split("=", 1)[1]
        elif a == "--fresh": fresh = True
        elif a == "--no-wip-restore": restore_wip = False
    orchestrate(game=game, max_level=maxl, proposer=proposer, model=model,
                minutes_per=minutes, tag=tag, seed_artifact=not fresh,
                restore_wip=restore_wip)
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
