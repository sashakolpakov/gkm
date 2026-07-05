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
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import gkm_arena as A
from gkm_solve_agent import discovered_context

SCRATCH = "/private/tmp/claude-501/-Users-sasha-gkm/e3e00be1-d1a5-4095-a6ef-4d720f42d84e/scratchpad"


def _loc(code: str) -> int:
    """Description length proxy: non-blank, non-comment lines."""
    return sum(1 for ln in (code or "").splitlines()
               if ln.strip() and not ln.strip().startswith("#"))


def marginal_complexity(legs_before: str, legs_after: str,
                        players_before: str, players_after: str) -> int:
    """C_marginal = NEW structure introduced this level. Growth of the shared library
    plus growth of the per-level players. A reused leg is already in legs_before, so
    it contributes 0 -- reuse is free, only novelty is paid for."""
    return (max(0, _loc(legs_after) - _loc(legs_before))
            + max(0, _loc(players_after) - _loc(players_before)))


def free_energy(levels: int, marginal_C_total: int, lam: float = 0.02) -> float:
    """F = R + lambda*C with R = -levels_reached and C = total marginal novelty."""
    return -float(levels) + lam * float(marginal_C_total)


CHECKPOINT_FILE = "checkpoint.json"
"""Filename for per-level marginal-C checkpoint (enables cross-run resume)."""

PROMOTED_FILES = ("legs.py", "players.py", "solve.py", "legs_log.md", CHECKPOINT_FILE)
"""Files that define a verified leg-library state and should survive scratch loss."""

SNAPSHOT_SKIP_DIRS = {"__pycache__", ".pytest_cache"}


def _save_checkpoint(ws: str, rep: Report) -> None:
    """Persist the Report so a later restart restores the full marginal-C history."""
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
    return Report(
        game=data["game"],
        reached=data["reached"],
        total_marginal_C=data["total_marginal_C"],
        records=[LevelRecord(**r) for r in data.get("records", [])],
        final_path=data.get("final_path", []),
        validated=data.get("validated", False),
    )


def artifact_dir(game: str, tag: str = "") -> str:
    """Stable, repo-visible storage for the latest verified leg-library artifact."""
    labdir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(labdir, "agent_solutions", f"{game}_legs")


def _wip_level_dir(art: str, level: int) -> str:
    return os.path.join(art, "wip_context", f"level_{level:02d}")


def _workspace_snapshot_files(ws: str) -> List[str]:
    files = []
    for name in sorted(os.listdir(ws)):
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


def seed_workspace_from_artifact(game: str, ws: str, tag: str = "", verbose: bool = True) -> Optional[Report]:
    """Overwrite scratch with the latest promoted verified state, if one exists.

    Scratch is treated as disposable and possibly contaminated by an unfinished next
    level. The repo artifact is the source of truth for resuming.
    """
    art = artifact_dir(game, tag)
    rep = _load_checkpoint(art)
    if rep is None or not rep.validated:
        return None
    for name in PROMOTED_FILES:
        src = os.path.join(art, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(ws, name))
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
    art = artifact_dir(game, tag)
    old = _load_checkpoint(art)
    if old is not None and old.validated and old.reached > rep.reached:
        if verbose:
            print(f"kept artifact at level {old.reached}; current verified level {rep.reached} is older")
        return False
    os.makedirs(art, exist_ok=True)
    _save_checkpoint(ws, rep)
    for name in PROMOTED_FILES:
        src = os.path.join(ws, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(art, name))
    with open(os.path.join(art, "README.md"), "w") as f:
        f.write(_artifact_readme(game, rep))
    with open(os.path.join(art, "run.log"), "w") as f:
        f.write(_artifact_run_log(game, rep))
    if verbose:
        print(f"promoted verified artifact: {art} (reached={rep.reached})")
    return True


# ---------------------------------------------------------------------------
# workspace + real verifier (running the agent's solve.py on the real game)
# ---------------------------------------------------------------------------
TESTER = '''import importlib.util, json, os, sys
sys.path.insert(0, {labdir!r})
import gkm_arena as A
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
    ):
        p = os.path.join(ws, name)
        if not os.path.exists(p):
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


class CreditOut(RuntimeError):
    """Raised when the proposer subprocess reports it is out of credits/quota, so the
    orchestrator can stop the whole sequence cleanly instead of burning the budget."""


# markers of a transient infrastructure failure (dropped connection, server error):
# the proposer never worked on the level, so the attempt is retried, not judged.
_TRANSIENT_MARKERS = ("api error", "connection closed", "connection error", "connection refused",
                      "overloaded", "internal server error", "service unavailable")

_TRANSIENT_RETRIES = 2
"""Extra proposer attempts per level when the failure looks infrastructural."""


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
    try:
        gkm_api_agent.run_agent(ws, task, model=model, minutes=minutes)
    except Exception as ex:
        with open(os.path.join(ws, "proposer_last.log"), "w") as fh:
            fh.write(f"API Error: {type(ex).__name__}: {ex}\n")
    blob = _read(os.path.join(ws, "proposer_last.log")).lower()
    if any(m in blob for m in _CREDIT_OUT_MARKERS):
        raise CreditOut(f"api proposer reported no credits/quota (see {ws}/proposer_last.log)")


def _propose_task(game, K, context, legs_index):
    return (A.PRECONCEPTIONS + "\n\n" + context +
            f"\n\nYou are growing a LEG LIBRARY across the levels of {game}. Existing "
            f"legs in legs.py: {legs_index or '(none yet)'}.\n"
            f"GOAL: make solve.py reach LEVEL {K}. First run `python gkm_try.py` to see "
            "where you are; solve.py dispatches to players.play_level_K. On a clone at "
            f"level {K}, learn its structure. Then WRITE `play_level_{K}(env)` in "
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
    legs_p, players_p, solve_p = (os.path.join(ws, f) for f in ("legs.py", "players.py", "solve.py"))
    if seed_artifact:
        seed_workspace_from_artifact(game, ws, tag, verbose=verbose)
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
    while rep.reached < max_level:
        K = rep.reached + 1
        legs_b, players_b = _read(legs_p), _read(players_p)

        # PHASE 0: auto-solve with existing legs (structural reuse, zero proposer cost)
        auto_result = _try_auto_solve(K, legs_b, players_b,
                                      players_p, solve_p, game, verify_fn)
        if auto_result is not None:
            levels, path, err = auto_result
            Cm = marginal_complexity(legs_b, _read(legs_p), players_b, _read(players_p))
            if verbose:
                print(f"level {K}: auto-solved via existing legs (marginal_C={Cm})")
            # still debrief: the auto-solve succeeded, but legs_log.md may need an entry
            try:
                debrief_fn(ws, K)
                levels2, path2, _ = verify_fn(game, solve_p)
            except CreditOut as ex:
                print(f"CREDIT-OUT during debrief after level {K}: {ex}; preserving solved level")
                levels2, path2 = levels, path
            reached = max(levels, levels2)
            path = path2 if levels2 >= levels else path
            rep.records.append(LevelRecord(level=K, marginal_C=Cm, reached=True))
            rep.total_marginal_C += Cm
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
        for attempt in range(1 + _TRANSIENT_RETRIES):
            try:
                propose_fn(ws, K)
            except CreditOut as ex:
                print(f"CREDIT-OUT at level {K}: {ex}; stopping (reached={rep.reached})")
                snapshot_wip_context(game, ws, K, "credit_out", rep.reached, str(ex), tag, verbose=verbose)
                credit_out = True
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
        Cm = marginal_complexity(legs_b, _read(legs_p), players_b, _read(players_p))
        snapshot_wip_context(game, ws, K, "reached_before_debrief", levels, err, tag, verbose=verbose)
        try:
            debrief_fn(ws, K)
            levels2, path2, _ = verify_fn(game, solve_p)  # behaviour preserved?
        except CreditOut as ex:
            print(f"CREDIT-OUT during debrief after level {K}: {ex}; preserving solved level")
            levels2, path2 = levels, path
            snapshot_wip_context(game, ws, K, "debrief_credit_out", levels, str(ex), tag, verbose=verbose)
        else:
            snapshot_wip_context(game, ws, K, "after_debrief", levels2, None, tag, verbose=verbose)
        reached = max(levels, levels2)
        path = path2 if levels2 >= levels else path
        rep.records.append(LevelRecord(level=K, marginal_C=Cm, reached=True))
        rep.total_marginal_C += Cm
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

    rep.validated = A.validate(game, rep.final_path, rep.reached) if rep.final_path else False
    _save_checkpoint(ws, rep)
    promote_verified_artifact(game, ws, rep, tag, verbose=verbose)
    if verbose:
        print(f"\n=== {game}: reached level {rep.reached} | validated={rep.validated} | "
              f"total_marginal_C={rep.total_marginal_C} | F={rep.free_energy:.3f} ===")
        print("  per-level marginal novelty (should trend DOWN as legs are reused): "
              + ", ".join(f"L{r.level}:{r.marginal_C}" for r in rep.records))
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
    game, model, minutes, maxl, proposer, tag, fresh = "wa30", None, 40, 9, "opencode", "", False
    for a in sys.argv[1:]:
        if a.startswith("--game="): game = a.split("=", 1)[1]
        elif a.startswith("--model="): model = a.split("=", 1)[1]
        elif a.startswith("--minutes="): minutes = int(a.split("=", 1)[1])
        elif a.startswith("--max-level="): maxl = int(a.split("=", 1)[1])
        elif a.startswith("--proposer="): proposer = a.split("=", 1)[1]
        elif a.startswith("--tag="): tag = a.split("=", 1)[1]
        elif a == "--fresh": fresh = True
    orchestrate(game=game, max_level=maxl, proposer=proposer, model=model,
                minutes_per=minutes, tag=tag, seed_artifact=not fresh)
