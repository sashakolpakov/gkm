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
import os
import subprocess
import time
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


# ---------------------------------------------------------------------------
# workspace + real verifier (running the agent's solve.py on the real game)
# ---------------------------------------------------------------------------
TESTER = '''import importlib.util, sys
sys.path.insert(0, {labdir!r})
import gkm_arena as A
spec = importlib.util.spec_from_file_location("solve", "solve.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
levels, path, err = A.run_program({game!r}, m.solve)
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
        return A.run_program(game, m.solve)
    finally:
        if added and wsdir in sys.path:
            sys.path.remove(wsdir)


def _read(path: str) -> str:
    return open(path).read() if os.path.exists(path) else ""


def setup_workspace(game: str) -> str:
    ws = os.path.join(SCRATCH, f"gkm_legs_ws_{game}")
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
def _claude_agent(ws: str, task: str, model: Optional[str], minutes: int) -> None:
    labdir = os.path.dirname(os.path.abspath(__file__))
    cmd = ["claude", "-p", task, "--allowedTools", "Bash", "Read", "Write", "Edit",
           "--dangerously-skip-permissions", "--add-dir", labdir, "--output-format", "text"]
    if model:
        cmd += ["--model", model]
    subprocess.run(cmd, cwd=ws, capture_output=True, text=True, timeout=minutes * 60)


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
                propose_fn: Optional[Callable] = None,
                verify_fn: Optional[Callable] = None,
                debrief_fn: Optional[Callable] = None,
                verbose=True) -> Report:
    """Per-level compose->verify->debrief with marginal-C accounting. propose_fn(ws,K)
    / verify_fn(game, solve_path)->(levels,path,err) / debrief_fn(ws,K) are injectable;
    defaults use the real Claude agent + real game (credits needed)."""
    ws = setup_workspace(game)
    legs_p, players_p, solve_p = (os.path.join(ws, f) for f in ("legs.py", "players.py", "solve.py"))
    context = discovered_context(game) if propose_fn is None else ""
    propose_fn = propose_fn or (lambda w, k: _claude_agent(w, _propose_task(game, k, context, _defs(_read(legs_p))), model, minutes_per))
    debrief_fn = debrief_fn or (lambda w, k: _claude_agent(w, _debrief_task(game, k), model, max(10, minutes_per // 2)))
    verify_fn = verify_fn or run_solve_file

    rep = Report(game=game, reached=0)
    # resume: if the workspace already clears some levels, start above them (don't
    # re-spend the proposer on solved levels; their marginal C was recorded earlier).
    lv0, _, _ = verify_fn(game, solve_p)
    if lv0 > 0:
        rep.reached = lv0
        if verbose:
            print(f"resuming from level {lv0} (existing legs.py/players.py)")
    while rep.reached < max_level:
        K = rep.reached + 1
        legs_b, players_b = _read(legs_p), _read(players_p)
        propose_fn(ws, K)
        levels, path, err = verify_fn(game, solve_p)
        if levels < K:
            if verbose:
                print(f"level {K}: NOT reached (got {levels}, err={err}); stopping")
            break
        Cm = marginal_complexity(legs_b, _read(legs_p), players_b, _read(players_p))
        debrief_fn(ws, K)
        levels2, path2, _ = verify_fn(game, solve_p)      # behaviour preserved?
        reached = max(levels, levels2)
        path = path2 if levels2 >= levels else path
        rep.records.append(LevelRecord(level=K, marginal_C=Cm, reached=True))
        rep.total_marginal_C += Cm
        rep.reached = reached
        rep.final_path = path
        if verbose:
            print(f"level {K}: reached={reached} marginal_C={Cm} "
                  f"total_C={rep.total_marginal_C} F={rep.free_energy:.3f}")
        if reached <= K - 1:
            break

    rep.validated = A.validate(game, rep.final_path, rep.reached) if rep.final_path else False
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
    game, model, minutes, maxl = "wa30", None, 40, 9
    for a in sys.argv[1:]:
        if a.startswith("--game="): game = a.split("=", 1)[1]
        elif a.startswith("--model="): model = a.split("=", 1)[1]
        elif a.startswith("--minutes="): minutes = int(a.split("=", 1)[1])
        elif a.startswith("--max-level="): maxl = int(a.split("=", 1)[1])
    orchestrate(game=game, max_level=maxl, model=model, minutes_per=minutes)
