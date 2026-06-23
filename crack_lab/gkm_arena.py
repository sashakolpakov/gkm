"""The RAWEST setup, chosen deliberately: the engine hands the agent ONLY a raw
frame + the ability to act + the reward. Everything else -- perceiving objects,
discovering the mechanics, planning paths, composing strategies like a two-agent
hand-off -- must be WRITTEN BY THE AGENT (the local LLM), carrying a lot of human
preconceptions about the world. The agent's program is admitted only if it
verifiably clears more levels on the real game (Schmidhuber Goedel-machine: the
proof obligation discharged empirically by the simulator).

No perception, no connector, no carry/search/hand-off helpers are provided. The
human contribution is exactly three things: (1) this thin raw harness, (2) a rich
human-preconception system prompt, (3) the verify-by-reward evolution loop.

    python raw_arena.py [--model=NAME] [--game=wa30] [--rounds=N]
"""
from __future__ import annotations
import copy
import re
import subprocess
import sys
import time
from typing import Optional

import numpy as np

import llm_binder
from lab import make_env
from arcengine import ActionInput, GameAction as EA


def propose_text(prompt: str, proposer: str = "ollama", model: Optional[str] = None,
                 timeout: int = 900) -> str:
    """The proposer is pluggable. 'ollama' = a LOCAL model (eval-legal, offline).
    'claude' = the actual Claude Code AGENT invoked headlessly as a SUBPROCESS
    (`claude -p`) -- a far stronger proposer, but it uses the network/API, so it is
    a DEMO / upper-bound, NOT eval-legal. The substrate is identical either way; only
    the cognition behind propose->verify changes."""
    if proposer == "claude":
        cmd = ["claude", "-p", prompt, "--output-format", "text"]
        if model:
            cmd += ["--model", model]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd="/tmp")
        return r.stdout or ""
    return llm_binder.ollama_text(prompt, num_predict=1600, timeout=600,
                                  **({} if model is None else {"model": model}))

_NAME = {1: "ACTION1", 2: "ACTION2", 3: "ACTION3", 4: "ACTION4", 5: "ACTION5"}


class _Budget:
    """Caps TOTAL engine steps (real + lookahead) so a runaway agent can't hang."""
    def __init__(self, cap): self.cap = cap; self.used = 0
    def tick(self):
        self.used += 1
        if self.used > self.cap:
            raise RuntimeError("step budget exhausted")


class Arena:
    """The whole interface the agent gets. Nothing game-specific is exposed."""
    def __init__(self, game="wa30", _budget=None, _clone=None):
        if _clone is not None:
            self._game = copy.deepcopy(_clone._game)
            self._fd = _clone._fd
            self.path = list(_clone.path)
            self._budget = _clone._budget
        else:
            env = make_env(game)(); env.reset()
            self._game = copy.deepcopy(env._env._game)
            self._fd = self._game.perform_action(ActionInput(id=EA.RESET), raw=True)
            self.path = []
            self._budget = _budget or _Budget(200000)

    @property
    def actions(self):
        return (1, 2, 3, 4, 5)

    @property
    def levels_completed(self) -> int:
        return self._fd.levels_completed

    def terminal(self) -> bool:
        return str(self._fd.state).endswith("GAME_OVER") or not getattr(self._fd, "frame", None)

    def frame(self) -> np.ndarray:
        return np.asarray(self._fd.frame[-1])

    def step(self, a: int) -> np.ndarray:
        self._budget.tick()
        if a not in (1, 2, 3, 4, 5):
            raise ValueError("action must be 1..5")
        self._fd = self._game.perform_action(ActionInput(id=EA[_NAME[a]]), raw=True)
        self.path.append(a)
        return self.frame()

    def clone(self) -> "Arena":
        return Arena(_clone=self)


PRECONCEPTIONS = """You are an agent with a deep, human-like model of the world. You will be given ONLY
a raw 64x64 grid of small integers (colours 0..15), the ability to take one of five
opaque actions, and a reward (levels_completed). Nothing is labelled. You must work
out EVERYTHING yourself, the way a person dropped into a strange video game would,
using your common sense:

PHYSICAL WORLD: the grid shows objects made of coloured cells; objects are solid and
occupy space; a background colour fills empty space; some colours form walls/barriers
that divide the space into regions you cannot cross. Things persist and don't vanish
without cause.

FIND YOURSELF FIRST: one object is YOU. Discover which by EXPERIMENT -- on a clone,
take each action and see which object moves consistently; that is your avatar and
those are your movement actions. A non-moving action is probably an interaction.

AFFORDANCES & MECHANICS: by experimenting on clones, learn what the special action
does (does walking into an object push it? does the action attach an object so it
moves WITH you, and again detaches it? = pick-up-and-carry). Learn the move grid step.

GOALS: infer the objective from the reward. A container/target region is where things
must end up; deduce the win condition by what changes when reward increases.

OTHER AGENTS (theory of mind): some objects move on their own each turn -- model them.
A HELPER autonomously brings objects to the goal; an ADVERSARY hinders you and may be
removed the way you grab things. They ACT ON EVERY MOVE YOU MAKE.

REACHABILITY & COOPERATION: for each agent work out, respecting walls, what it can
reach. If you can reach an object but not the goal while a helper can reach the goal
but not the object, RELAY it: carry it to the shared boundary, drop it where the
helper can pick it up, then step away (your move also lets the helper act). Never
assume an agent can cross a wall it cannot -- check.

PLAN & VERIFY: use clones to look ahead and plan; only commit moves on the real env.
Budget is limited; be decisive."""

API = """Write a Python function with EXACTLY this signature (no other top-level code):

    def solve(env):
        # drive env to clear as many levels as possible; no return value needed
        ...

`env` is the ONLY interface (do not import anything except numpy as np, math,
collections, heapq):
  env.reset() -> np.ndarray            # 64x64 int frame; also resets the run
  env.frame() -> np.ndarray            # current 64x64 frame
  env.step(a) -> np.ndarray            # commit action a in {1,2,3,4,5}; advances the
                                       #   REAL run (costs one move) and returns the new frame
  env.clone() -> env                   # a COPY for SAFE lookahead; stepping the clone
                                       #   does NOT affect the real env -- use it to learn
                                       #   the rules and to plan, then commit on the real env
  env.levels_completed -> int          # your reward; make it go up
  env.terminal() -> bool               # True when the game is over (out of moves)
  env.actions -> (1,2,3,4,5)

Discover perception and rules from the frames via clones, then act on env. Keep total
work bounded (a few hundred real moves). Output ONE ```python code block.

PERFORMANCE: env.clone()/env.step() are EXPENSIVE (~300 calls/second). You have a
wall-clock budget of a few minutes total, so do NOT run deep or exhaustive search
over clones -- experiment briefly to learn the rules, then act GREEDILY on the real
env, re-perceiving after each move. A short greedy controller that commits real moves
beats a thorough planner that gets cut off before it acts."""


def _save_program(game: str, rnd: int, code: str):
    """Persist each generated program so we can inspect WHAT the agent wrote."""
    import os
    d = "/private/tmp/claude-501/-Users-sasha-gkm/e3e00be1-d1a5-4095-a6ef-4d720f42d84e/scratchpad/gkm_programs"
    try:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{game}_round{rnd}.py"), "w") as fh:
            fh.write(code)
    except Exception:
        pass


def _extract(text: str) -> Optional[str]:
    m = re.search(r"```(?:python)?\s*(.*?)```", text or "", re.S)
    code = m.group(1) if m else text
    return code if code and "def solve" in code else None


def _compile(code: str):
    """Return (solve_fn, error_str). Permissive builtins -- this is a local research
    sandbox; the point is to test the MODEL, not to fight Python."""
    import math, collections, heapq, builtins
    ns = {"np": np, "math": math, "collections": collections, "heapq": heapq,
          "__builtins__": builtins}
    try:
        exec(code, ns)
        fn = ns.get("solve")
        return (fn, None) if fn else (None, "no top-level `def solve`")
    except Exception as exc:
        import traceback
        return None, f"{type(exc).__name__}: {exc}".strip()[:160]


def run_program(game, solve, step_cap=600, time_cap=600):
    """Run the agent program on a fresh real env; return (levels, path)."""
    env = Arena(game, _budget=_Budget(step_cap * 400))
    started = time.time()
    # wrap step to enforce wall-clock + a hard real-move cap
    real = {"n": 0}
    orig = env.step
    def guarded(a):
        real["n"] += 1
        if real["n"] > step_cap or time.time() - started > time_cap:
            raise RuntimeError("real-move/time cap")
        return orig(a)
    env.step = guarded
    err = None
    try:
        solve(env)
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}".strip()[:160]
    return env.levels_completed, list(env.path), err


def validate(game, path, expected):
    env = Arena(game)
    for a in path:
        if env.terminal():
            break
        env.step(a)
    return env.levels_completed >= expected


def free_energy(levels, code, path, lam_c=0.04, lam_a=0.002):
    """GKM free energy F = goal-risk R + lambda*complexity C (lower is better).
    R = -levels (reward); C = program description length + committed-path length.
    This is the Schmidhuber complexity pressure: a proposal is kept only if its
    reward gain pays for the description complexity it adds."""
    desc = len(code or "") / 1000.0
    return (-float(levels)) + lam_c * desc + lam_a * len(path or [])


def evolve(game="wa30", model=None, rounds=8, proposer="ollama", verbose=True):
    """Schmidhuber GKM evolution: a PROPOSER (local model, or the Claude Code agent
    as a subprocess) writes a solve() program; it is admitted only if it LOWERS the
    free energy F = R + lambda*C on the real game (and replay-validates). The
    simulator is the ground-truth verifier."""
    best_code, best_levels, best_path = None, 0, []
    best_F = float("inf")
    feedback = ""
    for r in range(rounds):
        prompt = (PRECONCEPTIONS + "\n\n" + API +
                  (("\n\nYour current best program reached level %d. Improve it to go "
                    "further (and keep it concise -- complexity is penalised). Feedback: "
                    % best_levels + feedback) if best_code else
                   ("\n\nFeedback: " + feedback if feedback else "")))
        if best_code:
            prompt += "\n\nCurrent best program:\n```python\n" + best_code + "\n```"
        try:
            raw = propose_text(prompt, proposer=proposer, model=model)
        except Exception as exc:
            if verbose: print(f"round {r}: proposer failed ({exc})"); feedback = "(call failed; keep it shorter)"; continue
        code = _extract(raw)
        if not code:
            if verbose: print(f"round {r}: no code"); feedback = "output exactly one ```python block defining def solve(env):"; continue
        _save_program(game, r, code)              # so we can see what the agent wrote
        solve, cerr = _compile(code)
        if solve is None:
            if verbose: print(f"round {r}: compile failed -> {cerr}")
            feedback = f"your code failed to compile: {cerr}. Fix it."; continue
        levels, path, rerr = run_program(game, solve)
        ok = validate(game, path, levels) if path else False
        F = free_energy(levels, code, path) if ok else float("inf")
        admit = ok and (F < best_F - 1e-9 or levels > best_levels)
        capped = bool(rerr and "cap" in rerr)
        if verbose:
            print(f"round {r}: level {levels} moves={len(path)} replay-ok={ok} "
                  f"F={F:.3f} (best {best_F:.3f}) -> {'ADMIT' if admit else 'reject'}"
                  + (f" err: {rerr}" if rerr else ""))
        if admit:
            best_code, best_levels, best_path, best_F = code, levels, path, F
            feedback = (f"admitted -- level {levels} in {len(path)} moves (F={F:.3f}). Go "
                        "further with a CONCISE change: later levels add a wall splitting the "
                        "board and a helper agent; relay boxes across the wall.")
        else:
            why = ("you were CUT OFF by the time limit -- CLONES ARE EXPENSIVE (~300 "
                   "env-steps/second). Do far less lookahead: experiment on clones only "
                   "briefly to learn the rules, then act GREEDILY on the real env. Avoid "
                   "deep/exhaustive search." if capped else
                   (f"it raised {rerr}" if rerr else
                    ("it committed 0 real env.step moves (you only stepped clones)" if not path
                     else f"its free energy F={F:.3f} did not beat the best {best_F:.3f}")))
            feedback = (f"not admitted: {why} Commit real env.step moves and keep clone use small.")
    print(f"\n=== {game}: best level {best_levels} | F={best_F:.3f} | proposer={proposer} | "
          f"replay-validated={validate(game, best_path, best_levels) if best_path else False} "
          f"| moves={len(best_path)} ===")
    if best_path:
        print(f"PATH={best_path}")
    return {"levels": best_levels, "path": best_path, "code": best_code, "F": best_F}


if __name__ == "__main__":
    model = None; game = "wa30"; rounds = 8; proposer = "ollama"
    for a in sys.argv[1:]:
        if a.startswith("--model="): model = a.split("=", 1)[1]
        elif a.startswith("--game="): game = a.split("=", 1)[1]
        elif a.startswith("--rounds="): rounds = int(a.split("=", 1)[1])
        elif a.startswith("--proposer="): proposer = a.split("=", 1)[1]
    evolve(game=game, model=model, rounds=rounds, proposer=proposer)
