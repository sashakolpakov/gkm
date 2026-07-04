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

_NAME = {1: "ACTION1", 2: "ACTION2", 3: "ACTION3", 4: "ACTION4", 5: "ACTION5", 6: "ACTION6"}


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
            self.game = _clone.game
            self._game = copy.deepcopy(_clone._game)
            self._fd = _clone._fd
            self.path = list(_clone.path)
            self._budget = _clone._budget
        else:
            self.game = game
            self._budget = _budget or _Budget(200000)
            self.reset()

    def reset(self) -> np.ndarray:
        """Restart the run from the first level; returns the initial frame."""
        env = make_env(self.game)(); env.reset()
        self._game = copy.deepcopy(env._env._game)
        self._fd = self._game.perform_action(ActionInput(id=EA.RESET), raw=True)
        self.path = []
        return self.frame()

    @property
    def actions(self):
        return (1, 2, 3, 4, 5, 6)

    @property
    def levels_completed(self) -> int:
        return self._fd.levels_completed

    def terminal(self) -> bool:
        return str(self._fd.state).endswith("GAME_OVER") or not getattr(self._fd, "frame", None)

    def frame(self) -> np.ndarray:
        return np.asarray(self._fd.frame[-1])

    def step(self, a: int) -> np.ndarray:
        self._budget.tick()
        if a not in (1, 2, 3, 4, 5, 6):
            raise ValueError("action must be 1..6")
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

FIND WHAT YOU CONTROL (there may be MORE THAN ONE): do NOT assume a single avatar.
One OR SEVERAL objects may be under your control. Discover them by EXPERIMENT -- on a
clone, take each action and see which object(s) move consistently. DIFFERENT actions
may steer DIFFERENT objects (e.g. one action-set drives avatar A, another drives
avatar B), or an action may move several objects at once. ENUMERATE EVERY controllable
object and record which actions steer each; a controllable object is one YOUR action
reproducibly moves. An action that moves nothing is probably an interaction. Some
objects move on their own no matter what you do -- those are AUTONOMOUS agents, NOT
yours (see OTHER AGENTS); tell them apart by whether your action reproducibly steers
them.

SEVERAL CONTROLLABLE AVATARS -> COORDINATE THEM (the colimit cone over avatars): when
more than one object is controllable, the level may be UNSOLVABLE by any single one
alone -- do not try to win with just one. Treat each controllable avatar as its own
controller with its own movement legs, and COMPOSE them: stage the world with one
avatar so another can finish, or drive them in turn toward the shared goal. Build a
SEPARATE control routine per avatar and glue them into one plan.

AFFORDANCES: an action can mean anything -- move, toggle, transform, attach, select,
rotate, open, fire. Do NOT assume. Discover what each action does by EXPERIMENT on
clones, and repeat the experiment in DIFFERENT contexts (next to an object, on a
special tile, facing different directions, after another action): an action's effect
often depends on where you are and on the current state. Whatever mechanic you find,
verify it on a clone before building a plan on it.

GOALS: infer the objective from the reward. Deduce the win condition by comparing
frames where the reward increases against frames where it does not.

SPARSE REWARD -> INVENT YOUR OWN DENSE PROGRESS (this is essential here): the level
reward usually fires ONLY when the whole goal is met. Do NOT wait for that sparse
signal or hill-climb mere "the frame changed". Construct your OWN dense progress
measure from the frame -- some count or distance that plausibly tracks partial
progress toward the inferred goal -- and drive it, decomposing the level into
SUBGOALS you can verify one at a time. Re-check the real reward, but steer by your
own dense measure.

OTHER AGENTS (theory of mind): some objects move on their own each turn -- model
them. They may help you, hinder you, or ignore you; work out which by watching what
they do to the things that matter for the goal. They ACT ON EVERY MOVE YOU MAKE.

REACHABILITY & COOPERATION: for each agent (you included) work out, respecting
walls, what it can reach. Never assume an agent can cross a barrier it cannot --
check. If no single agent can reach everything that matters, study how their
reachable regions interact: the solution may require staging the world so that
another agent can finish what you cannot.

LEVELS ARE VARIATIONS: after you clear a level, the next is usually the SAME mechanic
under a transformation -- mirrored or flipped layout, inverted direction of motion or
gravity, swapped colours, shifted geometry, more objects, tighter timing. On a new
level, FIRST test on a clone whether your previous method still works after such a
transform; write your skills parameterized by axis/direction/colour rather than baked
to one orientation, so the transformed retry is one call. Open free-form re-discovery
only after transformed reuse demonstrably fails.

CONSUMABLE AND HAZARDOUS ACTIONS: an action may be a limited resource -- usable only a
few times before penalty or game over, with or without a visible counter. Probe
unfamiliar actions on CLONES and check whether REPEATED use degrades or ends the game;
treat any such action as a scarce budget, spent only on clone-verified winning moves,
never on live experimentation.

TIMERS AND MOVE BUDGETS: a level may carry a hidden or displayed countdown (e.g. a bar
or row of cells that depletes every step). Measure early how many real steps you are
allowed; search on clones and commit only a SHORT verified sequence on the real env.

ALL-OR-NOTHING LEVELS: when no frame-derived measure correlates with progress, the
level may be a combination lock -- nothing visibly improves until everything is right.
Do not hunt for a gradient that does not exist: enumerate candidate parameters
(positions, counts, orders) on clones where failure is free, and commit only the
verified combination.

DELAYED EFFECTS: an action's consequence may unfold over several following frames;
after acting on a clone, let the dynamics settle (a few known-safe steps) before
judging what the action did.

DECOY ACTIONS: one or more actions may do nothing at all; identify no-ops once and
stop re-probing them in every context.

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

PERFORMANCE: env.clone()/env.step() cost ~300 calls/second and you have a few minutes.
So avoid EXHAUSTIVE GLOBAL search, but DO use BOUNDED per-subgoal search/lookahead
(e.g. a short best-first to bring one object one step closer to the goal region, a
few thousand clone-steps at most). Re-perceive and replan after each subgoal. A
controller that pursues a dense per-object subgoal will far outperform both blind
greedy wandering and an exhaustive planner that gets cut off. (numpy is 2.x: use
np.ptp(a), not a.ptp().)"""


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
