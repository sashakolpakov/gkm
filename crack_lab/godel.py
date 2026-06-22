"""Goedel-machine-style self-improvement: when the grounded cone PLATEAUS on a
level, the agent does not get a hand-coded new leg from the programmer. Instead the
local LLM, given the DISCOVERED semantics + the plateau + a stock of human
preconceptions, WRITES NEW LEG CODE; the agent compiles it and ADOPTS it only if it
verifiably helps on the real game (delivered count rises / level advances). The
proof obligation of Schmidhuber's Goedel machine is here discharged empirically by
the simulator, which is the ground-truth verifier.

This is the anti-hard-coding stance the user insists on: the hand-off leg for wa30
L3 (relay a box across the wall to the autonomous helper) is NOT written by the
programmer -- it must be produced here by the LLM and survive verification.

NB: executing model-written code is inherently unsafe; this is a local research
sandbox. We exec in a restricted namespace exposing only the connector API.
"""
from __future__ import annotations
import re
import time
from typing import Callable, Optional

import gkm_crack as G
import llm_binder
from gkm_crack import arr_of, step, terminal, PITCH, MOVES

HUMAN_PRECONCEPTIONS = """You bring a deep, structured model of how the physical and social world works -- the \
kind of common sense a person uses without noticing. Apply ALL of it, explicitly.

PHYSICAL WORLD & OBJECTS
- Objects are PERMANENT and occupy space; two solid things cannot share a cell; \
movement is blocked by whatever is solid. Nothing teleports or vanishes without a cause.
- Space is divided by WALLS/BARRIERS into connected REGIONS. Two cells are in the same \
region only if a clear path of walkable cells joins them. A full barrier makes regions \
genuinely disjoint: no path, no exceptions, however much you wish otherwise.
- AFFORDANCES: work out, from what you actually observed, what each thing affords -- can \
it be walked through, pushed, picked up and carried, toggled, or is it inert structure? \
Do not assume; use the discovered mechanic.
- ASYMMETRIES MATTER: the rule for moving YOURSELF and the rule for moving a CARRIED \
object can differ. A carried object can sometimes be placed one cell past where you \
yourself can stand (onto/against a boundary). Such an edge is exactly where things get \
handed across a barrier.

AGENTS, GOALS & THEORY OF MIND
- Some objects are AGENTS with their own goals and their own autonomous policy. Watch \
what they do and MODEL it: a HELPER independently fetches reachable objects and brings \
them to the goal; an ADVERSARY works against you (and may be neutralised the same way \
you grab an object). Decide which is which from behaviour, not appearance.
- The world is NOT static between your moves: other agents ACT ON EVERY MOVE YOU MAKE. \
Even a 'wasted' step buys a helper a step of progress -- exploit that.
- MEANS-ENDS: state the goal as a concrete final configuration, then reason BACKWARDS. \
Which subgoals must hold? Which agent can achieve each, given the regions?

REACHABILITY-FIRST PLANNING (do this before proposing any plan)
- For EACH agent compute, honestly, the cells it can reach and the objects it can act \
on, RESPECTING the barriers. NEVER claim an agent reaches something across a wall it \
cannot cross -- that is the classic blunder; check the geometry, do not wish it true.
- Then: for every object that must reach the goal, is there a CHAIN of agents whose \
reachable sets overlap that can move it there? An object in no such chain is stranded \
and the task is infeasible as posed -- say so rather than inventing a path.

COOPERATION / RELAY (hand-off across a barrier)
- If you can reach an object but not the goal, and another agent can reach the goal but \
not the object, RELAY it: carry the object to the shared boundary between your reachable \
area and theirs, RELEASE it there, then STEP AWAY (otherwise you instantly re-grab it; \
and stepping also lets the other agent act). The drop cell must lie within the OTHER \
agent's pick-up range -- the two ranges must actually meet at the boundary, or the relay \
fails.

CAUSALITY, BUDGET, ANTI-HALLUCINATION
- Every action has consequences, including on other agents; anticipate side effects.
- Moves are a LIMITED budget; prefer short, decisive plans; irreversible mistakes cost.
- When unsure, EXPERIMENT and observe rather than assume. Ground every belief in what \
the perception/probe actually showed -- not in what would be convenient."""

LEG_API = """Write ONE Python function with EXACTLY this signature:

    def leg(C, g, fd, deadline):
        ...
        return (status, g, fd, path)

status: "done" if you made progress, else "lost". path: the list of action ints you
actually committed (1=up, 2=down, 3=left, 4=right, 5=toggle pick-up/drop). g, fd: the
game clone and frame-data AFTER your committed actions.

In scope (no imports needed):
  step(g, a) -> (g2, fd2)     # apply ONE action to a CLONE (g is never mutated)
  terminal(fd) -> bool        # is the game over?
  PITCH = 4                   # pixels per cell
  C.avatar, C.carrier, C.region, C.toggle, C.carried_border, C.rest_border  # int colours/action
  C.avatar_xy(fd) -> (x, y) or None        # your position (pixel centre)
  C.boxes(fd) -> [box, ...]   # each: box.tl=(x,y) top-left, box.center=(x,y), box.border=int
  C.target_cells(fd) -> [(x,y), ...]       # pixel-aligned cells inside the goal container
  C.frontier(g, fd) -> (max_x, handoff)    # how far right you can walk; handoff=True if goal beyond it
  C.delivered(fd) -> int      # how many boxes rest on the goal now
  fd.levels_completed         # reward; goes up when the level is won

To pick up box B: walk so you are one cell from B FACING it, then step(C.toggle). A box
is yours while box.border == C.carried_border. Drop with step(C.toggle); after a drop
the box border is NOT carried_border. Explore on CLONES; commit only the path you keep.

EXAMPLE (carry the box nearest you to the goal -- copy this structure):
```python
def leg(C, g, fd, deadline):
    sc = C.boxes(fd); av = C.avatar_xy(fd)
    if not sc or av is None:
        return ("lost", g, fd, [])
    b = min(sc, key=lambda b: abs(b.center[0]-av[0]) + abs(b.center[1]-av[1]))
    tx, ty = b.center
    path = []
    for _ in range(40):
        if terminal(fd):
            return ("lost", g, fd, [])
        av = C.avatar_xy(fd)
        # move toward the box, then toggle to grab when adjacent
        if abs(av[0]-tx) + abs(av[1]-ty) <= PITCH + 1:
            a = C.toggle
        elif abs(av[0]-tx) >= abs(av[1]-ty):
            a = 4 if tx > av[0] else 3
        else:
            a = 2 if ty > av[1] else 1
        g, fd = step(g, a); path.append(a)
        if fd.levels_completed > 0 and C.delivered(fd) > 0:
            break
    return ("done", g, fd, path)
```
Adapt the idea to the situation (e.g. if the goal is walled off, relay to the boundary).
Keep it under ~45 committed actions."""


def _build_prompt(C, g, fd, feedback: str = "") -> str:
    arr = arr_of(fd)
    sc = C.perceive(arr, C.region_bbox(fd))
    fx, handoff = C._frontier(g, fd)
    helper = C._helper(arr)
    facts = [
        f"Discovered semantics: avatar=colour {C.B.avatar}, carrier(box)=colour {C.B.carrier}, "
        f"target region=colour {C.B.region}, mechanic=pick_up_and_carry, toggle action={C.B.toggle} "
        f"(attach faced box / drop carried), carried-border={C.B.carried_border}, rest-border={C.B.rest_border}.",
        f"Target container bbox = {C.region_bbox(fd)}.",
        f"You (avatar) at {tuple(round(v) for v in sc.avatar) if sc.avatar else None}.",
        f"Autonomous helper agent at {tuple(round(v) for v in helper) if helper else None}.",
        f"Boxes (top-left) at {[b.tl for b in sc.boxes]}.",
        f"Reachability: you can walk right only up to x={round(fx)}; target is "
        f"{'BEYOND your reach (a wall blocks you) -- you cannot deliver directly' if handoff else 'reachable'}.",
        f"Currently {C.delivered(fd)} of {len(sc.boxes)} boxes are on the target.",
    ]
    return (HUMAN_PRECONCEPTIONS + "\n\nSITUATION (a level you are stuck on):\n- "
            + "\n- ".join(facts) +
            ("\n\nPrevious attempt feedback: " + feedback if feedback else "") +
            "\n\n" + LEG_API +
            "\n\nThink about which boxes you can move and where to put them so the helper "
            "finishes. Output ONLY one ```python code block with the leg function.")


def _extract_code(text: str) -> Optional[str]:
    m = re.search(r"```(?:python)?\s*(.*?)```", text or "", re.S)
    code = m.group(1) if m else text
    return code if code and "def leg" in code else None


def compile_leg(code: str) -> Optional[Callable]:
    import numpy as np
    ns = {"step": step, "arr_of": arr_of, "terminal": terminal, "PITCH": PITCH,
          "MOVES": MOVES, "np": np, "__builtins__": {
              "len": len, "range": range, "min": min, "max": max, "abs": abs,
              "sorted": sorted, "set": set, "list": list, "dict": dict, "tuple": tuple,
              "enumerate": enumerate, "zip": zip, "round": round, "sum": sum,
              "any": any, "all": all, "int": int, "float": float, "bool": bool,
              "True": True, "False": False, "None": None}}
    try:
        exec(code, ns)
        return ns.get("leg")
    except Exception:
        return None


def verify(C, g, fd, leg, idle_cap=18, deadline=None):
    """Adopt iff the leg + letting the autonomous helper tick raises delivered count
    or wins. Returns (improved, g, fd, path, note)."""
    d0 = C.delivered(fd); lvl0 = fd.levels_completed
    try:
        status, ng, nfd, path = leg(C, g, fd, deadline or time.time() + 60)
    except Exception as exc:
        return (False, g, fd, [], f"leg raised {type(exc).__name__}: {str(exc)[:60]}")
    if status != "done" or nfd is None or terminal(nfd):
        return (False, g, fd, [], f"leg status={status}/terminal")
    if nfd.levels_completed > lvl0:
        return (True, ng, nfd, list(path), "won during leg")
    # let the helper act (idle, harmless moves) and watch for a delivery
    idle = []
    cg, cf = ng, nfd
    for i in range(idle_cap):
        cg, cf = step(cg, 3 if i % 2 == 0 else 4)
        if terminal(cf):
            break
        idle.append(3 if i % 2 == 0 else 4)
        if cf.levels_completed > lvl0:
            return (True, cg, cf, list(path) + idle, "won after helper ticks")
        if C.delivered(cf) > d0:
            return (True, cg, cf, list(path) + idle, f"delivered {d0}->{C.delivered(cf)}")
    return (False, g, fd, [], f"no delivery (delivered stayed {d0})")


def evolve(C, g, fd, level0, model=None, rounds=6, deadline_s=900, verbose=True):
    """Plateau -> ask the LLM for new leg code -> verify -> adopt. Repeat."""
    import llm_binder
    path = []
    started = time.time()
    feedback = ""
    for r in range(rounds):
        if C.level(fd) > level0 or C.solved(g):
            break
        prompt = _build_prompt(C, g, fd, feedback)
        try:
            raw = llm_binder.ollama_text(prompt, num_predict=900, timeout=600,
                                         **({} if model is None else {"model": model}))
        except Exception as exc:
            if verbose: print(f"  round {r}: LLM call failed ({exc}); retrying")
            feedback = "(previous call timed out -- keep the leg SHORT and simple)"
            continue
        text = raw if isinstance(raw, str) else str(raw)
        code = _extract_code(text)
        if not code:
            feedback = "your reply had no python code block with `def leg`"
            if verbose: print(f"  round {r}: no code produced"); continue
        leg = compile_leg(code)
        if leg is None:
            feedback = "your code failed to compile"
            if verbose: print(f"  round {r}: compile failed"); continue
        improved, ng, nfd, lp, note = verify(C, g, fd, leg, deadline=started + deadline_s)
        if verbose:
            print(f"  round {r}: proposed leg -> {note}")
        if improved:
            g, fd, path = ng, nfd, path + lp
            feedback = f"that leg worked ({note}); now improve further from the new state"
        else:
            feedback = f"that leg did not help ({note}); try a different idea"
        if time.time() - started > deadline_s:
            break
    return g, fd, path


def _try_library(C, g, fd, library, deadline):
    """PowerPlay reuse: try each already-adopted (LLM-written) leg; keep the first
    that still verifiably helps from the current state."""
    for i, leg in enumerate(library):
        improved, ng, nfd, lp, note = verify(C, g, fd, leg, deadline=deadline)
        if improved:
            return True, ng, nfd, lp, f"reused leg#{i} ({note})"
    return False, g, fd, [], ""


def solve(game="wa30", max_level=3, model=None, rounds_per_level=16,
          deadline_s=6000, verbose=True):
    """The whole Schmidhuber GKM / Goedel-machine method: a DISCOVERED connector
    (anchor + manipulation verb + colours, grounded by probe+LLM) and a library of
    legs that are ALL written by the local LLM and admitted only after empirical
    verification on the game. Crack L1..L3 by reusing/evolving legs; validate."""
    C = G.CarryConnector.build(game, model=model, use_llm=True, verbose=verbose)
    library = []
    g, fd = C.fresh()
    path = []
    started = time.time()
    while C.level(fd) < max_level:
        level0 = C.level(fd)
        if verbose:
            print(f"\n[level {level0 + 1}] {C.describe(fd)} -- evolving/reusing LLM legs")
        feedback = ""
        for r in range(rounds_per_level):
            if C.level(fd) > level0 or C.solved(g) or time.time() - started > deadline_s:
                break
            ok, ng, nfd, lp, note = _try_library(C, g, fd, library, started + deadline_s)
            if ok:
                g, fd, path = ng, nfd, path + lp
                if verbose: print(f"  round {r}: {note}")
                continue
            prompt = _build_prompt(C, g, fd, feedback)
            try:
                raw = llm_binder.ollama_text(prompt, num_predict=900, timeout=600,
                                             **({} if model is None else {"model": model}))
            except Exception as exc:
                if verbose: print(f"  round {r}: LLM timeout ({exc}); retry")
                feedback = "(timed out -- keep the leg SHORT)"; continue
            code = _extract_code(raw)
            leg = compile_leg(code) if code else None
            if leg is None:
                feedback = "no compilable `def leg` produced";
                if verbose: print(f"  round {r}: no usable code"); continue
            improved, ng, nfd, lp, note = verify(C, g, fd, leg, deadline=started + deadline_s)
            if improved:
                library.append(leg); g, fd, path = ng, nfd, path + lp
                feedback = f"that leg worked ({note}); keep going from the new state"
                if verbose: print(f"  round {r}: EVOLVED new leg -> {note} (library={len(library)})")
            else:
                feedback = f"that leg failed ({note}); try a different idea"
                if verbose: print(f"  round {r}: proposed leg rejected -> {note}")
        if C.level(fd) <= level0:
            if verbose: print(f"[level {level0 + 1}] NOT cracked (plateau)"); break
        if verbose: print(f"[level {level0 + 1}] CRACKED -> level {C.level(fd)} (cum path={len(path)})")
    ok = C.validate(path, C.level(fd)) if path else False
    print(f"\n=== {game}: reached level {C.level(fd)}/{max_level} | path_len={len(path)} | "
          f"replay-validated={ok} | LLM-written legs={len(library)} ===")
    if path:
        print(f"PATH={path}")
    return {"reached": C.level(fd), "path": path, "validated": ok, "legs": len(library)}


if __name__ == "__main__":
    import sys
    model = None
    for a in sys.argv[1:]:
        if a.startswith("--model="):
            model = a.split("=", 1)[1]
    if "--whole" in sys.argv:
        ml = 3
        for a in sys.argv[1:]:
            if a.startswith("--max-level="):
                ml = int(a.split("=", 1)[1])
        solve("wa30", max_level=ml, model=model)
        sys.exit(0)
    C = G.CarryConnector.build("wa30", model=model, use_llm=True, verbose=True)
    g, fd = C.fresh()
    base = []
    # reach L3 with the grounded cone (cracks L1, L2)
    for _ in range(2):
        g, fd, lp = G.gkm_cone(C, g, fd, C.level(fd), deadline=time.time()+900, verbose=False)
        base += lp
    L3 = C.level(fd)
    print(f"\nreached level {L3+1}; cone PLATEAUS here. Invoking Goedel-machine evolution.\n")
    g, fd, p = evolve(C, g, fd, L3, model=model, rounds=12, deadline_s=3000, verbose=True)
    full = base + p
    won = C.level(fd) > L3
    print(f"\nafter evolution: level {C.level(fd)+1}, delivered {C.delivered(fd)}, evolved steps {len(p)}")
    if won:
        ok = C.validate(full, L3 + 1)
        print(f"L3 CRACKED by evolved (LLM-written) legs; replay-validated={ok} path_len={len(full)}")
        print(f"PATH={full}")
