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
from gkm_crack import arr_of, step, terminal, PITCH, MOVES

HUMAN_PRECONCEPTIONS = """You reason about the world with a child's rich physical and social common sense:
- Objects persist and occupy space; two solid things can't share a cell.
- A TARGET/container is where things must end up; check if YOU can even reach it.
- BARRIERS split space into regions; if the target is walled off from you, you can \
NEVER carry a box there yourself.
- There may be OTHER agents. An autonomous HELPER will, on its own, pick up boxes it \
can reach and carry them to the target -- and it MOVES on every one of your moves.
- COOPERATION / RELAY: if you can reach a box but not the target, and a helper can \
reach the target but not the box, hand it off -- carry the box to the shared \
boundary between your reachable area and the helper's, DROP it there, then STEP AWAY \
(or you'll immediately pick it back up, and your move also lets the helper act).
- A carried object can sometimes be pushed one cell further than you can walk \
(into/onto a boundary), which is exactly where a hand-off happens."""

LEG_API = """Write a Python function with EXACTLY this signature:

    def leg(C, g, fd, deadline):
        # returns (status, g, fd, path)
        ...

where status is "done" (made progress / won) or "lost" (no good), path is a list of
action ints you took (1=up,2=down,3=left,4=right,5=toggle pick-up/drop), and g, fd
are the resulting game clone + frame-data AFTER your actions.

Available in scope (do NOT import anything):
  step(g, a) -> (g2, fd2)         # apply ONE action to a CLONE; never mutates g
  arr_of(fd) -> np.ndarray        # the 64x64 colour frame
  terminal(fd) -> bool            # game over?
  PITCH = 4                       # pixels per logical cell
  C.B                             # binding: .avatar .carrier .region .toggle
                                  #          .carried_border .rest_border colours
  C.perceive(arr) -> scene        # scene.avatar=(x,y); scene.boxes=[Box(tl,center,border)];
                                  #   scene.ring_bbox=(x0,y0,x1,y1); scene.targets=[(x,y)..]
  C._frontier(g, fd) -> (max_x, handoff_bool)   # how far right YOU can walk; handoff=target beyond it
  fd.levels_completed             # the reward; if it goes up you won the level

Rules: explore on CLONES via step(); only the actions you actually commit go in path.
Track a chosen box by continuity (nearest tl to its previous tl). Keep it under ~45
actions. Return ("lost", g, fd, []) if you can't make progress."""


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


if __name__ == "__main__":
    import sys
    model = None
    for a in sys.argv[1:]:
        if a.startswith("--model="):
            model = a.split("=", 1)[1]
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
