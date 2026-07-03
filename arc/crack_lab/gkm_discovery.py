"""Discovery phase: learn what the abstract actions MEAN by interaction, then let
the local LLM bind those raw effects to semantic verbs. This is the foundation the
later evolution stands on -- the connector's mechanic vocabulary is DISCOVERED, not
assumed (carry was one verb; gates/jumps/teleports are others, found the same way).

Pipeline (game-agnostic; uses only the anchor + logical-grid perception):
  1. PROBE abstract actions in varied contexts (free space; next to a movable
     object; next to a barrier) on real clones.
  2. CLASSIFY each (action, context) -> a channel-blind EFFECT signature:
       self_translate | object_push | object_attach | object_comove |
       object_release | barrier_open | object_appear | object_vanish | no_effect
  3. BIND effects -> named verbs with the local LLM from legible observations;
     the interaction signatures are the ground-truth verifier (the LLM may mis-name;
     propose->verify, never propose-only).

`discover(game)` returns the bound verb table and prints it.
"""
from __future__ import annotations
import copy
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import priors
from lab import make_env
from logical_grid import Grid, components, objects
from anchor_connector import AnchorConnector
from cofibrant import Anchor
from arcengine import ActionInput, GameAction as EA

NAME = {0: "RESET", 1: "ACTION1", 2: "ACTION2", 3: "ACTION3", 4: "ACTION4", 5: "ACTION5",
        6: "ACTION6", 7: "ACTION7"}
PITCH = 4


def arr_of(fd):
    return np.asarray(fd.frame[-1])


def step(g, a):
    gc = copy.deepcopy(g)
    return gc, gc.perform_action(ActionInput(id=EA[NAME[a]]), raw=True)


def over(fd):
    return str(fd.state).endswith("GAME_OVER") or not getattr(fd, "frame", None)


def _centroid(px):
    return (sum(p[0] for p in px)/len(px), sum(p[1] for p in px)/len(px))


@dataclass
class Effect:
    action: int
    context: str
    signature: str
    detail: dict = field(default_factory=dict)


@dataclass
class World:
    game: str
    factory: object
    anchor_color: int
    grid: Grid
    actions: Tuple[int, ...]
    movement: Dict[int, Tuple[int, int]]
    effects: Tuple[int, ...]          # non-movement action candidates
    barrier_colors: Tuple[int, ...]   # impassable structure colours (from priors)
    anchor: object = None             # primary per-component cofibrant Anchor
    anchors: List[Anchor] = field(default_factory=list)  # ALL steerable anchors


def survey(game: str, use_llm=True, model=None) -> World:
    factory = make_env(game)
    env = factory(); snap = env.reset()
    actions = tuple(env.available_actions or ())
    kw = {} if model is None else {"model": model}
    ar = AnchorConnector(use_llm=use_llm, **kw).identify(factory, actions=actions)
    if ar.anchor is None:
        raise RuntimeError("no controllable anchor")
    # discover ALL steerable components (colimit cone: every object with distinct
    # movement, not just the best one)
    all_anchors, _ = AnchorConnector(use_llm=False).identify_all(factory, actions=actions)
    frame = np.asarray(snap.frame)
    movement = {a: v for a, v in ar.anchor.vectors.items() if a in actions and v != (0, 0)}
    effects = tuple(a for a in actions if a not in movement)
    bg = int(np.bincount(frame.flatten()).argmax())
    struct = tuple(c for c in priors.structure_colours(frame.tolist()) if c != bg)
    return World(game, factory, ar.anchor.color, Grid.infer(frame),
                 actions, movement, effects, struct, anchor=ar.anchor,
                 anchors=all_anchors)


def _anchor_xy(arr, w: World):
    # use the per-component cofibrant anchor (handles avatars that are one of
    # several same-colour components, e.g. g50t/ls20); fall back to largest comp.
    if w.anchor is not None:
        cell = w.anchor.locate(arr, w.grid)
        if cell is not None:
            return (cell[0]*w.grid.pitch + w.grid.phase[0] + w.grid.pitch/2,
                    cell[1]*w.grid.pitch + w.grid.phase[1] + w.grid.pitch/2)
    comps = components(arr, w.anchor_color)
    return _centroid(max(comps, key=len)) if comps else None


def _small_objects(arr, w: World):
    """movable-object candidates: small non-anchor components, with pixel centroids."""
    out = []
    for col in set(int(v) for v in np.unique(arr)):
        if col == w.anchor_color:
            continue
        for comp in components(arr, col):
            if len(comp) > 8:
                continue
            out.append((col, w.grid.majblock(comp), len(comp), _centroid(comp)))
    return out


def _nav_to(w: World, g, fd, center, max_nodes=3500):
    """Best-first move-nav until the anchor is one cell from a pixel `center`."""
    import heapq
    cx, cy = center
    adj = [(cx-PITCH, cy), (cx+PITCH, cy), (cx, cy-PITCH), (cx, cy+PITCH)]

    def near(a):
        return min(abs(a[0]-p[0])+abs(a[1]-p[1]) for p in adj)
    a0 = _anchor_xy(arr_of(fd), w)
    seen = {(round(a0[0]), round(a0[1]))}
    heap = [(near(a0), 0, g, fd, [])]
    ctr = 1
    while heap:
        _, n, cg, cf, path = heapq.heappop(heap)
        a = _anchor_xy(arr_of(cf), w)
        if a and near(a) < PITCH and path:
            return cg, cf, path
        if n > max_nodes or len(path) > 32:
            continue
        for act in w.movement:
            ng, nf = step(cg, act)
            if over(nf):
                continue
            a2 = _anchor_xy(arr_of(nf), w)
            if not a2 or (round(a2[0]), round(a2[1])) in seen:
                continue
            seen.add((round(a2[0]), round(a2[1])))
            heapq.heappush(heap, (near(a2)+0.01*len(path), ctr, ng, nf, path+[act])); ctr += 1
    return None


def _barrier_cells(arr, w: World):
    cells = set()
    for col in w.barrier_colors:
        for o in objects(arr, w.grid, [col]):
            cells.add(o.cell)
    return cells


def probe(w: World, start=None) -> List[Effect]:
    """Run controlled experiments; classify each into a channel-blind effect. If
    `start=(g, fd)` is given, probe at THAT level (so the same discovery applies on
    L1, L2, L3, ...); otherwise probe the reset (L1) state."""
    if start is not None:
        g, fd = start
    else:
        env = w.factory(); env.reset()
        g = copy.deepcopy(env._env._game)
        fd = g.perform_action(ActionInput(id=EA.RESET), raw=True)
    found: List[Effect] = []

    # --- free-space: do movement actions translate the anchor? ---
    a0 = _anchor_xy(arr_of(fd), w)
    for act in w.actions:
        _, nf = step(g, act)
        if over(nf):
            continue
        a1 = _anchor_xy(arr_of(nf), w)
        if a0 and a1 and (abs(a1[0]-a0[0]) + abs(a1[1]-a0[1])) >= PITCH*0.6:
            found.append(Effect(act, "free_space", "self_translate",
                                {"vector": (round(a1[0]-a0[0]), round(a1[1]-a0[1]))}))

    # --- next to a movable object: push? attach+comove? release? ---
    movs = _small_objects(arr_of(fd), w)
    if movs:
        a0 = _anchor_xy(arr_of(fd), w)
        target = min(movs, key=lambda o: abs(o[3][0]-a0[0]) + abs(o[3][1]-a0[1]))
        tc0 = target[3]                               # target pixel centroid
        appr = _nav_to(w, g, fd, tc0)
        if appr:
            ag, af, _ = appr
            sa = _anchor_xy(arr_of(af), w)
            into = (4 if tc0[0] > sa[0] else 3) if abs(tc0[0]-sa[0]) >= abs(tc0[1]-sa[1]) else (2 if tc0[1] > sa[1] else 1)
            into = into if into in w.movement else next(iter(w.movement))

            def centroid_of(arr, prev):
                ms = _small_objects(arr, w)
                if not ms:
                    return None
                return min(ms, key=lambda o: abs(o[3][0]-prev[0]) + abs(o[3][1]-prev[1]))[3]
            # push test: a REAL push needs the agent to ADVANCE into the vacated
            # cell AND the object to move with it. If walking in leaves the agent
            # blocked (it didn't move), the object is not pushable -- no push.
            _, pf = step(ag, into)
            tp = centroid_of(arr_of(pf), tc0)
            if tp and (abs(tp[0]-tc0[0]) + abs(tp[1]-tc0[1])) >= PITCH*0.6:
                found.append(Effect(into, "facing_object", "object_push"))
            # attach + comove + release test for each effect action
            for eff in (w.effects or w.actions):
                fg, ff = step(ag, into); fg, ff = step(fg, eff)
                ta = centroid_of(arr_of(ff), tc0)
                if ta is None:
                    continue
                av0 = _anchor_xy(arr_of(ff), w)
                for mv in w.movement:
                    mg, mf = step(fg, mv)
                    av1 = _anchor_xy(arr_of(mf), w); tm = centroid_of(arr_of(mf), ta)
                    if not av1 or tm is None or av1 == av0:
                        continue
                    da = (av1[0]-av0[0], av1[1]-av0[1]); dt = (tm[0]-ta[0], tm[1]-ta[1])
                    if abs(da[0]-dt[0]) + abs(da[1]-dt[1]) < PITCH*0.6 and (abs(dt[0])+abs(dt[1])) > PITCH*0.6:
                        found.append(Effect(eff, "facing_object", "object_attach_comove"))
                        rg, rf = step(fg, eff)   # release?
                        rc = centroid_of(arr_of(rf), ta)
                        found.append(Effect(eff, "carrying_object", "object_release"))
                        break

    # --- next to a barrier: does any action OPEN it (impassable cell frees up)? ---
    barr0 = _barrier_cells(arr_of(fd), w)
    if barr0:
        bc = next(iter(barr0))
        appr = _nav_to(w, g, fd, (bc[0]*PITCH + w.grid.phase[0], bc[1]*PITCH + w.grid.phase[1]))
        if appr:
            bg, bf, _ = appr
            for act in w.actions:
                _, nf = step(bg, act)
                if over(nf):
                    continue
                if len(_barrier_cells(arr_of(nf), w)) < len(barr0):
                    found.append(Effect(act, "facing_barrier", "barrier_open"))
    return found


VERB_LIBRARY = ["move", "push", "pick_up_and_carry", "open_gate", "no_effect"]


def bind_verbs(effects: List[Effect], model=None, use_llm=True) -> Tuple[Dict[str, dict], str]:
    """Summarise effects, let the local LLM name verbs, verify by signature."""
    by_sig = Counter(e.signature for e in effects)
    # ground truth from signatures (the verifier)
    verbs = {}
    if any(e.signature == "self_translate" for e in effects):
        verbs["move"] = {"actions": sorted({e.action for e in effects if e.signature == "self_translate"})}
    if any(e.signature == "object_attach_comove" for e in effects):
        verbs["pick_up_and_carry"] = {
            "actions": sorted({e.action for e in effects if e.signature in ("object_attach_comove", "object_release")})}
    if any(e.signature == "object_push" for e in effects):
        verbs["push"] = {"actions": sorted({e.action for e in effects if e.signature == "object_push"})}
    if any(e.signature == "barrier_open" for e in effects):
        verbs["open_gate"] = {"actions": sorted({e.action for e in effects if e.signature == "barrier_open"})}

    src = "interaction-verifier"
    if use_llm:
        try:
            import llm_binder
            obs = "\n".join(f"- action {e.action} when {e.context.replace('_',' ')}: {e.signature.replace('_',' ')}"
                            for e in effects)
            prompt = ("You are reverse-engineering a grid game. Probing the actions produced these "
                      "observations:\n" + (obs or "- no effects observed") +
                      "\n\nName the agent's capabilities by choosing verbs from " + repr(VERB_LIBRARY) +
                      ". Reply JSON {\"verbs\": [{\"verb\":..,\"actions\":[..]}]}.")
            out = llm_binder.ollama_json(prompt, **({} if model is None else {"model": model}))
            named = {v.get("verb") for v in (out.get("verbs", []) if out else []) if v.get("verb") in VERB_LIBRARY}
            agree = named == set(verbs)
            src = f"llm{'' if agree else ' [verifier-corrected]'} ({sorted(named)})"
        except Exception:
            pass
    if not verbs:
        verbs["no_effect"] = {}
    return verbs, src


def discover(game="wa30", use_llm=True, model=None, verbose=True, start=None, level=1):
    w = survey(game, use_llm=use_llm, model=model)
    effects = probe(w, start=start)
    verbs, src = bind_verbs(effects, model=model, use_llm=use_llm)
    if verbose:
        print(f"DISCOVERY on {game} L{level}: anchor=colour {w.anchor_color} "
              f"movement={sorted(w.movement)} effect-actions={list(w.effects)} "
              f"barriers={list(w.barrier_colors)}")
        print("  raw effects:")
        for e in effects:
            print(f"    action {e.action} @ {e.context}: {e.signature} {e.detail or ''}")
        print(f"  BOUND VERBS [{src}]:")
        for v, d in verbs.items():
            print(f"    {v}: {d}")
    return verbs, effects, w


def discover_per_level(game="wa30", up_to=2, use_llm=True, model=None):
    """Run the SAME discovery phase on each level in turn (L1, L2, ...). Levels are
    reached by cracking the prior ones with the grounded cone (gkm_crack)."""
    import gkm_crack as G
    discover(game, use_llm=use_llm, model=model, level=1)
    c = G.CarryConnector.build(game, model=model, use_llm=use_llm, verbose=False)
    g, fd = c.fresh()
    while c.level(fd) + 1 < up_to + 1:
        before = c.level(fd)
        g, fd, _ = G.gkm_cone(c, g, fd, before, deadline=time.time()+900, verbose=False)
        if c.level(fd) <= before:
            print(f"\n(could not crack L{before+1} to reach the next level for discovery)")
            return
        print()
        discover(game, use_llm=use_llm, model=model, start=(g, fd), level=c.level(fd)+1)


if __name__ == "__main__":
    import sys
    game = next((a for a in sys.argv[1:] if not a.startswith("--")), "wa30")
    use_llm = "--no-llm" not in sys.argv
    if "--per-level" in sys.argv:
        discover_per_level(game, up_to=2, use_llm=use_llm)
    else:
        discover(game, use_llm=use_llm)
