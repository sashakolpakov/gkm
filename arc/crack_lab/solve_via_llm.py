"""End-to-end universal pipeline: LOCAL LLM binds verbs to the game's colours →
deterministic clone-based search VERIFIES each binding against levels_completed.

The LLM need not be perfect: it proposes verb + relevant colours; we expand each
proposal into sensible variants (e.g. carrier<->region swap, constrained to real
container-interiors and movable colours) and the search disambiguates by which
binding actually increases levels_completed. Robustness via verification — the
whole point of propose->verify. No game/colour constants in this file.
"""
from __future__ import annotations
import copy, heapq, random, sys, time
import numpy as np
from lab import arc, make_env
from priors import detect_avatar_color
import proposer, llm_binder
from llm_binder import BoundVerb, attach_goal_fn
from arcengine import ActionInput, GameAction as EA

GAME = sys.argv[1] if len(sys.argv) > 1 else "wa30"
MODEL = sys.argv[2] if len(sys.argv) > 2 else llm_binder.DEFAULT_MODEL
NAME = {1: "ACTION1", 2: "ACTION2", 3: "ACTION3", 4: "ACTION4", 5: "ACTION5"}
ROUND_BUDGET, SUB_DEPTH, RESTARTS, MAXHEAP = 15000, 55, 3, 12000


def root():
    e = make_env(GAME)(); s = e.reset()
    return e._env._game, s.win_levels, s.frame, (e.available_actions or [1, 2, 3, 4, 5])


def grid(fd):
    return np.asarray(fd.frame[-1])


def step(g, a):
    gc = copy.deepcopy(g)
    return gc, gc.perform_action(ActionInput(id=EA[NAME[a]]), raw=True)


def expand(bound, scene):
    """Variant expansion so the verifier can disambiguate imperfect LLM bindings."""
    interiors = {i for (_, i, _) in scene["containers"]}
    nonstruct = set(scene["colour_counts"]) - set(scene["structure_colours"])
    seen, out = set(), []
    for bv in bound:
        cands = [bv]
        if bv.verb == "transport":
            c, r = bv.params.get("carrier"), bv.params.get("region")
            cands = []
            for (cc, rr) in [(c, r), (r, c)]:
                if rr in interiors and cc in nonstruct and cc != rr:
                    cands.append(BoundVerb(f"transport({cc}->{rr})", "transport",
                                           {"carrier": cc, "region": rr}, bv.rationale))
            if not cands:  # fall back to the raw proposal
                cands = [bv]
        for c in cands:
            key = (c.verb, tuple(sorted(c.params.items())))
            if key not in seen:
                seen.add(key); out.append(c)
    return out


def verify(anchor_game, anchor_arr, level, goal_fn, moves, t0):
    """Clone-based per-box search (locked-primary goal) with stochastic restarts."""
    cur_g, cur_a = anchor_game, anchor_arr
    cur_h = goal_fn(cur_a)
    for _ in range(40):
        improved = False
        for seed in range(RESTARTS):
            rng = random.Random(seed)
            jitter = 0.0 if seed == 0 else 1e-6   # restart 0 deterministic; later restarts diversify
            seen = {cur_a.tobytes()}
            heap = [(cur_h, 0, 0, cur_g)]; ctr = 1; best = (cur_h, cur_g, cur_a); nodes = 0
            while heap and nodes < ROUND_BUDGET:
                _, _, d, game = heapq.heappop(heap)
                if d >= SUB_DEPTH:
                    continue
                for a in moves:
                    gc, fd = step(game, a); nodes += 1
                    if fd.levels_completed > level:
                        return ("win", gc, fd)
                    if str(fd.state).endswith("GAME_OVER"):
                        continue
                    arr = grid(fd); k = arr.tobytes()
                    if k in seen:
                        continue
                    seen.add(k)
                    gv = goal_fn(arr)
                    if gv < best[0]:
                        best = (gv, gc, arr)
                    heapq.heappush(heap, (gv + 0.02 * (d + 1) + jitter * rng.random(), ctr, d + 1, gc)); ctr += 1
                if len(heap) > MAXHEAP:
                    heap = heapq.nsmallest(MAXHEAP, heap); heapq.heapify(heap)
            if best[0] < cur_h - 1e-9:
                cur_g, cur_a, cur_h = best[1], best[2], best[0]; improved = True
                break
        if not improved:
            return ("stuck", cur_g, cur_a, cur_h)
    return ("stuck", cur_g, cur_a, cur_h)


def main():
    g0, win_levels, start_frame, avail = root()
    moves = [a for a in avail if 1 <= a <= 5]
    avatar_color, _, _ = detect_avatar_color(make_env(GAME))
    scene = proposer.scene_summary(start_frame, avail)
    print(f"{GAME}: avatar={avatar_color} containers={scene['containers']}")
    print(f"calling local LLM ({MODEL}) to bind verbs ...")
    t0 = time.time()
    bound = llm_binder.bind(scene, avatar_color, win_levels, model=MODEL)
    print(f"LLM proposed ({time.time()-t0:.0f}s):")
    for bv in bound:
        print(f"  [{bv.verb}] {bv.name} params={bv.params}")
    cands = expand(bound, scene)
    print(f"verifier candidates after variant-expansion: {[(c.verb, c.params) for c in cands]}")

    cur_game, cur_arr, level = g0, np.asarray(start_frame), 0
    while level < win_levels:
        scene = proposer.scene_summary(cur_arr.tolist(), avail)
        # rebind per level not needed for wa30 (same game), reuse expanded cands
        progressed = False
        for bv in cands:
            attach_goal_fn(bv, cur_arr.tolist(), avatar_color)
            res = verify(cur_game, cur_arr, level, bv.goal_fn, moves, t0)
            if res[0] == "win":
                cur_game, fd = res[1], res[2]; level = fd.levels_completed; cur_arr = grid(fd)
                print(f"  *** LEVEL {level} CLEARED via LLM-bound [{bv.verb}] {bv.params} ({time.time()-t0:.0f}s)")
                progressed = True
                break
            else:
                print(f"    [{bv.verb}] {bv.params}: best goal={res[3]:.1f} (no crack)")
        if not progressed:
            print(f"  no LLM-bound goal cracked level {level+1}.")
            break
    print(f"\nRESULT: {GAME} levels={level}/{win_levels} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
