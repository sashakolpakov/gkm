"""Crack wa30 via the cofibrant anchor + logical perception, and DISCOVER the
winning state (don't assume it).

Pipeline:
  1. cofibrant anchor + verified move-rule (cofibrant.identify_anchor).
  2. generic container detection (priors.containers -> ring/interior); the goal
     heuristic is "lock the ring colour into the interior footprint" (delivery,
     anti-covering) -- a perceptual prior, NOT the reward.
  3. best-first search on the REAL game (clone-based, O(1) expansion), all actions
     incl. ACTION5; dedup on full frame; reward = levels_completed (the only truth).
  4. when levels rises -> CAPTURE the win transition and CHARACTERISE it with the
     logical perception (which action, what changed in logical objects) = the
     discovered win mechanic. Then VALIDATE by replay on a fresh env.

Usage: python3 crack_wa30.py [level_cap=1] [budget=80000] [time=240]
"""
from __future__ import annotations
import copy, heapq, sys, time
from collections import Counter
import numpy as np

from lab import make_env
import priors
from logical_grid import Grid, objects
from cofibrant import identify_anchor
from anchor_connector import directed_probe
from l2_attack import deliver_level

NAME = {0: "RESET", 1: "ACTION1", 2: "ACTION2", 3: "ACTION3", 4: "ACTION4", 5: "ACTION5"}
DIR = {(0, -1): "UP", (0, 1): "DOWN", (-1, 0): "LEFT", (1, 0): "RIGHT", (0, 0): "stay"}


def arr_of(fd):
    return np.asarray(fd.frame[-1])


def detect_container(a0):
    struct = set(priors.structure_colours(a0.tolist()))
    cont = [(r, i) for (r, i, _c) in priors.containers(a0.tolist()) if r not in struct and i not in struct]
    cont.sort(key=lambda t: sum(1 for row in a0 for v in row if v == t[1]))
    return cont[0] if cont else (None, None)


def anchor_cell(arr, grid, anchor, prev=None):
    if anchor is None:
        return None
    cell = anchor.locate(arr, grid, prev_cell=prev or anchor.seed)
    if cell is not None:
        return cell
    same = [o.cell for o in objects(arr, grid, [anchor.color])]
    if not same:
        return None
    ref = prev or anchor.seed
    return min(same, key=lambda c: abs(c[0] - ref[0]) + abs(c[1] - ref[1]))


def characterise_win(before, after, action, grid, anchor):
    """Discover what the winning transition DID, in logical terms."""
    bobj = {(o.color, o.cell) for o in objects(before, grid)}
    aobj = {(o.color, o.cell) for o in objects(after, grid)}
    appeared = aobj - bobj
    vanished = bobj - aobj
    av_b = anchor_cell(before, grid, anchor)
    av_a = anchor_cell(after, grid, anchor, prev=av_b)
    mv = anchor.vectors.get(action) if anchor else None
    print("  --- DISCOVERED WIN MECHANIC ---")
    print(f"   trigger action : ACTION{action} (avatar effect: {DIR.get(mv, 'interact/none')})")
    print(f"   avatar cell    : {av_b} -> {av_a}")
    chg = Counter(c for c, _ in (appeared | vanished))
    print(f"   logical change : {sum(chg.values())} cell-changes over colours {dict(chg)}")
    print(f"   appeared(colour@cell): {sorted(appeared)[:8]}")
    print(f"   vanished(colour@cell): {sorted(vanished)[:8]}")


def crack(game="wa30", level_cap=1, budget=80000, time_cap=240.0):
    from arcengine import ActionInput, GameAction as EA
    e = make_env(game)(); e.reset(); g = copy.deepcopy(e._env._game)
    fd = g.perform_action(ActionInput(id=EA.RESET), raw=True)
    grid = Grid.infer(arr_of(fd))
    seqs, start = directed_probe(make_env(game))
    anchor = identify_anchor(seqs, grid)
    print(f"{game}: grid={grid}  anchor=colour {anchor.color if anchor else None} "
          f"move-rule={ {a: DIR.get(v) for a,v in (anchor.vectors.items() if anchor else [])} }")
    dirs = ({a: v for a, v in anchor.vectors.items() if v != (0, 0) and a in (1, 2, 3, 4)}
            if anchor else {})

    full_path = []
    while fd.levels_completed < level_cap:
        a0 = arr_of(fd); level0 = fd.levels_completed
        ring, region = detect_container(a0)
        if ring is None:
            print(f"L{level0+1}: no container detected; abort"); break
        if level0 == 0:
            fp = [(x, y) for y in range(a0.shape[0]) for x in range(a0.shape[1]) if a0[y][x] == region]
            print(f"\nL{level0+1}: container ring={ring} interior={region} slots={len(fp)} -- searching")
            path, _ = best_first(g, fd, fp, ring, level0, budget, time_cap, grid, anchor)
            if path is None:
                print(f"L{level0+1} NOT cracked; stop."); break
            for a in path:                               # advance the real game to the won state
                fd = g.perform_action(ActionInput(id=EA[NAME[a]]), raw=True)
            full_path += path
        else:
            print(f"\nL{level0+1}: switching to object-relative leg attack")
            prefix = []
            status, g, fd = deliver_level(g, fd, dirs, prefix)
            full_path += prefix
            if status != "win":
                print(f"L{level0+1} NOT cracked ({status}); stop."); break
        print(f"L{level0+1} cleared -> levels_completed={fd.levels_completed} (cum path={len(full_path)})")

    if not full_path:
        return None
    reached = fd.levels_completed
    ok = validate(game, full_path, reached)
    print(f"\nReached level {reached}; cum path_len={len(full_path)}  VALIDATED={ok}")
    print(f"PATH={full_path}")
    return full_path


def best_first(g0, fd0, fp, ring, level0, budget, time_cap, grid, anchor):
    from arcengine import ActionInput, GameAction as EA

    def ring_in_fp(arr):
        return sum(1 for (x, y) in fp if arr[y][x] == ring)

    a0 = arr_of(fd0)
    # anti-covering delivery goal: MAXIMISE ring locked in footprint -> minimise neg
    seen = {a0.tobytes()}
    start_score = ring_in_fp(a0)
    heap = [(-start_score, 0, 0, g0, ())]
    ctr = 1; nodes = 0; t0 = time.time(); best = start_score
    while heap and nodes < budget:
        if time.time() - t0 > time_cap:
            break
        neg, _, d, g, path = heapq.heappop(heap)
        if d >= 90:
            continue
        for a in (1, 2, 3, 4, 5):
            gc = copy.deepcopy(g)
            fd = gc.perform_action(ActionInput(id=EA[NAME[a]]), raw=True)
            nodes += 1
            if fd.levels_completed > level0:
                before = _frame_before(g0, path)        # pre-win frame (replay the prefix)
                print(f"\n*** WIN at node {nodes} ({time.time()-t0:.0f}s), path_len={len(path)+1}")
                characterise_win(before, arr_of(fd), a, grid, anchor)
                return list(path) + [a], arr_of(fd)
            if str(fd.state).endswith("GAME_OVER"):
                continue
            arr = arr_of(fd); k = arr.tobytes()
            if k in seen:
                continue
            seen.add(k)
            sc = ring_in_fp(arr)
            best = max(best, sc)
            heapq.heappush(heap, (-sc + 0.02 * (d + 1), ctr, d + 1, gc, path + (a,))); ctr += 1
        if len(heap) > 1500:
            heap = heapq.nsmallest(1500, heap); heapq.heapify(heap)
    print(f"   best ring-in-fp reached: {best} (nodes={nodes}, {time.time()-t0:.0f}s)")
    return None, None


def _frame_before(g0, path):
    """Replay `path` (the prefix before the winning action) on a fresh clone to get
    the pre-win frame."""
    from arcengine import ActionInput, GameAction as EA
    g = copy.deepcopy(g0); fd = g.perform_action(ActionInput(id=EA.RESET), raw=True)
    for a in path:
        fd = g.perform_action(ActionInput(id=EA[NAME[a]]), raw=True)
    return arr_of(fd)


def validate(game, path, expect_level):
    """Replay the full path on a FRESH env; confirm levels_completed reaches it."""
    from arcengine import ActionInput, GameAction as EA
    e = make_env(game)(); e.reset(); g = copy.deepcopy(e._env._game)
    fd = g.perform_action(ActionInput(id=EA.RESET), raw=True)
    for a in path:
        fd = g.perform_action(ActionInput(id=EA[NAME[a]]), raw=True)
    return fd.levels_completed >= expect_level


if __name__ == "__main__":
    lvl = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    bud = int(sys.argv[2]) if len(sys.argv) > 2 else 80000
    tc = float(sys.argv[3]) if len(sys.argv) > 3 else 240.0
    crack("wa30", lvl, bud, tc)
