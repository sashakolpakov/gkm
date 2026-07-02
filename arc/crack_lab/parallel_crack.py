"""Parallel non-monotone cofibrant search to crack wa30 L2 (no Claude API,
local-only, no hand-coded dynamics, game-agnostic goal).

- DYNAMICS: only the real game step (clone + perform_action). No model.
- GOAL (game-agnostic, anti-covering): maximise footprint ∩ ring-colour, where
  the container (ring, interior) is DETECTED generically (priors.containers).
  Covering a slot with the avatar/a frame adds no ring-colour, so it is not
  rewarded — no push/delivery/Sokoban schema, just "make the container's slots
  become the ring colour."
- COFIBRATION (correct): non-monotone full-state search — NO monotone pressure,
  so it may complicate (un-fill) before it unknots; cycle-safe via frame dedup.
- SPEED: the one constraint-compliant lever left — fan the search across CPU
  cores with per-worker diversity; first worker to reach the real win wins.
"""
from __future__ import annotations
import copy, heapq, random, time, sys
import multiprocessing as mp
import numpy as np

GAME = "wa30"
NAME = {0: "RESET", 1: "ACTION1", 2: "ACTION2", 3: "ACTION3", 4: "ACTION4", 5: "ACTION5"}
HEAP_CAP = 1200
MAX_DEPTH = 90


def frame_arr(fd):
    return np.asarray(fd.frame[-1])


def best_first(make_env, g0, fd0, footprint, ring, level0, moves, budget, time_cap, seed, jitter):
    """Non-monotone full-state best-first toward maximise footprint∩ring; the win
    is the real levels_completed rising above level0. Returns ('win', path) |
    ('best', best_ring_in_fp, nodes)."""
    from arcengine import ActionInput, GameAction as EA
    rng = random.Random(seed)
    fp = footprint

    def neg_ring(arr):
        return -sum(1 for (x, y) in fp if arr[y][x] == ring)   # minimise -> more ring in slots

    a0 = frame_arr(fd0)
    seen = {a0.tobytes()}
    heap = [(neg_ring(a0), 0, 0, g0, ())]
    ctr = 1; nodes = 0; t0 = time.time(); best = neg_ring(a0)
    while heap and nodes < budget:
        if time.time() - t0 > time_cap:
            break
        _, _, d, g, path = heapq.heappop(heap)
        if d >= MAX_DEPTH:
            continue
        for a in moves:
            gc = copy.deepcopy(g)
            fd = gc.perform_action(ActionInput(id=EA[NAME[a]]), raw=True)
            nodes += 1
            if fd.levels_completed > level0:
                return ("win", path + (a,))
            if str(fd.state).endswith("GAME_OVER"):
                continue
            arr = frame_arr(fd); k = arr.tobytes()
            if k in seen:
                continue
            seen.add(k)
            gv = neg_ring(arr)
            if gv < best:
                best = gv
            heapq.heappush(heap, (gv + 0.02 * (d + 1) + jitter * rng.random(), ctr, d + 1, gc, path + (a,))); ctr += 1
        if len(heap) > HEAP_CAP:
            heap = heapq.nsmallest(HEAP_CAP, heap); heapq.heapify(heap)
    return ("best", best, nodes)


def _make():
    from lab import make_env
    return make_env(GAME)


def _replay_to(g, fd_reset_fn, path):
    from arcengine import ActionInput, GameAction as EA
    fd = g.perform_action(ActionInput(id=EA.RESET), raw=True)
    for a in path:
        fd = g.perform_action(ActionInput(id=EA[NAME[a]]), raw=True)
    return fd


def worker(args):
    seed, l1_path, ring, region, budget, time_cap = args
    make_env = _make()
    e = make_env(); e.reset(); g = copy.deepcopy(e._env._game)
    fd = _replay_to(g, None, l1_path)                    # g,fd now at L2 start (level 1)
    arr = frame_arr(fd)
    fp = [(x, y) for y in range(arr.shape[0]) for x in range(arr.shape[1]) if arr[y][x] == region]
    r = best_first(make_env, g, fd, fp, ring, 1, [1, 2, 3, 4, 5], budget, time_cap, seed, 1e-6 * (seed + 1))
    if r[0] == "win":
        return ("win", list(l1_path) + list(r[1]))
    return ("best", r[1], seed)


def main():
    import priors
    from arcengine import ActionInput, GameAction as EA
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    L2_BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 120000
    L2_TIME = float(sys.argv[3]) if len(sys.argv) > 3 else 480.0
    make_env = _make()
    e = make_env(); e.reset(); g = copy.deepcopy(e._env._game)
    fd = g.perform_action(ActionInput(id=EA.RESET), raw=True)
    a0 = frame_arr(fd).tolist()
    struct = set(priors.structure_colours(a0))
    cont = [(r, i) for (r, i, _c) in priors.containers(a0) if r not in struct and i not in struct]
    cont.sort(key=lambda t: sum(1 for row in a0 for v in row if v == t[1]))
    ring, region = cont[0]
    print(f"detected container: ring={ring} interior(region)={region}")
    fp1 = [(x, y) for y in range(len(a0)) for x in range(len(a0[0])) if a0[y][x] == region]

    t0 = time.time()
    r = best_first(make_env, g, fd, fp1, ring, 0, [1, 2, 3, 4, 5], 60000, 180.0, 0, 0.0)
    if r[0] != "win":
        print(f"L1 not cleared (best ring-in-fp={-r[1]}); abort."); return
    l1_path = list(r[1])
    print(f"L1 cleared via generic goal: prefix len={len(l1_path)} ({time.time()-t0:.0f}s)")

    args = [(seed, l1_path, ring, region, L2_BUDGET, L2_TIME) for seed in range(N)]
    print(f"L2: fanning {N} non-monotone workers (cores)...")
    with mp.Pool(N) as pool:
        for res in pool.imap_unordered(worker, args):
            if res[0] == "win":
                print(f"  *** L2 CRACKED in parallel! path_len={len(res[1])} ({time.time()-t0:.0f}s)")
                print("PATH=", res[1])
                pool.terminate()
                return
            else:
                print(f"  worker seed={res[2]} best ring-in-fp={-res[1]} (no crack, {time.time()-t0:.0f}s)")
    print(f"\nL2 NOT cracked by any worker ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
