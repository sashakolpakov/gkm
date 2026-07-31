"""Bounded reward probe for raw-observation g50t level 6."""
import importlib.util
import json
import os
import sys
import time
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

import perception as p
import legs

spec = importlib.util.spec_from_file_location("workspace_solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


def mover_key(env):
    frame = p.arr(env.frame())
    movers = p.connected_components(frame, colors=(9, 14), min_area=20)
    compact = tuple((b.color, b.bbox) for b in movers
                    if b.size[0] <= 7 and b.size[1] <= 7)
    header = frame[:5, :13].tobytes()
    return int(env.levels_completed), compact, header


def actor_positions(env):
    blobs = p.connected_components(env.frame(), colors=(9, 14), min_area=18)
    player = [b.top_left for b in blobs
              if b.color == 9 and b.area >= 20
              and b.size[0] <= 7 and b.size[1] <= 7]
    npc = [b.top_left for b in blobs
           if b.color == 14 and b.size[0] <= 7 and b.size[1] <= 7]
    return (player[0] if player else None, npc[0] if npc else None)


def special_state(env):
    return tuple((b.color, b.bbox, b.area) for b in
                 p.connected_components(env.frame(), colors=(8, 11),
                                        min_area=4))


def heading_bfs(env, max_states=6000, max_depth=100):
    base = int(env.levels_completed)
    start = env.clone()
    _, npc0 = actor_positions(start)
    q = deque([(start, [], None)])
    seen = {(actor_positions(start), None, mover_key(start)[2],
             special_state(start))}
    while q and len(seen) <= max_states:
        node, path, prev_npc = q.popleft()
        if len(path) >= max_depth:
            continue
        _, current_npc = actor_positions(node)
        for action in (1, 2, 3, 4, 5):
            child = node.clone()
            child.step(action)
            child_pos = actor_positions(child)
            key = (child_pos, current_npc, mover_key(child)[2],
                   special_state(child))
            if key in seen:
                continue
            child_path = path + [action]
            if int(child.levels_completed) > base:
                print("SEARCH_STATS", len(seen), len(child_path), "WIN")
                return child_path
            seen.add(key)
            q.append((child, child_path, current_npc))
    print("SEARCH_STATS", len(seen), len(q), "EXHAUST_OR_LIMIT")
    return None


def search_level_6(env):
    t0 = time.time()
    walk = env.clone()
    trace = []
    for i in range(32):
        blobs = [b for b in p.connected_components(
            walk.frame(), colors=(14,), min_area=20)
                 if b.size[0] <= 7 and b.size[1] <= 7]
        trace.append(blobs[0].top_left if blobs else None)
        walk.step((3, 4)[i % 2])
    print("NPC_TRACE", trace)
    meter = env.clone()
    for i in range(160):
        if meter.terminal() or int(meter.levels_completed) > int(env.levels_completed):
            break
        meter.step((3, 4)[i % 2])
    bottom = p.color_counts(meter.frame()).get(9, 0)
    print("METER_TEST", i + 1, int(meter.levels_completed),
          bool(meter.terminal()), bottom)
    path = None
    base = int(env.levels_completed)
    path = heading_bfs(env)
    print("BFS6", {"path": path, "seconds": round(time.time() - t0, 3)})
    if path:
        for action in path:
            env.step(action)


solver.players.play_level_6 = search_level_6


def resumed_solver(env):
    checkpoint = None
    if os.path.exists("checkpoint.json"):
        with open("checkpoint.json") as handle:
            checkpoint = json.load(handle)
    if (checkpoint and checkpoint.get("game") == "g50t"
            and checkpoint.get("validated") and checkpoint.get("final_path")):
        for action in checkpoint["final_path"]:
            env.step(action)
    solver.solve(env)


levels, path, err = arena.run_program("g50t", resumed_solver)
print("PROBE_RESULT", {"levels": levels, "moves": len(path), "err": str(err)})
