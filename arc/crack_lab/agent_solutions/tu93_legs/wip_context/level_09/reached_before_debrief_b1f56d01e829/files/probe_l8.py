import importlib.util
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import drive_dynamic_directional_waypoints, parse_block_maze
from perception import action_deltas, color_counts, connected_components


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


def probe(env):
    solver.solve(env)
    print("LEVEL", env.levels_completed, "ACTIONS", tuple(env.actions))
    print("COLORS", color_counts(env.frame()))
    blobs = connected_components(env.frame(), min_area=2)
    print("BLOBS", [(b.color, b.bbox, b.area) for b in blobs if b.color != 2])
    print("DELTAS", {
        action: (delta["count"], delta["bbox"], delta["samples"][:8])
        for action, delta in action_deltas(env, env.actions).items()
    })
    for cell in (2, 3, 4, 5):
        graph = parse_block_maze(env.frame(), cell)
        if graph:
            print("MAZE", cell, graph["origin"], graph["marker"], graph["open"],
                  graph["start"], graph["goal"], len(graph["nodes"]))
            if cell == 3:
                print("NODES", sorted(graph["nodes"].items()))
                print("EDGES", {
                    node: graph["neigh"](*node) for node in sorted(graph["nodes"])
                })
    trial = env.clone()
    success = drive_dynamic_directional_waypoints(
        trial, waypoint_color=8, marker_color=15, avatar_color=9,
        max_states=300, max_depth=50)
    print("EXISTING_LEG", success, trial.levels_completed, trial.terminal())


levels, path, err = arena.run_program("tu93", probe)
print("PROBE_RESULT", levels, len(path), err)
