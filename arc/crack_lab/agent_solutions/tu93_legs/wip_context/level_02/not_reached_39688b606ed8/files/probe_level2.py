import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from legs import drive_block_maze, parse_block_maze, maze_path_actions
from perception import action_deltas, color_counts, connected_components
import numpy as np


def compact_blobs(frame):
    return [
        (b.color, b.bbox, b.area)
        for b in connected_components(frame, min_area=2)
        if b.area < 2000
    ]


def probe(env):
    drive_block_maze(env)
    print("level", env.levels_completed, "actions", env.actions)
    print("colors", color_counts(env.frame()))
    print("blobs", compact_blobs(env.frame()))
    graph = parse_block_maze(env.frame())
    print("maze", None if graph is None else
          (graph["start"], graph["goal"], len(graph["nodes"])))
    print("plan", maze_path_actions(env.frame()))
    for action, delta in action_deltas(env, env.actions).items():
        print("action", action, "count", delta["count"], "bbox", delta["bbox"],
              "samples", delta["samples"][:12])
    clone = env.clone()
    won = drive_block_maze(clone)
    print("existing_leg", won, clone.levels_completed)
    clone = env.clone()
    for i, action in enumerate(maze_path_actions(env.frame()) or []):
        clone.step(action)
        f = np.asarray(clone.frame())
        singles = [(int(v), tuple(map(int, p)))
                   for v in (4, 15)
                   for p in np.argwhere(f == v)]
        g = parse_block_maze(f)
        print("step", i + 1, action, "lvl", clone.levels_completed,
              "singles", singles,
              "parse", None if g is None else (g["start"], g["goal"]),
              "rare", {k: v for k, v in color_counts(f).items()
                       if v < 20})


print(A.run_program("tu93", probe)[:1])
