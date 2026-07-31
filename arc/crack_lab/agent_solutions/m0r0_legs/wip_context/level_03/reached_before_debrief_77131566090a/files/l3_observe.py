"""Compact clean-room observations for the current level-3 start state."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from legs import MIRRORED_PAIR_ASCENT, MIRRORED_PAIR_MAZE_REUNION
import numpy as np

from perception import action_deltas, bounded_bfs, color_counts, connected_components, replay


def reach_level_3(env):
    for name, route in (
        ("level1", MIRRORED_PAIR_ASCENT),
        ("level2", MIRRORED_PAIR_MAZE_REUNION),
    ):
        probe = env.clone()
        for action in route[:-1]:
            probe.step(action)
        before = probe.frame()
        before_level = probe.levels_completed
        probe.step(route[-1])
        print("reward_boundary", name, "action", route[-1],
              "levels", (before_level, probe.levels_completed),
              "delta", (action_deltas_from(before, probe.frame())))
        print("reward_objects", name, "before",
              [(b.color, b.bbox, b.area) for b in
               connected_components(before, colors=(9, 10), min_area=2)])
        for action in route:
            env.step(action)


def action_deltas_from(before, after):
    a, b = np.asarray(before), np.asarray(after)
    changed = np.argwhere(a != b)
    if len(changed) == 0:
        return 0, None
    return len(changed), (
        int(changed[:, 0].min()), int(changed[:, 1].min()),
        int(changed[:, 0].max()), int(changed[:, 1].max()),
    )


def observe(env):
    reach_level_3(env)
    blobs = connected_components(env.frame(), min_area=2)
    print("level", env.levels_completed, "actions", env.actions)
    print("colors", color_counts(env.frame()))
    print("blobs", [(b.color, b.bbox, b.area) for b in blobs])
    print("deltas", {a: (d["count"], d["bbox"]) for a, d in action_deltas(env, env.actions).items()})
    for action in env.actions:
        clone = env.clone()
        clone.step(action)
        pieces = connected_components(clone.frame(), colors=(9, 10), min_area=2)
        print("after", action, [(b.color, b.bbox, b.area) for b in pieces])
    frame = np.asarray(env.frame())
    chars = {5: "#", 8: ".", 9: "G", 10: "A", 15: " "}
    print("map")
    for r in range(0, 64, 4):
        row = []
        for c in range(0, 64, 4):
            vals, counts = np.unique(frame[r:r + 4, c:c + 4], return_counts=True)
            row.append(chars.get(int(vals[counts.argmax()]), "?"))
        print("".join(row))
    print("walk_cells")
    for r in range(2, 63, 4):
        row = []
        for c in range(2, 63, 4):
            tile = set(int(v) for v in np.unique(frame[r:r + 4, c:c + 4]))
            if 9 in tile:
                row.append("G")
            elif 10 in tile:
                row.append("A")
            elif 5 in tile:
                row.append(".")
            else:
                row.append(" ")
        print("".join(row))
    trials = {
        "left_goal": [4, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1],
        "right_goal": [1, 3, 3, 1, 1, 1, 4, 4],
    }
    for name, path in trials.items():
        for suffix in ([], [5], [6]):
            node = replay(env, path + suffix)
            pieces = connected_components(node.frame(), colors=(9, 10), min_area=2)
            print("trial", name, suffix, "level", node.levels_completed,
                  "colors", {k: color_counts(node.frame()).get(k, 0) for k in (9, 10)},
                  "pieces", [(b.color, b.bbox, b.area) for b in pieces])
    goals = [b.bbox for b in connected_components(env.frame(), colors=(9,), min_area=2)]

    def touches(node, goal):
        f = np.asarray(node.frame())
        r0, c0, r1, c1 = goal
        ring = np.concatenate((
            f[r0 - 1, c0:c1 + 1],
            f[r1 + 1, c0:c1 + 1],
            f[r0:r1 + 1, c0 - 1],
            f[r0:r1 + 1, c1 + 1],
        ))
        return bool(np.any(ring == 10))



if __name__ == "__main__":
    levels, path, err = A.run_program("m0r0", observe)
    print("run", levels, len(path), err)
