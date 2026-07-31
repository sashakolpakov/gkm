import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import (
    arr,
    bounded_bfs,
    color_counts,
    connected_components,
    object_candidates,
)


def compact_delta(before, after):
    a, b = arr(before)[:63], arr(after)[:63]
    changed = a != b
    if not changed.any():
        return (0, None, {})
    ys, xs = changed.nonzero()
    transitions = {}
    for y, x in zip(ys, xs):
        pair = (int(a[y, x]), int(b[y, x]))
        transitions[pair] = transitions.get(pair, 0) + 1
    return (
        len(ys),
        (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())),
        transitions,
    )


def non_background(env):
    return [
        (b.color, b.bbox, b.area)
        for b in connected_components(
            env.frame(), colors=(0, 4, 12, 13, 14), min_area=1
        )
        if b.bbox[0] < 63
    ]


def probe(env):
    for level in range(1, 5):
        getattr(players, f"play_level_{level}")(env)
    print("level", env.levels_completed, "actions", env.actions)
    print("colors", color_counts(env.frame()))
    objects = [
        (o["color"], o["bbox"], o["size"], o["area"])
        for o in object_candidates(env.frame(), min_area=4)
    ]
    print("objects", objects)
    base = arr(env.frame()).copy()
    print("pieces", non_background(env))
    print("arrow_deltas", {
        action: compact_delta(base, (lambda c: (c.step(action), c.frame())[1])(env.clone()))
        for action in (1, 2, 3, 4)
    })
    for x, y in ((28, 49), (49, 28), (27, 30), (27, 34)):
        selected = env.clone()
        before_select = arr(selected.frame()).copy()
        selected.step(6, x, y)
        selection_delta = compact_delta(before_select, selected.frame())
        effects = {}
        for action in (1, 2, 3, 4):
            moved = selected.clone()
            before_move = arr(moved.frame()).copy()
            moved.step(action)
            effects[action] = compact_delta(before_move, moved.frame())
        print("select", (x, y), selection_delta, "arrows", effects)
    path = bounded_bfs(
        env,
        lambda node, _: node.levels_completed > 4,
        actions=(1, 2, 3, 4),
        key_fn=lambda node: arr(node.frame())[:63].tobytes(),
        max_states=5000,
        max_depth=50,
    )
    print("bfs_path", path)
    if path:
        replayed = env.clone()
        trace = []
        for action in path:
            replayed.step(action)
            zeros = list(zip(*((arr(replayed.frame())[:63] == 0).nonzero())))
            trace.append((action, zeros[:2], replayed.levels_completed))
        print("trace", trace)
        print("verified", replayed.levels_completed > 4)
        for ups in (8, 9, 10, 11):
            variant = env.clone()
            for action in [1] * ups + [4] * 7 + [2]:
                variant.step(action)
            print("variant_ups", ups, "won", variant.levels_completed > 4)


levels, path, err = A.run_program("ka59", probe)
print("result", levels, len(path), err)
