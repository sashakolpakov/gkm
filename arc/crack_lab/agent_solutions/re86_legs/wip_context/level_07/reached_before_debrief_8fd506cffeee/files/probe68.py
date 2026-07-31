import heapq
import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import DOWN, LEFT, RIGHT, UP, arr
from probe48 import NAME, bare_frame, descriptor


PATH = json.load(open("checkpoint.json"))["final_path"]
TARGETS = ((30, 45), (48, 39), (48, 51))
DELTA = {UP: (-3, 0), DOWN: (3, 0), LEFT: (0, -3), RIGHT: (0, 3)}
OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}


def desc_key(desc):
    return desc[0], desc[1], desc[2], tuple(sorted(desc[3]))


def segment_distance(point, axes):
    row, col = point
    r0, cr, r1, c0, cc, c1 = axes
    return min(
        abs(col - cc) + max(r0 - row, 0, row - r1),
        abs(row - cr) + max(c0 - col, 0, col - c1),
    )


def search(root, bare, max_states=16000, max_depth=60):
    center = (48, 12)
    start = descriptor(arr(root.frame()), bare, "small-cross", center)
    queue = [(
        sum(segment_distance(target, start[2]) for target in TARGETS),
        0,
        0,
        root.clone(),
        (),
        center,
        start,
    )]
    seen = {desc_key(start): 0}
    serial = 0
    best = (999, None, ())
    while queue and len(seen) < max_states:
        _, depth, _, node, path, center, old = heapq.heappop(queue)
        if depth >= max_depth:
            continue
        for action in (UP, DOWN, LEFT, RIGHT):
            if path and action == OPPOSITE[path[-1]]:
                continue
            child = node.clone()
            child.step(action)
            dr, dc = DELTA[action]
            child_center = center[0] + dr, center[1] + dc
            if not (
                0 <= child_center[0] < 63
                and 0 <= child_center[1] < 64
            ):
                continue
            desc = descriptor(
                arr(child.frame()), bare, "small-cross", child_center
            )
            if desc is None:
                continue
            next_depth = depth + 1
            state = desc_key(desc)
            if seen.get(state, 999) <= next_depth:
                continue
            seen[state] = next_depth
            next_path = path + (action,)
            distances = tuple(
                segment_distance(target, desc[2]) for target in TARGETS
            )
            value = sum(distances)
            if value < best[0]:
                best = value, desc[:3], next_path
                print(
                    "BEST",
                    value,
                    desc[:3],
                    distances,
                    len(next_path),
                    "".join(NAME[action] for action in next_path),
                    flush=True,
                )
            if value == 0:
                print(
                    "SOLVED",
                    desc[0],
                    len(next_path),
                    "".join(NAME[action] for action in next_path),
                    desc[:3],
                    len(seen),
                    flush=True,
                )
                return next_path
            serial += 1
            heapq.heappush(
                queue,
                (
                    next_depth + 3 * value,
                    next_depth,
                    serial,
                    child,
                    next_path,
                    child_center,
                    desc,
                ),
            )
    print(
        "FAILED",
        len(seen),
        best[0],
        "".join(NAME[action] for action in best[2]),
        flush=True,
    )


def run(env):
    for action in PATH:
        env.step(action)
    bare = bare_frame(env)
    solution = search(env, bare)
    if solution:
        print("FULL", solution, flush=True)


A.run_program("re86", run)
