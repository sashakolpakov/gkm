import heapq
import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr


CONTROLS = (
    ("A<", (6, 4, 51)), ("A>", (6, 10, 51)), ("B", (6, 17, 51)),
    ("C<", (6, 26, 51)), ("C>", (6, 32, 51)), ("D", (6, 39, 51)),
    ("F<", (6, 4, 58)), ("F>", (6, 10, 58)), ("G", (6, 17, 58)),
    ("H<", (6, 26, 58)), ("H>", (6, 32, 58)),
)
LOOKUP = dict(CONTROLS)
PREFIX = (
    "F<", "A>", "B", "B", "A>", "F>", "A>", "C>", "A>", "C>",
    "A>", "D",
    "H>", "H>", "H>", "H>", "H>",
    "H<", "H<", "H<", "H<", "H<",
    "C<", "C<", "C<",
    "F>", "F>", "F>", "F>",
    "D", "D",
    "C>", "C>", "C>", "C>",
    "A>", "A>", "A>", "D", "C>", "C>", "C>",
    "H>", "H>", "H>", "H>", "H>", "H>", "H>", "H>",
    "H<", "H<", "H<", "H<", "H<",
    "F>", "F>",
    "C<", "F>", "C<", "F>", "C<", "F>",
    "C<", "F>", "C<", "F>", "C<", "F>",
)
RINGS = {
    (6, 22), (7, 21), (7, 23), (8, 22),
    (15, 25), (16, 24), (16, 26), (17, 25),
}


def key(env):
    return arr(env.frame())[:42].tobytes()


def cells(frame, color):
    grid = arr(frame)[:42]
    return tuple((int(r), int(c)) for r, c in zip(*((grid == color).nonzero())))


def joint(frame, first, second):
    one, two = cells(frame, first), cells(frame, second)
    if not one or not two:
        return (99, 99)
    a, b = min(
        ((p, q) for p in one for q in two),
        key=lambda pair: abs(pair[0][0] - pair[1][0])
        + abs(pair[0][1] - pair[1][1]),
    )
    return ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)


def features(frame):
    upstream = joint(frame, 11, 14)
    outer = joint(frame, 14, 9)
    inner = joint(frame, 9, 12)
    marker = tuple(
        p for p in cells(frame, 13)
        if p not in RINGS and p != (7, 22)
    )
    return upstream, outer, inner, marker[0] if len(marker) == 1 else (99, 99)


def score(frame):
    upstream, outer, inner, marker = features(frame)
    if 99 in upstream + outer + inner + marker:
        return 999
    return abs(inner[0] - 16) + abs(inner[1] - 8)


def search(root, max_states=8000, max_depth=55):
    serial = 0
    queue = [(score(root.frame()), 0, serial, root.clone(), ())]
    seen = {key(root)}
    best = score(root.frame())
    while queue and len(seen) < max_states:
        _, depth, _, node, path = heapq.heappop(queue)
        if depth >= max_depth:
            continue
        for index, (_, action) in enumerate(CONTROLS):
            child = node.clone()
            before = key(child)
            child.step(*action)
            child_key = key(child)
            if child_key == before or child_key in seen:
                continue
            seen.add(child_key)
            child_path = path + (index,)
            if child.levels_completed > 6:
                return child_path, len(seen)
            child_score = score(child.frame())
            if child_score < best:
                best = child_score
                print(
                    "best", best, features(child.frame()),
                    [CONTROLS[i][0] for i in child_path],
                    flush=True,
                )
            serial += 1
            heapq.heappush(
                queue,
                (child_score + len(child_path) // 4, depth + 1, serial,
                 child, child_path),
            )
    return None, len(seen)


def run(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    for name in PREFIX:
        env.step(*LOOKUP[name])
    print("root", features(env.frame()), score(env.frame()), flush=True)
    path, states = search(env)
    print("states", states)
    print("path", None if path is None else [CONTROLS[i][0] for i in path])


arena.run_program("s5i5", run)
