import heapq
import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import arr
from probe48 import (
    DOWN, LEFT, NAME, PATH, RIGHT, SPECS, UP,
    bare_frame, covered, descriptor, dot, heuristic,
)


BASE = tuple(
    [UP] * 10 + [RIGHT, UP, DOWN, DOWN] + [RIGHT] * 8
    + [UP] + [RIGHT] * 4 + [UP, DOWN]
)


def desc_key(desc):
    if desc is None:
        return None
    return desc[0], desc[1], desc[2], tuple(sorted(desc[3]))


def search(root, bare, kind, wanted, targets, max_states=12000, max_depth=36):
    frame = arr(root.frame())
    desc = descriptor(frame, bare, kind, dot(frame))
    serial = 0
    queue = [(heuristic(desc, kind, wanted, targets), 0, serial,
              root.clone(), (), desc)]
    seen = {desc_key(desc): 0}
    best = None
    while queue and len(seen) < max_states:
        _, depth, _, node, path, old_desc = heapq.heappop(queue)
        if depth >= max_depth:
            continue
        for action in (UP, DOWN, LEFT, RIGHT):
            child = node.clone()
            child.step(action)
            frame = arr(child.frame())
            center = dot(frame)
            desc = descriptor(frame, bare, kind, center)
            key = desc_key(desc)
            next_depth = depth + 1
            if seen.get(key, 999) <= next_depth:
                continue
            seen[key] = next_depth
            next_path = path + (action,)
            hits = len(covered(desc, kind, targets))
            score = (hits, bool(desc and desc[0] == wanted))
            if best is None or score > best[0]:
                best = (score, desc[:3] if desc else None, next_path)
                print("best", best[0], best[1], len(next_path),
                      "".join(NAME[a] for a in next_path), flush=True)
            if desc and desc[0] == wanted and hits == len(targets):
                print("SOLVED", len(next_path),
                      "".join(NAME[a] for a in next_path),
                      desc[:3], len(seen), flush=True)
                return next_path
            serial += 1
            cost = next_depth + 2 * heuristic(desc, kind, wanted, targets)
            heapq.heappush(
                queue, (cost, next_depth, serial, child, next_path, desc)
            )
    print("FAILED", len(seen), best, flush=True)
    return None


def run(env):
    for action in PATH:
        env.step(action)
    root = env.clone()
    bare = bare_frame(root)
    for action in BASE:
        root.step(action)
    kind, wanted, targets = SPECS[0]
    desc = descriptor(arr(root.frame()), bare, kind, dot(arr(root.frame())))
    print("base", len(BASE), desc[:3], covered(desc, kind, targets), flush=True)
    fixed = tuple([LEFT] * 7 + [UP] * 2 + [DOWN] * 2 + [RIGHT] * 7)
    test = root.clone()
    for action in fixed:
        test.step(action)
    test_desc = descriptor(
        arr(test.frame()), bare, kind, dot(arr(test.frame()))
    )
    print("fixed", len(fixed), test_desc[:3],
          covered(test_desc, kind, targets), flush=True)
    if test_desc[0] == wanted and len(covered(test_desc, kind, targets)) == len(targets):
        print("FIXED-SOLVED", "".join(NAME[a] for a in fixed), flush=True)
        return
    suffix = search(test, bare, kind, wanted, targets,
                    max_states=5000, max_depth=18)
    if suffix:
        for action in suffix:
            test.step(action)
        print("level", test.levels_completed,
              "total", len(BASE) + len(fixed) + len(suffix),
              flush=True)


A.run_program("re86", run)
