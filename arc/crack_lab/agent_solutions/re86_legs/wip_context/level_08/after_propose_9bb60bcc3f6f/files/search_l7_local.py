import json
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import arr
from search_l7_segments import PREFIX, SEGMENTS, expand, step_many


CASES = (
    (0, 1, 5),
    (0, 2, 5),
    (1, 6, 9),
    (1, 7, 9),
    (2, 0, 5),
    (2, 5, 8),
    (2, 6, 11),
    (2, 8, 10),
    (2, 10, 12),
)


def shortest(root, target_key, max_depth, max_states=8000):
    queue = deque([(root.clone(), [])])
    seen = {arr(root.frame()).tobytes()}
    while queue and len(seen) < max_states:
        node, path = queue.popleft()
        if arr(node.frame()).tobytes() == target_key:
            return path, len(seen)
        if len(path) >= max_depth:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            key = arr(child.frame()).tobytes()
            if key not in seen:
                seen.add(key)
                queue.append((child, path + [action]))
    return None, len(seen)


def probe(env):
    with open("checkpoint.json") as handle:
        full = json.load(handle)["final_path"]
    step_many(env, full[:PREFIX])
    segment_roots = []
    for index, runs in enumerate(SEGMENTS):
        segment_roots.append(env.clone())
        step_many(env, expand(runs))
        if index < 2:
            env.step(5)

    for segment_index, run_start, run_end in CASES:
        runs = SEGMENTS[segment_index]
        before = expand(runs[:run_start])
        window = expand(runs[run_start:run_end])
        root = segment_roots[segment_index].clone()
        step_many(root, before)
        target = root.clone()
        step_many(target, window)
        result, seen = shortest(
            root, arr(target.frame()).tobytes(), len(window) - 1
        )
        print(
            "case",
            segment_index + 1,
            run_start,
            run_end,
            "known",
            len(window),
            "result",
            result,
            "seen",
            seen,
            flush=True,
        )


if __name__ == "__main__":
    A.run_program("re86", probe)
