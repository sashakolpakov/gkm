"""Test hidden-agent positioning before the second level-9 region is revealed."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import connected_components, safe_step


FIRST_RELAY = (
    ((42, 18), (42, 30)), ((48, 24), (36, 24)),
    ((42, 24), (30, 24)), ((36, 24), (24, 24)),
    ((30, 24), (18, 24)), ((18, 24), (18, 36)),
    ((18, 36), (30, 36)), ((24, 36), (36, 36)),
    ((30, 36), (42, 36)), ((36, 36), (48, 36)),
    ((48, 42), (48, 30)), ((48, 30), (36, 30)),
    ((48, 36), (36, 36)), ((36, 30), (36, 42)),
)


def compact(frame):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(7, 9, 11, 12, 14, 15))
        if blob.color not in (9, 12) or blob.area >= 12
    )


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    prefixes = [()]
    prefixes += [(action,) * count
                 for action in (1, 2, 3)
                 for count in range(1, 13)]
    prefixes += [
        (1, 2), (2, 1), (1, 1, 2, 2), (2, 2, 1, 1),
        (1, 2, 1, 2), (2, 1, 2, 1),
    ]
    seen = {}
    for prefix in prefixes:
        clone = env.clone()
        for action in prefix:
            safe_step(clone, action)
        for source, destination in FIRST_RELAY:
            safe_step(clone, (6, source[1] + 1, source[0] + 1))
            safe_step(clone, (6, destination[1] + 1, destination[0] + 1))
        state = compact(clone.frame())
        seen.setdefault(state, prefix)
    print("prereveal_states", len(seen), [
        (prefix, state) for state, prefix in seen.items()
    ])


arena.run_program("lf52", probe)
