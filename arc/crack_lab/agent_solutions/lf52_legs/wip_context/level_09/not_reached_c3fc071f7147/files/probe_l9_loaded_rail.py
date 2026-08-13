"""Check loaded-carrier rail controls beyond the visible right board."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, connected_components, safe_step


FIRST_RELAY = (
    ((42, 18), (42, 30)), ((48, 24), (36, 24)),
    ((42, 24), (30, 24)), ((36, 24), (24, 24)),
    ((30, 24), (18, 24)), ((18, 24), (18, 36)),
    ((18, 36), (30, 36)), ((24, 36), (36, 36)),
    ((30, 36), (42, 36)), ((36, 36), (48, 36)),
    ((48, 42), (48, 30)), ((48, 30), (36, 30)),
    ((48, 36), (36, 36)), ((36, 30), (36, 42)),
)


def key(env):
    return arr(env.frame())[1:, :].tobytes()


def compact(frame):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(9, 11, 12, 14, 15))
        if blob.color != 9 or blob.area >= 12
    )


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    for source, destination in FIRST_RELAY:
        safe_step(env, (6, source[1] + 1, source[0] + 1))
        safe_step(env, (6, destination[1] + 1, destination[0] + 1))

    node = env.clone()
    observations = []
    for offset in range(41):
        enabled = []
        for action in (1, 2):
            before = key(node)
            safe_step(node, action)
            if key(node) != before:
                enabled.append((action, compact(node.frame())))
                break
        before = key(node)
        safe_step(node, 4)
        observations.append((offset, enabled, key(node) != before))
        if enabled:
            break
    print("loaded_offsets", observations, flush=True)

    patterns = (
        (4,) * 9 + (1, 1, 3, 3, 1, 1),
        (4,) * 9 + (2, 2, 3, 3, 2, 2),
        (4,) * 15 + (1, 1, 3, 3, 1, 1),
        (4,) * 15 + (2, 2, 3, 3, 2, 2),
    )
    for pattern in patterns:
        child = env.clone()
        for action in pattern:
            safe_step(child, action)
        print("pattern", pattern, int(child.levels_completed),
              compact(child.frame()), flush=True)


arena.run_program("lf52", probe)
