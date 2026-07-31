"""Reward-preserving minimization of the verified level-8 clone path."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena


SELECT_LEFT = (6, 14, 58)
SELECT_TOP = (6, 37, 58)


def apply(env, action):
    if isinstance(action, tuple):
        env.step(*action)
    else:
        env.step(action)


def winning_prefix(root, path):
    node = root.clone()
    base_level = node.levels_completed
    for index, action in enumerate(path, 1):
        apply(node, action)
        if node.levels_completed > base_level:
            return index
    return None


def minimize(root, path):
    won_at = winning_prefix(root, path)
    if won_at is None:
        raise RuntimeError("seed path does not win")
    path = path[:won_at]
    print("SEED_WIN", len(path), flush=True)

    pass_number = 0
    while True:
        pass_number += 1
        changed = False
        for size in (32, 16, 8, 4, 2, 1):
            index = 0
            while index + size <= len(path):
                candidate = path[:index] + path[index + size :]
                won_at = winning_prefix(root, candidate)
                if won_at is not None:
                    path = candidate[:won_at]
                    changed = True
                    print(
                        "DROP",
                        pass_number,
                        size,
                        index,
                        "LEN",
                        len(path),
                        flush=True,
                    )
                else:
                    index += size
        if not changed:
            break
    return path


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        checkpoint = json.load(checkpoint_file)
    for action in checkpoint["final_path"]:
        env.step(action)

    shelf_prefix = (
        [4] * 4
        + [3] * 4
        + [SELECT_TOP, 4, 2, 3, 1, 3, 2, 1, SELECT_LEFT]
        + [4] * 3
        + [SELECT_TOP, 4]
        + [2] * 6
        + [1] * 4
        + [4]
    )
    collision_release_suffix = (
        [SELECT_LEFT]
        + [3] * 3
        + [2]
        + [4] * 4
        + [3] * 4
        + [1]
        + [4] * 5
        + [3, 1]
        + [3] * 3
        + [SELECT_TOP]
        + [2] * 5
        + [1] * 6
        + [3] * 2
        + [2, 1]
        + [4] * 2
        + [SELECT_LEFT, 2]
        + [4] * 6
        + [3] * 2
        + [SELECT_TOP]
        + [2] * 2
        + [SELECT_LEFT, 1]
        + [3] * 3
        + [SELECT_TOP]
        + [1] * 2
        + [3] * 2
        + [2, 1]
        + [4] * 2
        + [2] * 6
        + [1] * 5
        + [SELECT_LEFT]
        + [4] * 6
        + [3] * 6
        + [2] * 2
        + [4] * 6
        + [3] * 5
    )
    path = minimize(env, shelf_prefix + collision_release_suffix)
    print("MINIMIZED", len(path), json.dumps(path), flush=True)


levels, path, err = arena.run_program("sk48", probe)
print("MINIMIZE_RESULT", levels, len(path), err)
