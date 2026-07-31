"""Map the four selector destinations from the occupied remote hub."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve


TOP = (6, 54, 6)
MAIN = (6, 50, 26)
SELECTOR = (6, 50, 46)
TO_BRIDGE = [3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 2, 3]
TO_REMOTE_PAD = [4, 1, 4, 1, 4, 4, 1, 1, 1]


def avatar_position(env):
    for blob in perception.connected_components(
            env.frame(), colors=(14,), min_area=2):
        if blob.bbox[1] < 32:
            return blob.top_left
    return None


def avatar_blobs(env):
    return [
        (blob.bbox, blob.area)
        for blob in perception.connected_components(
            env.frame(), colors=(14,), min_area=1
        )
    ]


def pad(env):
    return tuple(
        int(value)
        for value in perception.arr(env.frame())[48:50, 18:20].flat
    )


def enter_glyph(env):
    node = env.clone()
    for _ in range(4):
        node.step(*TOP)
    for action in TO_BRIDGE:
        node.step(action)
    node.step(*TOP)
    node.step(1)
    node.step(3)
    node.step(*TOP)
    for _ in range(6):
        node.step(1)
    node.step(*MAIN)
    for _ in range(13):
        node.step(1)
    return node


def return_to_hub(node):
    for _ in range(13):
        node.step(2)
    for _ in range(6):
        node.step(2)
    node.step(*TOP)
    node.step(*TOP)
    node.step(4)
    node.step(*TOP)
    node.step(2)
    node.step(*TOP)
    for action in TO_REMOTE_PAD:
        node.step(action)


def observe(env):
    solve.solve(env)
    base_level = env.levels_completed
    glyph = enter_glyph(env)
    for selector_steps in range(4):
        branch = glyph.clone()
        for _ in range(selector_steps):
            branch.step(*SELECTOR)
        selected = pad(branch)
        return_to_hub(branch)
        hub = avatar_position(branch)
        branch.step(*MAIN)
        print(
            "DEST", selector_steps, selected, hub,
            avatar_position(branch), branch.levels_completed,
            branch.levels_completed - base_level, branch.terminal(),
            avatar_blobs(branch),
        )


arena.run_program("dc22", observe)
