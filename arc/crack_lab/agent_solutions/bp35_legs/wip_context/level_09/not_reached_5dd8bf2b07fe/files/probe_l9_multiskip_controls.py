"""Test each surviving switch at the two- and three-skip frontiers."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_multi_skip_bottom import root_for
from probe_l9_route_deletions import enter_level_9


def pieces(env):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            env.frame(), colors=(7, 8, 9, 11, 12, 14, 15), min_area=2
        )
        if blob.bbox[0] < 63
        and (blob.color in (7, 8, 9, 11, 12, 14) or blob.area == 21)
    )


def brief(env):
    return (
        bool(env.terminal()),
        int(env.levels_completed) + 1,
        tuple(controls(env)),
        compact(env)["grid9"],
        pieces(env),
    )


def run(root, count, switch_index, actions):
    child = root_for(root, count)
    switch = controls(child)[switch_index]
    name = (count, switch_index, switch)
    print(name, 0, brief(child), flush=True)
    child.step(*switch)
    print(name, 1, switch, brief(child), flush=True)
    for index, action in enumerate(actions, 2):
        child.step(action)
        print(name, index, action, brief(child), flush=True)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


def probe(env):
    enter_level_9(env)
    for count in (2, 3):
        for switch_index in range(3):
            run(env, count, switch_index, (3, 3, 4, 4, 4))


if __name__ == "__main__":
    arena.run_program("bp35", probe)
