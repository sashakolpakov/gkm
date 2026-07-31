"""Verify the first ring/global-phase configuration with expanded reach."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve
from probe_l6_right import avatar_position, enter_right, movement_reach


A = (6, 56, 8)
B = (6, 50, 26)
DL = (6, 46, 36)
EXPAND = [3, DL, DL, 1, A, 2, 4, B, 2, 2, 2, 2, A]
PLAIN = [B, A, 2, 2, 2, 2, A]
CONTROLS = {
    "A": A,
    "B": B,
    "S": (6, 50, 46),
    "DU": (6, 50, 34),
    "DD": (6, 50, 40),
    "DL": DL,
    "DR": (6, 54, 36),
}


def apply(env, path):
    for action in path:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def rings(env):
    return tuple(
        (blob.bbox, blob.area)
        for blob in perception.connected_components(
            env.frame(), colors=(8,), min_area=20
        )
        if blob.bbox[1] < 40
    )


def exits(env):
    return tuple(
        blob.bbox
        for blob in perception.connected_components(
            env.frame(), colors=(11,), min_area=4
        )
        if blob.area == 4 and blob.size == (2, 2) and blob.bbox[1] < 40
    )


def observe(env):
    solve.solve(env)
    base_level = env.levels_completed
    controller = enter_right(env, 3)
    plain = controller.clone()
    apply(plain, PLAIN)
    plain_reach, plain_win = movement_reach(plain)

    expanded = controller.clone()
    apply(expanded, EXPAND)
    expanded_reach, expanded_win = movement_reach(expanded)
    plain_positions = {
        position for position in plain_reach if position is not None
    }
    expanded_positions = {
        position for position in expanded_reach if position is not None
    }
    novel = sorted(expanded_positions - plain_positions)
    print(
        "EXPANDED79_ROOT", avatar_position(expanded), rings(expanded),
        "level", expanded.levels_completed,
        "plain", len(plain_positions), "expanded", len(expanded_positions),
        "plain_win", plain_win, "expanded_win", expanded_win,
        "exits", exits(expanded), flush=True,
    )
    print(
        "EXPANDED79_NOVEL",
        [(position, expanded_reach[position]) for position in novel],
        flush=True,
    )
    for position in novel:
        walked = expanded.clone()
        apply(walked, expanded_reach[position])
        effects = []
        for label, control in CONTROLS.items():
            child = walked.clone()
            before = perception.arr(child.frame())[:63].copy()
            child.step(*control)
            changed = perception.frame_delta(before, child.frame()[:63])["count"]
            if changed or child.levels_completed > base_level:
                effects.append(
                    (
                        label, changed, avatar_position(child),
                        child.levels_completed, exits(child),
                    )
                )
        print("EXPANDED79_SITE", position, "effects", effects, flush=True)


arena.run_program("dc22", observe)
