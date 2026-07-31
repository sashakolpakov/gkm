"""Measure exact top-island reach with the ring at its north terminal."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve
from probe_l6_cargo_top import CARGO_TOP_PATH
from probe_l6_right import (
    MAIN,
    SELECTOR,
    TOP,
    avatar_position,
    enter_right,
    movement_reach,
)


def apply(env, path):
    for action in path:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def exits(env):
    return tuple(
        blob.bbox
        for blob in perception.connected_components(
            env.frame(), colors=(11,), min_area=4
        )
        if blob.area == 4 and blob.size == (2, 2) and blob.bbox[1] < 40
    )


def metric(reached):
    positions = sorted(
        position for position in reached if position is not None
    )
    return (
        len(positions),
        min(row for row, _ in positions),
        max(row for row, _ in positions),
        min(col for _, col in positions),
        max(col for _, col in positions),
    )


def observe(env):
    solve.solve(env)
    base_level = env.levels_completed
    terminal = enter_right(env, 3)
    apply(terminal, CARGO_TOP_PATH)
    terminal.step(2)
    best = None
    for orientation in range(2):
        oriented = terminal.clone()
        if orientation:
            oriented.step(4)
            oriented.step(*MAIN)
            oriented.step(3)
        for bridge_phase in range(6):
            staged = oriented.clone()
            apply(staged, [TOP] * bridge_phase)
            staged.step(*MAIN)
            apply(staged, [SELECTOR] * 3)
            staged.step(*MAIN)
            reached, win = movement_reach(staged)
            current = metric(reached)
            if best is None or current > best:
                best = current
                print(
                    "TOP_REACH_PROGRESS", orientation, bridge_phase,
                    avatar_position(staged), current, "win", win,
                    "exits", exits(staged), flush=True,
                )
            if win is not None or staged.levels_completed > base_level:
                print("TOP_REACH_WIN", orientation, bridge_phase, win)
                return

            # Rotate B while safely off the patterned portal, then remeasure.
            off_portal = next(
                (
                    (position, path)
                    for position, path in reached.items()
                    if position not in (None, (4, 4))
                ),
                None,
            )
            if off_portal is None:
                continue
            _, path = off_portal
            rotated = staged.clone()
            apply(rotated, path)
            rotated.step(*MAIN)
            rotated_reach, rotated_win = movement_reach(rotated)
            rotated_metric = metric(rotated_reach)
            if rotated_metric > best:
                best = rotated_metric
                print(
                    "TOP_ROTATED_PROGRESS", orientation, bridge_phase,
                    avatar_position(rotated), rotated_metric,
                    "win", rotated_win, "exits", exits(rotated), flush=True,
                )
            if rotated_win is not None or rotated.levels_completed > base_level:
                print(
                    "TOP_ROTATED_WIN", orientation, bridge_phase,
                    path, rotated_win,
                )
                return
    print("TOP_REACH_DONE", best, flush=True)


arena.run_program("dc22", observe)
