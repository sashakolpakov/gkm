"""Verify the dense distinct-spoke target against the real level reward."""
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from plan_l6 import BOARDS, bad_columns, find_distinct_plan, read_board, swap
from probe_l6 import CONTROLS
from solve import solve


BOARD_CONTROLS = (
    (CONTROLS[0], CONTROLS[1]),
    (CONTROLS[4], CONTROLS[6]),
    (CONTROLS[2], CONTROLS[3]),
)
PORT_ANGLES = tuple(spec[1] for spec in BOARDS)
SWAP_CONTROL = CONTROLS[5]


def compile_clicks(states, plans):
    rounds = max(map(len, plans))
    padded = []
    for state, plan in zip(states, plans):
        final = state
        for index in plan:
            final = swap(final, index)
        dummy = next(i for i in range(24) if final[i] == final[24])
        padded.append(plan + (dummy,) * (rounds - len(plan)))

    angular_offsets = [0, 0, 0]
    radial_offsets = [0, 0, 0]
    clicks = []
    for turn in range(rounds):
        for board in range(3):
            index = padded[board][turn]
            radius, angle = divmod(index, 8)
            da = (angle - PORT_ANGLES[board] -
                  angular_offsets[board]) % 8
            dr = (2 - radius - radial_offsets[board]) % 3
            clicks.extend([BOARD_CONTROLS[board][0]] * da)
            clicks.extend([BOARD_CONTROLS[board][1]] * dr)
            angular_offsets[board] = (
                angular_offsets[board] + da) % 8
            radial_offsets[board] = (
                radial_offsets[board] + dr) % 3
        clicks.append(SWAP_CONTROL)
    return clicks


def run(env):
    solve(env)
    base_level = env.levels_completed
    frame = np.asarray(env.frame())
    states = tuple(read_board(frame, spec) for spec in BOARDS)
    plans = tuple(find_distinct_plan(state)[0] for state in states)
    clicks = compile_clicks(states, plans)
    print("plans", plans, "clicks", len(clicks))
    clone = env.clone()
    swaps = 0
    for step, click in enumerate(clicks, 1):
        clone.step(6, *click)
        if click == SWAP_CONTROL:
            swaps += 1
            if clone.levels_completed == base_level:
                current = tuple(read_board(np.asarray(clone.frame()), spec)
                                for spec in BOARDS)
                print("swap", swaps, "bad",
                      tuple(bad_columns(state) for state in current))
        if clone.levels_completed > base_level or clone.terminal():
            print("reward", clone.levels_completed, "at", step, click)
            print("PATH", clicks[:step])
            return
    current = tuple(read_board(np.asarray(clone.frame()), spec)
                    for spec in BOARDS)
    print("no_reward", clone.levels_completed,
          "bad", tuple(bad_columns(state) for state in current))
    print("PATH", clicks)


if __name__ == "__main__":
    A.run_program("lp85", run)
