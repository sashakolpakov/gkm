"""Observe both couriers from the pristine level-9 frame."""

import gkm_try

from probe9_prefix_shortcuts import reach_level_9
from probe9_verify import boxes, target_state


def inspect(env):
    reach_level_9(env)
    clone = env.clone()
    prior = None
    for turn in range(0, 71):
        target = target_state(clone.frame())
        current = (
            boxes(clone.frame(), 12),
            boxes(clone.frame(), 15),
            boxes(clone.frame(), 4),
            target["empty"],
            target["filled"],
        )
        if turn == 0 or current != prior:
            print("INITIAL_IDLE", turn, current, flush=True)
        prior = current
        if clone.terminal():
            break
        clone.step(5)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
