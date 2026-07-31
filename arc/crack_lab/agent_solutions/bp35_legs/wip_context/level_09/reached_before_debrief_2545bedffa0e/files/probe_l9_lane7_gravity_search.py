"""Run the existing gravity-room planner from the verified early-flip root."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import gravity_room_search, run_actions
from probe_l9_control_row import compact
from probe_l9_lane7_early_flip_local import root_lane7
from probe_l9_route_deletions import enter_level_9


def probe(env):
    enter_level_9(env)
    root = root_lane7(env)
    path = gravity_room_search(
        root,
        max_states=320,
        max_depth=27,
        debug=False,
    )
    print("PATH", path, flush=True)
    if path:
        child = root.clone()
        run_actions(child, path)
        print(
            "VERIFY",
            "level",
            int(child.levels_completed) + 1,
            "terminal",
            bool(child.terminal()),
            "state",
            compact(child),
            flush=True,
        )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
