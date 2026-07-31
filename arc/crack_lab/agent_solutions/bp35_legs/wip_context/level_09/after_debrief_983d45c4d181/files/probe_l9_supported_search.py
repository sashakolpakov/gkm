"""Run the existing bounded gravity-room planner from the staged four-skip frontier."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import gravity_room_search, run_actions
from probe_l9_control_row import controls
from probe_l9_route_deletions import enter_level_9
from probe_l9_skip4_landing_supports import staged
from probe_l9_skip4_switch_choices import report


def supported(root):
    child = staged(root)
    child.step(6, 21, 57)
    child.step(*controls(child)[0])
    return child


def probe(env):
    enter_level_9(env)
    root = supported(env)
    report("ROOT", root)
    path = gravity_room_search(root, max_states=500, max_depth=48)
    print("PATH", path, flush=True)
    if path:
        run_actions(root, path)
    report("RESULT", root)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
