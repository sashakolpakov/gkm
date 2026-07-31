"""Run the existing local-hazard climb search from the final flipped maze."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import local_hazard_climb_search, run_actions
from probe_l9_control_row import compact
from probe_l9_route_deletions import enter_level_9
from probe_l9_super_horizontal import start


def probe(env):
    enter_level_9(env)
    root = start(env, 3, 3, 0)
    path = local_hazard_climb_search(root, max_expansions=300)
    print("PATH", len(path), path, flush=True)
    if path:
        run_actions(root, path)
    print(
        "RESULT",
        "levels",
        int(root.levels_completed),
        "terminal",
        bool(root.terminal()),
        "state",
        compact(root),
        flush=True,
    )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
