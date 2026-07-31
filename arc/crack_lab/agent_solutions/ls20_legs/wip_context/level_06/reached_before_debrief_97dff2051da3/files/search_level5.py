"""Bounded reward search from the pristine level-5 entry."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import bounded_bfs, level_goal
from probe_level5 import reach_level_5


def inspect(env):
    reach_level_5(env)
    path = bounded_bfs(
        env,
        level_goal(4),
        actions=env.actions,
        max_states=5000,
        max_depth=70,
    )
    print("search_path", path)


if __name__ == "__main__":
    levels, path, error = A.run_program("ls20", inspect)
    print("probe_result", levels, len(path), error)
