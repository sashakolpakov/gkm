"""Search for an intervention faster than level 8's final 15 courier turns."""

import gkm_try

from perception import bounded_bfs
from probe8_entry import ReachedLevel8, StopAtLevel8
from probe8_reverse_stage import compact
from probe8_tail_mutations import PREFIX, THIRD


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel8(env))
    except ReachedLevel8:
        pass
    start = env.clone()
    for action in PREFIX + THIRD:
        start.step(action)
    base_level = start.levels_completed
    print("L8_TAIL_BFS_START", compact(start, len(PREFIX + THIRD)), flush=True)
    path = bounded_bfs(
        start,
        lambda node, _path: node.levels_completed > base_level,
        max_states=5000,
        max_depth=14,
    )
    print("L8_TAIL_BFS_PATH", path, flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
