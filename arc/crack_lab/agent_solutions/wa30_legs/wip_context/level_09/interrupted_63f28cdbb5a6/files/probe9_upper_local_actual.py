"""Find the shortest way to carry the upper member of the local cargo pair."""

import numpy as np

import gkm_try

from perception import arr, bounded_replay_bfs
from probe9_actual_candidates import state
from probe9_two_staged_trace import DISMISS, TWO_STAGED
from probe9_verify import tile_map


def carrying(frame):
    return int(np.count_nonzero(arr(frame) == 0)) > 4


def upper_held(env, _path):
    grid = arr(env.frame())
    upper = set(int(value) for value in grid[28:32, 4:8].flat)
    lower = set(int(value) for value in grid[32:36, 4:8].flat)
    return carrying(grid) and 4 not in upper and 4 in lower


def both_moved_one_held(env, _path):
    grid = arr(env.frame())
    upper = set(int(value) for value in grid[28:32, 4:8].flat)
    lower = set(int(value) for value in grid[32:36, 4:8].flat)
    return carrying(grid) and 4 not in upper and 4 not in lower


def inspect(env):
    gkm_try.resumed_solve(env)
    base = env.clone()
    for action in TWO_STAGED + DISMISS:
        base.step(action)
    path = bounded_replay_bfs(
        base,
        upper_held,
        lambda node: node.actions,
        max_states=5000,
        max_depth=12,
    )
    print("UPPER_LOCAL_PATH", path, flush=True)
    if path is None:
        return
    child = base.clone()
    for action in path:
        child.step(action)
    print("UPPER_LOCAL_STATE", 44 + len(path), state(child), flush=True)
    print(*tile_map(child.frame()), sep="\n", flush=True)

    both_path = bounded_replay_bfs(
        base,
        both_moved_one_held,
        lambda node: node.actions,
        max_states=10000,
        max_depth=14,
    )
    print("BOTH_LOCAL_PATH", both_path, flush=True)
    if both_path is None:
        return
    child = base.clone()
    for action in both_path:
        child.step(action)
    print("BOTH_LOCAL_STATE", 44 + len(both_path), state(child), flush=True)
    print(*tile_map(child.frame()), sep="\n", flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
