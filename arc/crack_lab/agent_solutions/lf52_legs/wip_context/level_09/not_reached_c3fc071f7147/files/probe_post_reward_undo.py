"""Check action-7 history behavior immediately after a level reward."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import solve_peg_solitaire
from perception import arr, frame_delta, safe_step


def delta(before, after):
    left = arr(before).copy()
    right = arr(after).copy()
    right[0] = left[0]
    out = frame_delta(left, right)
    return out["count"], out["bbox"]


def probe(env):
    node = env.clone()
    solve_peg_solitaire(node)
    rewarded = node.frame()
    level = int(node.levels_completed)
    observations = []
    for count in range(1, 5):
        before = node.frame()
        safe_step(node, 7)
        observations.append((count, int(node.levels_completed),
                             delta(before, node.frame()),
                             delta(rewarded, node.frame())))
    print("post_reward_undo", level, observations, flush=True)


arena.run_program("lf52", probe)
