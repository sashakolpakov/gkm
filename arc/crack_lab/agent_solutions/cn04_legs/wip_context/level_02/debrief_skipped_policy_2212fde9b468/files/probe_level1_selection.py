"""Test whether level 1 is symmetric under selecting the colored body."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import perception
from probe_level2_collect import avatar_key


def probe(env):
    ys, xs = np.where(perception.arr(env.frame()) == 14)
    env.step(6, int(xs[0]), int(ys[0]))
    print("selected", perception.color_counts(env.frame()),
          [(b.color, b.area, b.bbox)
           for b in perception.connected_components(
               env.frame(), colors=(0, 15), min_area=4
           )])
    path = perception.bounded_bfs(
        env,
        perception.level_goal(0),
        actions=(1, 2, 3, 4, 5),
        key_fn=avatar_key,
        max_states=8000,
        max_depth=80,
    )
    print("finish", path)


arena.run_program("cn04", probe)
