"""Search each selectable figure's independent pose space for reward."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
from players import play_level_1


POINTS = (("A", (18, 21)), ("B", (45, 15)), ("C", (18, 39)), ("D", (51, 51)))


def probe(env):
    play_level_1(env)
    for name, point in POINTS:
        node = env.clone()
        node.step(6, *point)
        path = perception.bounded_bfs(
            node,
            perception.level_goal(1),
            actions=(1, 2, 3, 4, 5),
            key_fn=lambda child: perception.arr(child.frame()).tobytes(),
            max_states=5000,
            max_depth=80,
        )
        print("agent", name, "path", path)
        if path is not None:
            print("solved", name, [6, *point], path)
            return


arena.run_program("cn04", probe)
