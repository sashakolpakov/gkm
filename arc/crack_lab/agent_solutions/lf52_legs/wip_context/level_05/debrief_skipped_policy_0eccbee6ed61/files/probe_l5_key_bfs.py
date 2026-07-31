import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import arr, bounded_bfs, level_goal


PREFIX = (
    (6, 13, 25), (6, 25, 25),
    3, 3, 2, 2, 2,
    (6, 25, 25), (6, 37, 25),
    4, 4, 4, 2, 2, 2,
    (6, 43, 25), (6, 31, 25),
)


def act(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    assert checkpoint["game"] == "lf52" and checkpoint["validated"]
    for action in checkpoint["final_path"]:
        act(env, tuple(action) if isinstance(action, list) else action)
    for action in PREFIX:
        act(env, action)
    solution = bounded_bfs(
        env,
        level_goal(env.levels_completed),
        actions=(1, 2, 3, 4),
        key_fn=lambda node: arr(node.frame())[1:, :].tobytes(),
        max_states=900,
        max_depth=36,
    )
    print("SOLUTION", solution)


levels, path, err = A.run_program("lf52", probe)
print("END", levels, len(path), err)
