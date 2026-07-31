import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _movable_bridge_board
from perception import color_counts, frame_delta


def board(frame):
    return tuple(tuple(sorted(part)) for part in _movable_bridge_board(frame))


def probe(env):
    with open("checkpoint.json") as stream:
        actions = json.load(stream)["final_path"]
    for action in actions[:-1]:
        env.step(*action) if isinstance(action, list) else env.step(action)
    before = env.frame()
    print("BEFORE", env.levels_completed, actions[-1], board(before),
          color_counts(before))
    action = actions[-1]
    env.step(*action) if isinstance(action, list) else env.step(action)
    after = env.frame()
    delta = frame_delta(before, after)
    print("AFTER", env.levels_completed, board(after), color_counts(after),
          (delta["count"], delta["bbox"]))


levels, path, error = arena.run_program("lf52", probe)
print("RESULT", levels, len(path), error)
