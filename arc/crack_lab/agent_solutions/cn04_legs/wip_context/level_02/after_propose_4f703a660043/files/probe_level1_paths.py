"""Falsify path dependence in the known level-1 rewarded placement."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception


def probe(env):
    paths = {
        "known": [2] * 7 + [4] * 4 + [5] * 3,
        "right_first": [4] * 4 + [2] * 7 + [5] * 3,
        "detour": [1, 2] + [2] * 7 + [4] * 4 + [5] * 3,
        "rotate_early": [5] * 4 + [2] * 7 + [4] * 4 + [5] * 3,
    }
    for name, path in paths.items():
        node = perception.replay(env, path)
        print(name, node.levels_completed, len(path))


arena.run_program("cn04", probe)
