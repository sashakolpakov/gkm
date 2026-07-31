"""Relay through the exact peg handles reached by each bridge pose."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
from players import play_level_1
from probe_level2_collect import peg_centers


A_VARIANTS = (
    ((0, 5), [4] * 8 + [5, 5]),
    ((2, 4), [4] * 14),
    ((2, 4), [2] + [4] * 14),
    ((2, 4), [2] * 2 + [4] * 16),
    ((2, 4), [4] * 11 + [5, 5] + [4] * 3),
    ((2, 4), [2] * 3 + [4] * 14),
)
B_ROUTE = [2] * 7 + [3] * 10 + [5, 5]
C_VARIANTS = (
    [1] + [4] * 9,
    [1] + [4] * 8 + [5],
)


def apply(node, path, route):
    for action in route:
        node.step(action)
        path.append(action)
        if node.levels_completed > 1:
            return True
    return False


def click_peg(node, path, pegs, index):
    r, c = pegs[index]
    action = (6, c, r)
    node.step(*action)
    path.append(action)
    return node.levels_completed > 1


def probe(env):
    play_level_1(env)
    pegs = peg_centers(env.frame())
    root = env.clone()
    trials = 0
    for a_pair, a_route in A_VARIANTS:
        for b_handle in a_pair:
            for c_handle in (6, 8):
                for c_route in C_VARIANTS:
                    for d_handle in (None, 9, 10):
                        for d_turns in range(4):
                            trials += 1
                            node = root.clone()
                            path = []
                            if apply(node, path, a_route):
                                print("solved", path)
                                return
                            if click_peg(node, path, pegs, b_handle):
                                print("solved", path)
                                return
                            if apply(node, path, B_ROUTE):
                                print("solved", path)
                                return
                            if click_peg(node, path, pegs, c_handle):
                                print("solved", path)
                                return
                            if apply(node, path, c_route):
                                print("solved", path)
                                return
                            if d_handle is not None:
                                if click_peg(node, path, pegs, d_handle):
                                    print("solved", path)
                                    return
                                if apply(node, path, [5] * d_turns):
                                    print("solved", path)
                                    return
    print("unsolved", "trials", trials)


arena.run_program("cn04", probe)
