"""Test true consecutive peg captures with one selection or far destination."""

import gkm_try
from legs import _peg_board
from perception import arr


def state(node):
    return tuple(sorted(_peg_board(node.frame())[1])), arr(node.frame())[1:, :].tobytes()


def probe(env):
    root = env.clone()
    baseline = root.clone()
    for action in ((6, 18, 19), (6, 30, 19), (6, 30, 19), (6, 42, 19)):
        baseline.step(*action)
    far = root.clone()
    for action in ((6, 18, 19), (6, 42, 19)): far.step(*action)
    chained = root.clone()
    for action in ((6, 18, 19), (6, 30, 19), (6, 42, 19)): chained.step(*action)
    keyed = root.clone()
    keyed.step(6, 18, 19); keyed.step(4)
    keyed_twice = root.clone()
    keyed_twice.step(6, 18, 19); keyed_twice.step(4); keyed_twice.step(4)
    print(
        "MULTIHOP", state(root)[0], state(baseline)[0], state(far)[0], state(chained)[0],
        state(far)[1] == state(baseline)[1], state(chained)[1] == state(baseline)[1],
        state(keyed)[0], state(keyed_twice)[0],
    )


gkm_try.A.run_program("lf52", probe)
