"""Test whether coordinate moves may span multiple aligned bridge hops."""

import json

import gkm_try
from legs import _bridge_carrier_state
from perception import arr


def summary(node):
    state = _bridge_carrier_state(node.frame())
    return tuple(sorted(state[1])), tuple(sorted(state[2])), tuple(sorted(state[3]))


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        path = json.load(checkpoint_file)["final_path"]
    for action in path[:476]:
        env.step(action)
    # Align the carriers and stage the three bridge moves preceding the peg relay.
    for action in path[476:489]:
        env.step(action)
    root = env.clone()
    print("LONG_ENTRY", summary(root))
    candidates = (
        ((6, 13, 25), (6, 37, 25)),
        ((6, 13, 25), (6, 49, 25)),
        ((6, 13, 25), (6, 49, 37)),
    )
    for actions in candidates:
        node = root.clone()
        before = arr(node.frame())[1:, :].copy()
        for action in actions:
            node.step(*action)
        print(
            "LONG_TEST", actions,
            int((before != arr(node.frame())[1:, :]).sum()),
            summary(node), node.levels_completed,
        )
    chained = root.clone()
    for action in ((6, 13, 25), (6, 25, 25), (6, 37, 25)):
        chained.step(*action)
    print("LONG_CHAINED_DEST", summary(chained), chained.levels_completed)


gkm_try.A.run_program("lf52", probe)
