"""Verify the exit from the first target-visible landing."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import _cell_shape, avatar_column, band_shift, click_action
from perception import arr
from probe_l8_stage1 import lattice, target
from probe_l8_stage4 import PREFIX


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


LEFT_EXIT = [click_action(6, 5), (3,), (3,)]
RIGHT_EXIT = [click_action(6, 7), (4,)]


def run(node, route, trace=False):
    height = 0
    for index, action in enumerate(route, 1):
        if node.terminal():
            break
        before = arr(node.frame()).copy()
        shape = None
        if len(action) == 3 and action[1] >= 15:
            shape = _cell_shape(
                before, (action[2] - 3) // 6, (action[1] - 15) // 6
            )
        node.step(*action)
        shift = 0 if node.terminal() else band_shift(before, node.frame())
        height += shift
        if trace:
            print(
                "STEP",
                index,
                action,
                "shape",
                shape,
                "alive",
                not node.terminal(),
                "height",
                height,
                "col",
                None if node.terminal() else avatar_column(node.frame()),
                "target",
                None if node.terminal() else target(node.frame()),
                "grid",
                "" if node.terminal() else lattice(node.frame()),
            )
    return height


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    for label, suffix in (("left", LEFT_EXIT), ("right", RIGHT_EXIT)):
        for count in range(1, len(suffix) + 1):
            node = env.clone()
            gain = run(node, [*PREFIX, *suffix[:count]])
            print(
                "CASE",
                label,
                count,
                "alive",
                not node.terminal(),
                "height",
                gain,
                "col",
                None if node.terminal() else avatar_column(node.frame()),
                "target",
                None if node.terminal() else target(node.frame()),
                "grid",
                "" if node.terminal() else lattice(node.frame()),
            )

    print("DIRECT")
    run(env, [*PREFIX, *LEFT_EXIT], trace=True)


if __name__ == "__main__":
    levels, path, err = A.run_program("bp35", probe)
    print("RESULT", levels, len(path), err)
