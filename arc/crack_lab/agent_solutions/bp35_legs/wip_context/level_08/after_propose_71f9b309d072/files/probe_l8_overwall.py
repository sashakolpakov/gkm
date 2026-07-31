"""Cross above the wall and test the nearby target/support affordances."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import _cell_shape, avatar_column, band_shift, click_action
from perception import arr, connected_components
from probe_l8_climb4 import RELEASE, ROOT_ROUTE
from probe_l8_stage1 import lattice, target


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


HANDOFFS = []
for column in range(2, 8):
    HANDOFFS.extend([click_action(6, column), (4,)])

OVERWALL = [*ROOT_ROUTE, RELEASE, RELEASE, *HANDOFFS]


def target_click(frame):
    blobs = [
        blob
        for blob in connected_components(frame, colors=(7,), min_area=2)
        if blob.bbox[0] < 63
    ]
    if not blobs:
        return None
    blob = blobs[0]
    return 6, round(blob.centroid[1]), round(blob.centroid[0])


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
                "level",
                node.levels_completed,
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

    for count in range(1, len(HANDOFFS) + 1):
        node = env.clone()
        gain = run(node, [*ROOT_ROUTE, RELEASE, RELEASE, *HANDOFFS[:count]])
        print(
            "CROSS",
            count,
            not node.terminal(),
            node.levels_completed,
            gain,
            None if node.terminal() else avatar_column(node.frame()),
            None if node.terminal() else target(node.frame()),
        )

    root = env.clone()
    gain = run(root, OVERWALL)
    print("ROOT", gain, root.levels_completed, root.terminal(), lattice(root.frame()))
    print(
        "OTHER",
        [
            (blob.color, blob.bbox, blob.area)
            for blob in connected_components(
                root.frame(), colors=(0, 8, 14), min_area=1
            )
            if blob.bbox[0] < 63
        ],
    )
    click = target_click(root.frame())
    candidates = [(3,), (4,), (7,), click_action(7, 6)]
    if click is not None:
        candidates.append(click)
    for action in candidates:
        node = root.clone()
        before = arr(node.frame()).copy()
        node.step(*action)
        print(
            "TRY",
            action,
            node.levels_completed,
            node.terminal(),
            0 if node.terminal() else band_shift(before, node.frame()),
            None if node.terminal() else avatar_column(node.frame()),
            None if node.terminal() else target(node.frame()),
            "" if node.terminal() else lattice(node.frame()),
        )

    if click is not None:
        print("DIRECT_TARGET", click)
        run(env, [*OVERWALL, click], trace=True)


if __name__ == "__main__":
    levels, path, err = A.run_program("bp35", probe)
    print("RESULT", levels, len(path), err)
