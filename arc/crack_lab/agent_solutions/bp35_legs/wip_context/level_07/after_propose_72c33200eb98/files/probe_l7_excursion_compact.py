"""Fresh support-staging experiments across level 7's right-hand shaft."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena
import gkm_legs as campaign

from legs import COL_ANCHORS, ROW_ANCHORS, _cell_shape, click_action
from probe_l7_raw_search import SEED, avatar_position, target_path_distance


if campaign._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


PALETTE = {
    0: "v", 3: "#", 5: "D", 7: "T", 8: "g", 9: "A",
    10: ".", 11: "a", 12: "s", 14: "Y", 15: "h",
}
ENTRY = [(6, 3, 47), (4,), (4,), (3,)]


def cells(frame, colors):
    return tuple(
        (int(frame[y][x]), i, j, _cell_shape(frame, i, j)[1])
        for i, y in enumerate(ROW_ANCHORS)
        for j, x in enumerate(COL_ANCHORS)
        if int(frame[y][x]) in colors
    )


def lattice(frame):
    return "/".join(
        "".join(PALETTE.get(int(frame[y][x]), "?") for x in COL_ANCHORS)
        for y in ROW_ANCHORS
    )


def summary(node):
    if node.terminal():
        return ("dead", int(node.levels_completed))
    frame = node.frame()
    return (
        "alive", int(node.levels_completed),
        avatar_position(frame), target_path_distance(frame),
        cells(frame, (0, 7, 12, 15)), lattice(frame),
    )


def run(node, actions):
    for action in actions:
        node.step(*action)
        if node.terminal() or node.levels_completed > 6:
            break


def probe(env):
    with open("checkpoint.json") as stream:
        for action in json.load(stream)["final_path"]:
            env.step(action)
    run(env, [*SEED, (7,), (7,)])
    root = env.clone()
    print("ROOT", summary(root))
    top = root.clone()
    for index, action in enumerate(ENTRY, 1):
        top.step(*action)
        print("ENTRY", index, action, summary(top))
        if top.terminal():
            return

    support_actions = [
        click_action(i, j)
        for color, i, j, area in cells(top.frame(), (12,))
        if area <= 21
    ]
    print("SUPPORT_ACTIONS", support_actions)
    limit = int(os.environ.get("SUPPORT_LIMIT", str(len(support_actions))))
    for support in support_actions[:limit]:
        for releases in (0, 1, 2, 3, 4, 5):
            node = top.clone()
            run(node, [support, *([(7,)] * releases)])
            print("STAGE", support, releases, summary(node))
        returned = top.clone()
        route = [support, *([(7,)] * 5), (4,), (4,)]
        for index, action in enumerate(route, 1):
            returned.step(*action)
            print("DESCENT_STEP", support, index, action, summary(returned))
            if returned.terminal() or returned.levels_completed > 6:
                break
        print("DESCENT", support, summary(returned))


arena.run_program("bp35", probe)
