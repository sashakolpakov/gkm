import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import _cell_shape, band_shift, click_action
from perception import arr, connected_components
from probe_level7_reward_recovery import (
    PREFIX, SUFFIX, avatar_cell, controls,
)


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


def compact_supports(frame):
    avatar = avatar_cell(frame)
    if avatar is None:
        return []
    ai, aj = avatar
    return [
        (i, j)
        for i in range(max(0, ai - 2), min(10, ai + 3))
        for j in range(max(0, aj - 2), min(8, aj + 3))
        if _cell_shape(frame, i, j)[0] in (12, 14)
        and _cell_shape(frame, i, j)[1] < 21
    ]


def signed_shift(before, after):
    a, b = arr(before)[:63], arr(after)[:63]
    scored = []
    for bands in range(-10, 11):
        offset = 6 * bands
        if offset >= 0:
            left, right = a[:63 - offset], b[offset:]
        else:
            left, right = a[-offset:], b[:63 + offset]
        rows = sum(bool((left[index] == right[index]).all())
                   for index in range(len(left)))
        pixels = int((left == right).sum())
        scored.append((rows, pixels, -abs(bands), bands))
    return max(scored)[-1]


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    route = [
        *PREFIX,
        click_action(5, 2),
        *SUFFIX,
        (3,), (6, 3, 9), (4,), (6, 3, 39),
        (3,), (3,), (3,),
    ]
    node = env.clone()
    height = 0
    net_height = 0
    for index, action in enumerate(route, 1):
        before = arr(node.frame()).copy()
        clicked_shape = None
        if len(action) == 3 and action[1] != 3:
            clicked_shape = [
                (i, j, _cell_shape(before, i, j))
                for i in range(10)
                for j in range(8)
                if click_action(i, j) == action
            ]
        node.step(*action)
        gain = 0 if node.terminal() else band_shift(before, node.frame())
        height += gain
        net_height += 0 if node.terminal() else signed_shift(before, node.frame())
        if gain or clicked_shape:
            print(
                "STEP", index, action, "clicked", clicked_shape,
                "gain", gain, "height", height, "net", net_height,
                "avatar", None if node.terminal()
                else avatar_cell(node.frame()),
                "controls", [] if node.terminal()
                else controls(node.frame()),
                "compact", [] if node.terminal()
                else compact_supports(node.frame()),
            )
        if node.terminal():
            break
    print(
        "FINAL", len(route), height, net_height,
        node.levels_completed, node.terminal(),
        None if node.terminal() else avatar_cell(node.frame()),
        [] if node.terminal() else controls(node.frame()),
        [] if node.terminal() else [
            (blob.bbox, blob.area)
            for blob in connected_components(
                node.frame(), colors=(7,), min_area=2
            )
            if blob.bbox[0] < 63
        ],
    )


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
