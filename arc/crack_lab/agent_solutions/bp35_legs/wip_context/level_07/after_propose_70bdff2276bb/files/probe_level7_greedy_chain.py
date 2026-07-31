import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import _cell_shape, click_action
from perception import arr
from probe_level7_coordinate_decode import advance
from probe_level7_greedy2 import PREFIX, MACRO1
from probe_level7_reward_recovery import avatar_cell, controls, lattice


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


def support_actions(frame):
    avatar = avatar_cell(frame)
    if avatar is None:
        return []
    ai, aj = avatar
    return [
        click_action(i, j)
        for i in range(max(0, ai - 2), min(10, ai + 3))
        for j in range(max(0, aj - 1), min(8, aj + 2))
        if _cell_shape(frame, i, j)[0] in (12, 14)
        and _cell_shape(frame, i, j)[1] < 21
    ]


def successors(root):
    outcomes = {}
    for support in [None, *support_actions(root.frame())]:
      supported = root.clone()
      support_path = [] if support is None else [support]
      support_gain = advance(supported, support_path)
      if supported.terminal():
        continue
      for pre in ([], [(3,)], [(4,)]):
        staged = supported.clone()
        pre_gain = support_gain + advance(staged, pre)
        if staged.levels_completed > 6:
            return [], [*support_path, *pre]
        if staged.terminal():
            continue
        for y1 in controls(staged.frame()):
            flipped = staged.clone()
            gain1 = pre_gain + advance(flipped, [(6, 3, y1)])
            path1 = [*support_path, *pre, (6, 3, y1)]
            if flipped.levels_completed > 6:
                return [], path1
            if flipped.terminal():
                continue
            for cross in ((3,), (4,)):
                middle = flipped.clone()
                gain2 = gain1 + advance(middle, [cross])
                path2 = [*path1, cross]
                if middle.levels_completed > 6:
                    return [], path2
                if middle.terminal():
                    continue
                for y2 in controls(middle.frame()):
                    child = middle.clone()
                    gain3 = gain2 + advance(child, [(6, 3, y2)])
                    path = [*path2, (6, 3, y2)]
                    if child.levels_completed > 6:
                        return [], path
                    if child.terminal():
                        continue
                    frame = arr(child.frame())
                    key = frame[:63].tobytes()
                    item = (
                        gain3, len(controls(frame)),
                        min(avatar_cell(frame)[1], 7 - avatar_cell(frame)[1]),
                        path, child,
                    )
                    previous = outcomes.get(key)
                    if previous is None or item[:3] > previous[:3]:
                        outcomes[key] = item
    return list(outcomes.values()), None


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    node = env.clone()
    route = [
        *PREFIX, *MACRO1,
        (6, 3, 15), (3,), (6, 3, 45),
    ]
    height = advance(node, route)
    print(
        "START", height, avatar_cell(node.frame()), controls(node.frame()),
        flush=True,
    )
    for depth in range(1, 9):
        options, winning = successors(node)
        if winning is not None:
            print("WIN", [*route, *winning], flush=True)
            return
        if not options:
            print(
                "BLOCKED", depth, height, avatar_cell(node.frame()),
                controls(node.frame()), lattice(node.frame()),
            )
            return
        best = max(
            options,
            key=lambda item: (item[1], item[0], item[2]),
        )
        gain, remaining, central, suffix, child = best
        height += gain
        route.extend(suffix)
        node = child
        print(
            "GREEDY", depth, height, gain, remaining,
            avatar_cell(node.frame()), controls(node.frame()), suffix,
            flush=True,
        )
    print("END", height, route, lattice(node.frame()))


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
