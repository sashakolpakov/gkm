import json
import itertools
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import ROW_ANCHORS, _cell_shape, click_action
from probe_level7_reward_recovery import (
    PREFIX, SUFFIX, advance, avatar_cell, controls,
)


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


BASE_OPENING = [(2, 2), (4, 2), (4, 4), (1, 3)]


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    supports = [
        (i, j)
        for i in range(10)
        for j in range(8)
        if _cell_shape(env.frame(), i, j)[0] == 12
        and (i, j) not in set(BASE_OPENING)
    ]
    outcomes = []
    for support in supports:
        route = [
            *PREFIX[:4],
            click_action(*support),
            *PREFIX[4:],
            click_action(5, 2),
            *SUFFIX,
            (3,), (6, 3, 9), (4,), (6, 3, 39),
            (3,), (3,), (3,),
        ]
        staged = env.clone()
        height = advance(staged, route)
        if staged.terminal():
            outcomes.append((6, False, height, support, None, ()))
            continue
        visible = controls(staged.frame())
        if not visible:
            outcomes.append(
                (
                    staged.levels_completed, True, height, support,
                    avatar_cell(staged.frame()), (),
                )
            )
            continue
        for y in visible:
            node = staged.clone()
            node.step(6, 3, y)
            outcome = (
                node.levels_completed,
                not node.terminal(),
                height,
                support,
                None if node.terminal() else avatar_cell(node.frame()),
                () if node.terminal() else tuple(controls(node.frame())),
            )
            outcomes.append(outcome)
            if node.levels_completed > 6:
                print("WIN", support, y, route, flush=True)
                return
    outcomes.sort(
        key=lambda item: (
            -item[0], -item[1], -item[2], item[3], item[4] or (-1, -1),
        )
    )
    print("NO_OPENING_WIN", len(outcomes))
    for outcome in outcomes:
        print("OPENING", outcome)

    subset_outcomes = []
    for count in range(len(BASE_OPENING) + 1):
        for subset in itertools.combinations(BASE_OPENING, count):
            route = [
                *(click_action(*cell) for cell in subset),
                *PREFIX[4:],
                click_action(5, 2),
                *SUFFIX,
                (3,), (6, 3, 9), (4,), (6, 3, 39),
            ]
            root = env.clone()
            height = advance(root, route)
            if root.terminal():
                subset_outcomes.append((False, height, subset, None, ()))
                continue
            best = (
                True, height, subset, avatar_cell(root.frame()),
                tuple(controls(root.frame())),
            )
            for left_count in range(7):
                staged = root.clone()
                advance(staged, [(3,)] * left_count)
                for y in controls(staged.frame()):
                    node = staged.clone()
                    node.step(6, 3, y)
                    if node.levels_completed > 6:
                        print(
                            "SUBSET_WIN", subset, left_count, y,
                            [
                                *route, *([(3,)] * left_count), (6, 3, y)
                            ],
                            flush=True,
                        )
                        return
                    if not node.terminal():
                        candidate = (
                            True, height, subset,
                            avatar_cell(node.frame()),
                            tuple(controls(node.frame())),
                        )
                        if candidate[3:] > best[3:]:
                            best = candidate
            subset_outcomes.append(best)
    subset_outcomes.sort(
        key=lambda item: (-item[0], -item[1], -len(item[4]), item[2])
    )
    print("NO_SUBSET_WIN", len(subset_outcomes))
    for outcome in subset_outcomes:
        print("SUBSET", outcome)


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
