import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import click_action
from probe_level7_reward_recovery import (
    PREFIX, SUFFIX, advance, avatar_cell, controls, lattice,
)


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    root = env.clone()
    base_height = advance(root, [*PREFIX, click_action(5, 2), *SUFFIX])
    print(
        "ROOT", base_height, avatar_cell(root.frame()), controls(root.frame()),
        lattice(root.frame()),
    )
    cross_patterns = [
        [],
        [(3,)], [(3,), (3,)], [(3,), (3,), (3,)],
        [(4,)], [(4,), (4,)], [(4,), (4,), (4,)],
    ]
    outcomes = {}
    for pre in ([], [(3,)], [(4,)]):
        staged = root.clone()
        pre_gain = advance(staged, pre)
        for y1 in controls(staged.frame()):
            flipped = staged.clone()
            first_gain = pre_gain + advance(flipped, [(6, 3, y1)])
            if flipped.terminal():
                continue
            for cross in cross_patterns:
                middle = flipped.clone()
                middle_gain = first_gain + advance(middle, cross)
                if middle.terminal():
                    continue
                for y2 in controls(middle.frame()):
                    landed = middle.clone()
                    gain = middle_gain + advance(landed, [(6, 3, y2)])
                    route = [*pre, (6, 3, y1), *cross, (6, 3, y2)]
                    if landed.levels_completed > 6:
                        print("WIN", route)
                        return
                    if landed.terminal():
                        continue
                    walked = landed.clone()
                    walked_gain = gain
                    for count in range(5):
                        avatar = avatar_cell(walked.frame())
                        key = (
                            walked_gain, avatar, tuple(controls(walked.frame())),
                        )
                        outcomes.setdefault(
                            key, [*route, *([(4,)] * count)]
                        )
                        if avatar is not None and avatar[1] >= 6:
                            print(
                                "RIGHT_LANDING", base_height + walked_gain,
                                avatar, controls(walked.frame()),
                                outcomes[key], lattice(walked.frame()),
                            )
                            return
                        if walked.levels_completed > 6 or walked.terminal():
                            break
                        walked_gain += advance(walked, [(4,)])
    ordered = sorted(
        outcomes.items(),
        key=lambda item: (
            -item[0][0],
            -(item[0][1][1] if item[0][1] else -1),
            -len(item[0][2]),
        ),
    )
    print("NO_RIGHT_LANDING", len(ordered))
    for state, route in ordered[:20]:
        print("OPTION", base_height + state[0], state[1], state[2], route)


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
