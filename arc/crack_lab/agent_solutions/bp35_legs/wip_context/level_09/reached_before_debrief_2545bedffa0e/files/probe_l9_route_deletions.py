"""Find individually removable actions in the verified route to chamber two."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

import solve
from legs import band_grid, click_action
from perception import connected_components


def enter_level_9(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)
    solve.solve(env)


def route():
    actions = []

    def add(section, *items):
        actions.extend((section, item) for item in items)

    add("open", (6, 3, 39), 4, 4, 4, (6, 3, 5))
    add("stage", click_action(5, 4), click_action(1, 2))
    for col in (3, 2):
        add("stage", click_action(6, col), 3)
    add("upper_climb", *([click_action(5, 2)] * 10))
    for col in (3, 4, 5):
        add("upper_cross", click_action(6, col), 4)
    add("upper_flip", (6, 3, 3))
    add("right_drop", click_action(5, 5), click_action(5, 5))
    for col in (6, 7):
        add("right_cross", click_action(4, col), 4)
    add("right_flip", (6, 3, 5))
    add("right_climb", *([click_action(5, 7)] * 7))
    add("k_clear", click_action(3, 2))
    for col in (6, 5):
        add("k_cross", click_action(6, col), 3)
    add("k_climb", click_action(5, 5))
    add("control_approach", (6, 9, 33), (6, 45, 33))
    for x in (39, 33, 27, 21, 15, 9, 3):
        add("control_cross", (6, x, 39), 3)
    add("control_climb", (6, 3, 33))
    add("outer_flip", (6, 9, 3))
    add("outer_drop", *([(6, 3, 33)] * 8), (6, 3, 59))
    add(
        "gap_bridge",
        *((6, x, 45) for x in (51, 45, 39, 33, 27, 21, 15, 9)),
    )
    add("gap_flip_drop", (6, 15, 3), (6, 3, 35))
    for col in range(1, 10):
        add("lane9_cross", (6, 3 + 6 * col, 27), 4)
    add("lane9_drop", *([(6, 57, 35)] * 7), (6, 3, 41))
    add("return_left", *([3] * 9))
    return actions


def step(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def visible(env, color):
    return tuple(
        blob.bbox
        for blob in connected_components(env.frame(), colors=(color,), min_area=3)
        if blob.bbox[0] < 63
    )


def summary(env):
    return {
        "terminal": bool(env.terminal()),
        "level": int(env.levels_completed) + 1,
        "avatar": visible(env, 9) + visible(env, 11),
        "controls": visible(env, 8),
        "grid": tuple("".join(row) for row in band_grid(env.frame())),
    }


def replay(root, actions, skips=()):
    child = root.clone()
    for index, (_, action) in enumerate(actions):
        if index in skips:
            continue
        step(child, action)
        if child.terminal() or int(child.levels_completed) >= 9:
            break
    return child


def probe(env):
    enter_level_9(env)
    root = env.clone()
    actions = route()
    baseline = replay(root, actions)
    target = summary(baseline)
    print("BASE", len(actions), target)

    candidates = [
        index
        for index, (section, _) in enumerate(actions)
        if section
        in {
            "open",
            "stage",
            "upper_climb",
            "upper_cross",
            "right_drop",
            "right_cross",
            "right_climb",
            "k_clear",
            "k_cross",
            "k_climb",
            "control_approach",
            "control_cross",
            "control_climb",
        }
    ]
    for ordinal, index in enumerate(candidates, 1):
        child = replay(root, actions, skips=(index,))
        result = summary(child)
        if (
            not result["terminal"]
            and result["level"] == 9
            and result["avatar"] == target["avatar"]
            and len(result["controls"]) == len(target["controls"])
        ):
            print("CANDIDATE", index, actions[index], result)
        if ordinal % 10 == 0:
            print("PROGRESS", ordinal, "of", len(candidates))


if __name__ == "__main__":
    arena.run_program("bp35", probe)
