import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A
import gkm_legs as G

from perception import arr, frame_delta
from players import play_level_1


HUB_PATH = (1, 4, 1, 1, 4, 4, 4, 4, 4, 4, 4)
BOTTOM_PATH = (
    (6, 52, 40),
    2, 2, 2, 2, 2,
    4, 4, 4, 4, 4,
    (6, 52, 22),
    2, 2, 2, 2, 2, 2,
)
SOLUTION = (
    (6, 52, 40),
    2, 2, 2, 2, 2,
    4, 4, 4, 4, 4,
    (6, 52, 22),
    2, 2, 2, 2, 2, 2,
    1, 1, 1, 1, 1, 1,
    (6, 52, 22),
    3, 3, 3, 3, 3,
    1, 1, 1, 1, 1,
    (6, 52, 40),
    1, 4, 1, 1,
    4, 4, 4, 4, 4, 4, 4,
    (6, 52, 31),
    1, 1, 1, 1, 1, 1,
)


def compact_click_scan(env, label):
    base = arr(env.frame()).copy()
    effects = {}
    for y in range(0, 56, 2):
        for x in range(0, 64, 2):
            child = env.clone()
            child.step(6, x, y)
            delta = frame_delta(base, child.frame())
            visible = tuple(
                sample for sample in delta["samples"] if sample[0] < 56
            )
            if visible:
                key = (delta["count"], delta["bbox"], visible)
                effects.setdefault(key, []).append((x, y))
    print(label, "effect_classes", len(effects))
    for key, points in effects.items():
        print(" ", "points", (points[0], points[-1], len(points)),
              "delta", key)


def click_then_move(env, label, points):
    print(label)
    for point in points:
        changed = []
        for action in (1, 2, 3, 4):
            control = env.clone()
            control.step(action)
            selected = env.clone()
            selected.step(6, *point)
            selected.step(action)
            if not np.array_equal(arr(control.frame()), arr(selected.frame())):
                changed.append(action)
        print(" ", point, "changed_actions", changed)


def probe(env):
    play_level_1(env)
    for label, node in (("start-use", env.clone()),):
        before = arr(node.frame()).copy()
        try:
            node.step(6)
            print(label, frame_delta(before, node.frame()))
        except Exception as exc:
            print(label, type(exc).__name__, str(exc))
    compact_click_scan(env, "start")
    click_then_move(
        env, "start-click-then-move",
        ((22, 25), (14, 25), (30, 25), (22, 12), (6, 30)),
    )

    hub = env.clone()
    before = arr(hub.frame()).copy()
    for action in HUB_PATH:
        hub.step(action)
    delta = frame_delta(before, hub.frame())
    non_avatar = [
        sample for sample in delta["samples"]
        if sample[0] < 56 and sample[2] != 14 and sample[3] != 14
    ]
    ys, xs = np.where(arr(hub.frame())[:56] == 14)
    print("hub", "avatar", (int(ys.min()), int(xs.min())),
          "non_avatar", non_avatar)
    for action in (1, 2, 3, 4):
        moved = hub.clone()
        moved_before = arr(moved.frame()).copy()
        moved.step(action)
        print("hub-move", action, frame_delta(moved_before, moved.frame()))
    for point in ((22, 25), (22, 12), (14, 25), (30, 25)):
        reward_probe = hub.clone()
        reward_probe.step(6, *point)
        print("hub-click-reward", point, reward_probe.levels_completed,
              reward_probe.terminal())
    before_use = arr(hub.frame()).copy()
    try:
        hub.step(6)
        print("hub-use", frame_delta(before_use, hub.frame()))
    except Exception as exc:
        print("hub-use", type(exc).__name__, str(exc))
    compact_click_scan(hub, "hub")
    for point in ((22, 25), (14, 25), (30, 25)):
        selected_hub = hub.clone()
        selected_hub.step(6, *point)
        compact_click_scan(selected_hub, "hub-after-click-" + str(point))
    click_then_move(
        hub, "hub-click-then-move",
        ((22, 25), (14, 25), (30, 25), (22, 12), (22, 24)),
    )

    bottom = env.clone()
    for index, action in enumerate(BOTTOM_PATH, 1):
        before = arr(bottom.frame()).copy()
        if isinstance(action, tuple):
            bottom.step(*action)
        else:
            bottom.step(action)
        if index >= len(BOTTOM_PATH) - 2:
            ys, xs = np.where(arr(bottom.frame())[:56] == 14)
            delta = frame_delta(before, bottom.frame())
            print("bottom-step", index, action,
                  (int(ys.min()), int(xs.min())),
                  delta["count"], delta["bbox"])
    for point in ((17, 53), (18, 53), (17, 54), (16, 52)):
        child = bottom.clone()
        before = arr(child.frame()).copy()
        child.step(6, *point)
        print("bottom-click", point, child.levels_completed,
              frame_delta(before, child.frame()))
    compact_click_scan(bottom, "bottom")

    solved = env.clone()
    for action in SOLUTION:
        if isinstance(action, tuple):
            solved.step(*action)
        else:
            solved.step(action)
    print("solution", solved.levels_completed, solved.terminal(),
          len(SOLUTION))


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted")
A.run_program("dc22", probe)
