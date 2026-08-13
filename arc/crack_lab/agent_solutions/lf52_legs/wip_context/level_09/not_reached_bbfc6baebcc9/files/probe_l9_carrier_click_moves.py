"""Test carrier-border plus rail-cell coordinate interactions on level 9."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, frame_delta, safe_step


FIRST_RELAY = (
    ((42, 18), (42, 30)), ((48, 24), (36, 24)),
    ((42, 24), (30, 24)), ((36, 24), (24, 24)),
    ((30, 24), (18, 24)), ((18, 24), (18, 36)),
    ((18, 36), (30, 36)), ((24, 36), (36, 36)),
    ((30, 36), (42, 36)), ((36, 36), (48, 36)),
    ((48, 42), (48, 30)), ((48, 30), (36, 30)),
    ((48, 36), (36, 36)), ((36, 30), (36, 42)),
)


def move(env, source, destination):
    safe_step(env, (6, source[1] + 1, source[0] + 1))
    safe_step(env, (6, destination[1] + 1, destination[0] + 1))


def delta(before, after):
    left = arr(before).copy()
    right = arr(after).copy()
    right[0] = left[0]
    out = frame_delta(left, right)
    return out["count"], out["bbox"]


def interactions(root, label):
    carrier_pixels = ((21, 35), (26, 35), (21, 40), (26, 40),
                      (22, 36), (25, 39))
    rail_pixels = tuple((col + 1, 37) for col in (4, 10, 16, 28, 34,
                                                  40, 46, 52, 58))
    base = root.frame()
    changed = []
    tested = 0
    for carrier_pixel in carrier_pixels:
        carrier_action = (6, carrier_pixel[0], carrier_pixel[1])
        for rail_pixel in rail_pixels:
            rail_action = (6, rail_pixel[0], rail_pixel[1])
            for actions in ((carrier_action, rail_action),
                            (rail_action, carrier_action)):
                tested += 1
                child = root.clone()
                for action in actions:
                    safe_step(child, action)
                change = delta(base, child.frame())
                if change[0]:
                    changed.append((actions, change,
                                    int(child.levels_completed)))
    print(label, tested, tuple(changed), flush=True)


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    for action in campaign:
        env.step(action)
    for source, destination in FIRST_RELAY:
        move(env, source, destination)

    interactions(env, "carrier_click_loaded")

    empty = env.clone()
    move(empty, (36, 22), (36, 10))
    interactions(empty, "carrier_click_empty")

    midpoint = empty.clone()
    before = midpoint.frame()
    move(midpoint, (36, 16), (36, 28))
    print("empty_carrier_midpoint", delta(before, midpoint.frame()),
          int(midpoint.levels_completed), flush=True)


if __name__ == "__main__":
    arena.run_program("lf52", probe)
