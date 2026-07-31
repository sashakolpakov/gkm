import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import numpy as np

import players
from perception import action_deltas, color_counts, object_candidates, block_signatures


def reach_level_4(env):
    players.play_level_1(env)
    players.play_level_2(env)
    players.play_level_3(env)


def compact_objects(frame):
    return [
        (o["color"], o["bbox"], o["area"])
        for o in object_candidates(frame, min_area=4)
    ]


def compact_blocks(frame):
    sigs = block_signatures(frame)
    rows = []
    for r in range(16):
        row = []
        for c in range(16):
            sig = sigs[(r, c)]
            row.append("".join(f"{v:x}" for v in sig))
        rows.append(" ".join(f"{v:5s}" for v in row))
    return rows


CHARS = {0: "0", 2: "2", 3: "3", 4: "#", 5: ".", 8: "8",
         9: "9", 11: "b", 12: "c", 14: "e", 15: "f"}


def crop(frame, r0, c0, r1, c1):
    a = np.asarray(frame)
    return [
        "".join(CHARS[int(v)] for v in a[r, c0:c1])
        for r in range(r0, r1)
    ]


def non_ui_objects(frame):
    return [
        item for item in compact_objects(frame)
        if item[1][0] >= 16 and item[1][0] < 63
    ]


def active_payload_center(frame):
    active = [
        o for o in object_candidates(frame, min_area=4)
        if o["color"] not in (0, 2, 3, 4, 5)
        and o["area"] == 12
        and o["bbox"][0] >= 16
        and not (34 <= o["bbox"][0] and o["bbox"][2] <= 43
                 and 27 <= o["bbox"][1] and o["bbox"][3] <= 36)
    ]
    if len(active) != 1:
        return None
    r0, c0, r1, c1 = active[0]["bbox"]
    return ((c0 + c1) // 2, (r0 + r1) // 2)


POSITIONS = {
    "N": (),
    "NW": (3,),
    "W": (3, 2),
    "SW": (3, 2, 2),
    "S": (3, 2, 2, 4),
    "NE": (4,),
    "E": (4, 2),
    "SE": (4, 2, 2),
}


def canvas(frame):
    return crop(frame, 34, 27, 44, 37)


def delta_without_counter(before, after):
    a, b = np.asarray(before), np.asarray(after)
    changed = np.argwhere(a != b)
    changed = changed[changed[:, 0] < 63]
    if len(changed) == 0:
        return (0, None)
    return (len(changed), tuple(changed.min(0)) + tuple(changed.max(0)))


def probe(env):
    reach_level_4(env)
    print("LEVEL", env.levels_completed + 1, "ACTIONS", env.actions)
    print("COLORS", color_counts(env.frame()))
    print("OBJECTS")
    for obj in compact_objects(env.frame()):
        print(obj)
    print("BLOCKS")
    for row in compact_blocks(env.frame()):
        print(row)
    print("DELTAS")
    for action, delta in action_deltas(env).items():
        print(action, {k: v for k, v in delta.items() if k != "samples"})
    print("TARGET")
    print("\n".join(crop(env.frame(), 0, 0, 16, 16)))
    print("CENTER")
    print("\n".join(crop(env.frame(), 16, 12, 46, 52)))
    print("ACTION_OBJECTS")
    for action in range(1, 6):
        clone = env.clone()
        clone.step(action)
        print(action, non_ui_objects(clone.frame()))
    print("POSITION_USE_MASKS")
    for name, path in POSITIONS.items():
        clone = env.clone()
        for action in path:
            clone.step(action)
        clone.step(5)
        print(name, path, "/".join(canvas(clone.frame())))
    print("COORD_DELTAS")
    for x, y in ((22, 4), (28, 4), (34, 4), (40, 4), (46, 4),
                 (52, 4), (58, 4), (31, 20), (31, 38)):
        clone = env.clone()
        before = np.asarray(clone.frame()).copy()
        clone.step(6, x, y)
        print((x, y), delta_without_counter(before, clone.frame()),
              non_ui_objects(clone.frame()))
    print("POSITION_PAYLOADS")
    for name, path in POSITIONS.items():
        clone = env.clone()
        for action in path:
            clone.step(action)
        center = active_payload_center(clone.frame())
        print(name, center, non_ui_objects(clone.frame()))
        if center is not None:
            clone.step(6, *center)
            print("STAMP", "/".join(canvas(clone.frame())))
    print("CANDIDATE")
    clone = env.clone()
    for action in (3,):
        clone.step(action)
    clone.step(6, 34, 4)
    clone.step(5)
    for action in (4, 4, 2, 2):
        clone.step(action)
    clone.step(6, 28, 4)
    clone.step(5)
    for action in (3, 3, 1):
        clone.step(action)
    clone.step(6, 58, 4)
    clone.step(5)
    clone.step(6, 40, 4)
    center = active_payload_center(clone.frame())
    print("PAYLOAD", center)
    clone.step(6, *center)
    print("LEVELS", clone.levels_completed)
    print("\n".join(canvas(clone.frame())))


A.run_program("cd82", probe)
