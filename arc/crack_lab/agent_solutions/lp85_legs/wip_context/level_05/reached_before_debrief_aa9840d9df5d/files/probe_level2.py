import sys
from collections import Counter

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import arr, color_counts, frame_delta, object_candidates


def transition_summary(before, after):
    a, b = arr(before), arr(after)
    return dict(sorted(Counter((int(x), int(y)) for x, y in zip(a[a != b], b[a != b])).items()))


def tokens(frame):
    return {(o["bbox"][0], o["bbox"][1]): o["color"]
            for o in object_candidates(frame, min_area=4)
            if o["size"] == (2, 2) and o["area"] == 4 and o["bbox"][0] >= 16}


def token_rows(frame):
    ts = tokens(frame)
    return ["%02d:" % r + ",".join("%02d=%x" % (c, ts[(r, c)])
                                     for c in sorted(c for rr, c in ts if rr == r))
            for r in sorted(set(r for r, _ in ts))]


def probe(env):
    for _ in range(5):
        env.step(6, 5, 32)
    print("level", env.levels_completed + 1, "actions", env.actions)
    print("colors", color_counts(env.frame()))
    objects = object_candidates(env.frame(), min_area=4)
    for obj in objects:
        print("obj", obj["color"], obj["bbox"], obj["area"])
    tests = {
        "empty": (5, 5),
        "board_c1": (17, 26),
        "board_c2": (23, 23),
        "board_c9": (26, 17),
        "left8": (20, 17),
        "right14": (39, 17),
    }
    before = env.frame()
    print("token_rows", *token_rows(before))
    for name, (x, y) in tests.items():
        clone = env.clone()
        clone.step(6, x, y)
        print("click", name, (x, y), frame_delta(before, clone.frame())["count"],
              transition_summary(before, clone.frame()), "level", clone.levels_completed)
        if name in ("left8", "right14"):
            print("after", name, *token_rows(clone.frame()))
    buttons = ((20, 17), (14, 26), (14, 35), (39, 17), (48, 26), (48, 35))
    initial_tokens = tokens(before)
    for i, (x, y) in enumerate(buttons):
        clone = env.clone()
        order = None
        for n in range(1, 13):
            clone.step(6, x, y)
            if tokens(clone.frame()) == initial_tokens:
                order = n
                break
        print("order", i, order, "level", clone.levels_completed)


A.run_program("lp85", probe)
