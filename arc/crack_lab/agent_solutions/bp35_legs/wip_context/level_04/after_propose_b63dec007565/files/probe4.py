"""Compact clean-room probes for bp35 level 4."""
import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from legs import (AVATAR_ROW, COL_ANCHORS, GRID_COLS, GRID_ROWS, ROW_ANCHORS,
                  avatar_column, band_grid, band_shift, moves_used)
from perception import color_counts, connected_components, frame_delta
from players import play_level_1, play_level_2, play_level_3


def advance(env):
    try:
        with open("level4_prefix.json") as f:
            prefix = json.load(f)
        for action in prefix:
            env.step(*(action if isinstance(action, list) else [action]))
        assert env.levels_completed == 3
        return
    except FileNotFoundError:
        pass
    for player in (play_level_1, play_level_2, play_level_3):
        player(env)
    assert env.levels_completed == 3


def summary(env, label):
    frame = env.frame()
    print(label, "lvl", env.levels_completed, "term", env.terminal(),
          "actions", tuple(env.actions), "moves", moves_used(frame),
          "col", avatar_column(frame), "colors", color_counts(frame))
    for i, row in enumerate(band_grid(frame)):
        print(i, "".join(row), "<" if i == AVATAR_ROW else "")
    blobs = connected_components(frame, min_area=4)
    print("blobs", [(b.color, b.bbox, b.area) for b in blobs
                    if b.color not in (3, 5, 10)])


def avatar_bbox(env):
    bs = connected_components(env.frame(), colors=(9, 11), min_area=1)
    return [(b.color, b.bbox, b.area) for b in bs]


def probe(env):
    advance(env)
    summary(env, "START")
    base = env.frame()
    for action in env.actions:
        c = env.clone()
        try:
            c.step(int(action))
            print("KEY", action, "delta", frame_delta(base, c.frame()),
                  "col", avatar_column(c.frame()), "shift", band_shift(base, c.frame()),
                  "lvl", c.levels_completed, "term", c.terminal())
        except Exception as exc:
            print("KEY", action, "ERR", type(exc).__name__, str(exc))
    for action in (6, 7):
        effects = []
        for i in range(GRID_ROWS):
            for j in range(GRID_COLS):
                c = env.clone()
                try:
                    c.step(action, COL_ANCHORS[j], ROW_ANCHORS[i])
                except Exception:
                    continue
                d = frame_delta(base, c.frame())
                if d["count"]:
                    effects.append((i, j, band_grid(base)[i][j], d["count"],
                                    d["bbox"], avatar_column(c.frame()),
                                    band_shift(base, c.frame()),
                                    c.levels_completed, c.terminal()))
        print("COORD", action, effects)
    c = env.clone()
    for n in range(1, 13):
        before = c.frame()
        c.step(4)
        print("RIGHT", n, "col", avatar_column(c.frame()),
              "shift", band_shift(before, c.frame()),
              "lvl", c.levels_completed, "term", c.terminal(),
              "grid", "/".join("".join(r) for r in band_grid(c.frame())))
        if c.terminal() or c.levels_completed > 3:
            break
    tests = {
        "lefts": [(3,)] * 8,
        "clear0_lefts": [(6, COL_ANCHORS[0], ROW_ANCHORS[9])] + [(3,)] * 8,
        "clear_both_lefts": [
            (6, COL_ANCHORS[0], ROW_ANCHORS[9]),
            (6, COL_ANCHORS[1], ROW_ANCHORS[9]),
        ] + [(3,)] * 8,
        "climb_then_left": [(4,)] * 4 + [(3,)] * 8,
        "climb_click_star": [(4,)] * 4 + [(6, COL_ANCHORS[3], ROW_ANCHORS[4])],
        "climb_click_below": [(4,)] * 4 + [(6, COL_ANCHORS[3], ROW_ANCHORS[5])],
        "climb_click_star_rights": (
            [(4,)] * 4 + [(6, COL_ANCHORS[3], ROW_ANCHORS[4])] + [(4,)] * 8
        ),
        "climb_click_star_lefts": (
            [(4,)] * 4 + [(6, COL_ANCHORS[3], ROW_ANCHORS[4])] + [(3,)] * 8
        ),
    }
    for name, actions in tests.items():
        c = env.clone()
        total = 0
        trace = []
        for action in actions:
            before = c.frame()
            c.step(*action)
            gain = band_shift(before, c.frame())
            total += gain
            trace.append((avatar_column(c.frame()), gain, c.levels_completed,
                          c.terminal(), color_counts(c.frame()).get(15, 0),
                          avatar_bbox(c)))
            if c.terminal() or c.levels_completed > 3:
                break
        print("TEST", name, "height", total, "trace", trace,
              "grid", "/".join("".join(r) for r in band_grid(c.frame())))
    climbed = env.clone()
    for _ in range(4):
        climbed.step(4)
    base = climbed.frame()
    effects = []
    for i, y in enumerate(ROW_ANCHORS):
        for j, x in enumerate(COL_ANCHORS):
            c = climbed.clone()
            c.step(6, x, y)
            d = frame_delta(base, c.frame())
            if d["count"] > 1:
                effects.append((i, j, band_grid(base)[i][j], d["count"],
                                d["bbox"], c.levels_completed, c.terminal(),
                                avatar_bbox(c)))
    print("POST_CLIMB_CLICKS", effects)


result = A.run_program("bp35", probe)
print("run", result)
if result[0] == 3:
    with open("level4_prefix.json", "w") as f:
        json.dump(result[1], f)
