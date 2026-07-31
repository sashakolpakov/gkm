"""Compact staged probes for bp35 level 4."""
import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from legs import COL_ANCHORS, ROW_ANCHORS, band_grid, climb_by_search
from perception import connected_components, frame_delta


PREFIX = json.load(open("level4_prefix.json"))
TRANSITION = [(4,), (4,), (4,), (4,), (6, COL_ANCHORS[3], ROW_ANCHORS[4])]


def advance(env):
    for action in PREFIX:
        env.step(*(action if isinstance(action, list) else [action]))
    for action in TRANSITION:
        env.step(*action)
    assert env.levels_completed == 3


def symbol(env):
    return "/".join("".join(row) for row in band_grid(env.frame()))


def objects(env):
    return [(b.color, b.bbox, b.area) for b in connected_components(
        env.frame(), colors=(8, 9, 11, 14, 15), min_area=1)]


def interaction_effects(env, label):
    effects = []
    before_grid = band_grid(env.frame())
    for i, y in enumerate(ROW_ANCHORS):
        for j, x in enumerate(COL_ANCHORS):
            child = env.clone()
            before = child.frame()
            child.step(6, x, y)
            delta = frame_delta(before, child.frame())
            if delta["count"] > 1:
                avatar = [(b.color, b.bbox) for b in connected_components(
                    child.frame(), colors=(9,), min_area=1)]
                effects.append((i, j, before_grid[i][j], delta["count"],
                                child.levels_completed, child.terminal(),
                                symbol(child), avatar))
    print("STATE_EFFECTS", label, effects)


def probe(env):
    advance(env)
    print("START", tuple(env.actions), symbol(env), objects(env))

    for key in (3, 4, 6, 7):
        child = env.clone()
        before = child.frame()
        try:
            child.step(key)
            delta = frame_delta(before, child.frame())
            print("KEY", key, delta["count"], delta["bbox"],
                  child.levels_completed, child.terminal(), symbol(child), objects(child))
        except Exception as exc:
            print("KEY", key, "ERR", type(exc).__name__)

    for coordinate_action in (6, 7):
        effects = []
        for i, y in enumerate(ROW_ANCHORS):
            for j, x in enumerate(COL_ANCHORS):
                child = env.clone()
                before = child.frame()
                try:
                    child.step(coordinate_action, x, y)
                except Exception:
                    continue
                delta = frame_delta(before, child.frame())
                if delta["count"] > 1:
                    effects.append((i, j, band_grid(before)[i][j],
                                    delta["count"], delta["bbox"],
                                    child.levels_completed, child.terminal(),
                                    symbol(child), objects(child)))
        print("COORD", coordinate_action, effects)

    for key in (3, 4):
        child = env.clone()
        trace = []
        for n in range(12):
            before = child.frame()
            child.step(key)
            delta = frame_delta(before, child.frame())
            trace.append((n + 1, delta["count"], child.levels_completed,
                          child.terminal(), symbol(child), objects(child)))
            if child.terminal() or child.levels_completed > 3:
                break
        print("REPEAT", key, trace)

    tests = {
        "left6_drop0": [(3,)] * 6 + [(6, COL_ANCHORS[0], ROW_ANCHORS[5])],
        "left5_drop1": [(3,)] * 5 + [(6, COL_ANCHORS[1], ROW_ANCHORS[5])],
        "open0_left6": [(6, COL_ANCHORS[0], ROW_ANCHORS[5])] + [(3,)] * 6,
        "open1_left5": [(6, COL_ANCHORS[1], ROW_ANCHORS[5])] + [(3,)] * 5,
        "clear_lower_drop0": (
            [(6, COL_ANCHORS[j], ROW_ANCHORS[7]) for j in range(3, 7)]
            + [(3,)] * 6 + [(6, COL_ANCHORS[0], ROW_ANCHORS[5])]
        ),
    }
    for name, actions in tests.items():
        child = env.clone()
        trace = []
        for action in actions:
            child.step(*action)
            trace.append((action, child.levels_completed, child.terminal(),
                          symbol(child), objects(child)))
            if child.terminal() or child.levels_completed > 3:
                break
        print("TEST", name, trace)

    stage2 = env.clone()
    for action in tests["left6_drop0"]:
        stage2.step(*action)
    interaction_effects(stage2, "after_drop0")
    next_tests = {
        "right1_clear_top1": [(4,), (6, COL_ANCHORS[1], ROW_ANCHORS[0])],
        "right3_clear_upper3": [(4,)] * 3 + [
            (6, COL_ANCHORS[3], ROW_ANCHORS[2])
        ],
        "right1_toggle1": [(4,), (6, COL_ANCHORS[1], ROW_ANCHORS[6])],
        "right1_toggle1_clear": [
            (4,),
            (6, COL_ANCHORS[1], ROW_ANCHORS[6]),
            (6, COL_ANCHORS[1], ROW_ANCHORS[5]),
        ],
        "right3_toggle3": [(4,)] * 3 + [
            (6, COL_ANCHORS[3], ROW_ANCHORS[6])
        ],
        "right3_toggle3_clear": (
            [(4,)] * 3
            + [(6, COL_ANCHORS[3], ROW_ANCHORS[6])]
            + [(6, COL_ANCHORS[3], ROW_ANCHORS[5])]
        ),
    }
    for name, actions in next_tests.items():
        child = stage2.clone()
        for action in actions:
            child.step(*action)
            if child.terminal() or child.levels_completed > 3:
                break
        print("NEXT", name, child.levels_completed, child.terminal(),
              symbol(child), objects(child))

    resumed = stage2.clone()
    for action in next_tests["right3_toggle3_clear"]:
        resumed.step(*action)
    solved = climb_by_search(resumed, max_expansions=300)
    print("RESUMED_SEARCH", solved, resumed.levels_completed, resumed.terminal(),
          symbol(resumed))
    for key in (3, 4):
        child = resumed.clone()
        trace = []
        for _ in range(8):
            child.step(key)
            trace.append((child.levels_completed, child.terminal(), symbol(child)))
            if child.terminal() or child.levels_completed > 3:
                break
        print("RESUMED_MOVE", key, trace)


print(A.run_program("bp35", probe))
