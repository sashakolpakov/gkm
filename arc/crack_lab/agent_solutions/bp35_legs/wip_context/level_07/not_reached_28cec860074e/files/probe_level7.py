import importlib.util
import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from perception import color_counts, connected_components, frame_delta
from legs import (
    COL_ANCHORS, ROW_ANCHORS, _cell_shape, avatar_column, band_shift,
    click_action, cross_persistent_support_rooms, gravity_room_search,
    run_actions,
)


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")

spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


def symbolic(frame):
    palette = {3: "#", 5: "#", 9: "A", 10: ".", 11: "A", 14: "Y"}
    return [
        "".join(palette.get(int(frame[y][x]), hex(int(frame[y][x]))[-1])
                for x in COL_ANCHORS)
        for y in ROW_ANCHORS
    ]


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    print("LEVEL", env.levels_completed + 1, "ACTIONS", env.actions)
    frame = env.frame()
    print("COUNTS", color_counts(frame))
    print("GRID", "/".join(symbolic(frame)))
    blobs = connected_components(frame, colors=(9, 11, 14), min_area=2)
    print("OBJECTS", [(b.color, b.bbox, b.area) for b in blobs])
    mechanics = connected_components(frame, colors=(8, 12, 15), min_area=2)
    print("MECHANICS", [(b.color, b.bbox, b.area) for b in mechanics])
    route = gravity_room_search(
        env.clone(), max_states=5000, max_depth=80,
        support_radius=2, edge_gravity=True,
    )
    verified = env.clone()
    run_actions(verified, route)
    print("SEARCH", len(route), route)
    print("VERIFY", verified.levels_completed, verified.terminal())
    for cell in ((6, 4), (8, 4), (4, 1)):
        i, j = cell
        pts = [
            (x, y)
            for y in range(6 * i, 6 * i + 6)
            for x in range(13 + 6 * j, 13 + 6 * j + 6)
            if int(frame[y][x]) == int(frame[ROW_ANCHORS[i]][COL_ANCHORS[j]])
        ]
        outcomes = []
        for x, y in pts:
            node = env.clone()
            node.step(6, x, y)
            outcomes.append(((x, y), _cell_shape(node.frame(), i, j),
                             avatar_column(node.frame())))
        print("PIXELS", cell, outcomes)
    all_supports = [
        click_action(i, j)
        for i in range(10) for j in range(8)
        if int(frame[ROW_ANCHORS[i]][COL_ANCHORS[j]]) == 12
    ]
    expanded = env.clone()
    run_actions(expanded, all_supports)
    print("ALL_SUPPORTS", len(all_supports), expanded.levels_completed,
          expanded.terminal(), avatar_column(expanded.frame()))
    prior = env.clone()
    cross_persistent_support_rooms(prior)
    print("PRIOR_LEG", prior.levels_completed, prior.terminal(),
          avatar_column(prior.frame()), band_shift(frame, prior.frame()))
    def summary(node):
        f = node.frame()
        shapes = [
            (i, j, _cell_shape(f, i, j)[1])
            for i in range(10) for j in range(8)
            if _cell_shape(f, i, j)[0] in (12, 15)
        ]
        return (node.levels_completed, node.terminal(), avatar_column(f),
                band_shift(frame, f), shapes, "/".join(symbolic(f)))

    c64 = click_action(6, 4)
    c84 = click_action(8, 4)
    f41 = click_action(4, 1)
    f45 = click_action(4, 5)
    gravity = (6, 3, 3)
    paths = [
        (("c84 gravity",), [c84, gravity]),
        (("c84 gravity R",), [c84, gravity, (4,)]),
        (("c84 gravity RR",), [c84, gravity, (4,), (4,)]),
        (("c84 gravity RRR",), [c84, gravity, (4,), (4,), (4,)]),
        (("c84 gravity L",), [c84, gravity, (3,)]),
        (("f41 R",), [f41, (4,)]),
        (("f41 RR",), [f41, (4,), (4,)]),
        (("f45 R",), [f45, (4,)]),
        (("RRR c84 f41 L",), [(4,), (4,), (4,), c84, f41, (3,)]),
        (("RRR c84 f45 L",), [(4,), (4,), (4,), c84, f45, (3,)]),
        (("RRR c84 f45 R",), [(4,), (4,), (4,), c84, f45, (4,)]),
        (("RRR oscillate",), [(4,), (4,), (4,)] + [(3,), (4,)] * 12),
        (("RR",), [(4,), (4,)]),
        (("RRR",), [(4,), (4,), (4,)]),
        (("RRR c64",), [(4,), (4,), (4,), c64]),
        (("RRR c64 L",), [(4,), (4,), (4,), c64, (3,)]),
        (("RRR c64 R",), [(4,), (4,), (4,), c64, (4,)]),
        (("RRR c64 c64",), [(4,), (4,), (4,), c64, c64]),
        (("RRR c84",), [(4,), (4,), (4,), c84]),
        (("RRR c84 L",), [(4,), (4,), (4,), c84, (3,)]),
        (("RRR c84 R",), [(4,), (4,), (4,), c84, (4,)]),
        (("RRR c84 c84 R",), [(4,), (4,), (4,), c84, c84, (4,)]),
        (("RR c64",), [(4,), (4,), c64]),
        (("RR c64 R",), [(4,), (4,), c64, (4,)]),
        (("RR c64 c64",), [(4,), (4,), c64, c64]),
        (("c64 RR c64 R",), [c64, (4,), (4,), c64, (4,)]),
    ]
    for label, route in paths:
        node = env.clone()
        run_actions(node, route)
        print("PATH", label[0], summary(node))


levels, path, err = A.run_program("bp35", probe)
print("PROBE_RESULT", levels, len(path), err)
