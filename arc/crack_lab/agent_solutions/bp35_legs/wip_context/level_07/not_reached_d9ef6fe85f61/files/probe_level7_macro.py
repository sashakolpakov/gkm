import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from perception import arr, bounded_replay_bfs, connected_components
from legs import CLICK, COL_ANCHORS, ROW_ANCHORS, run_actions


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


def edge_row(frame):
    return next((i for i, y in enumerate(ROW_ANCHORS)
                 if int(frame[y][3]) == 8), 10)


def avatar_cell(frame):
    blobs = connected_components(frame, colors=(9,), min_area=2)
    if not blobs:
        return None
    y0, x0, y1, x1 = blobs[0].bbox
    return min(range(10), key=lambda i: abs(ROW_ANCHORS[i] - (y0 + y1) // 2)), \
        min(range(8), key=lambda j: abs(COL_ANCHORS[j] - (x0 + x1) // 2))


def actions(node):
    frame = node.frame()
    out = [(3,), (4,)]
    for y in ROW_ANCHORS:
        if int(frame[y][3]) == 8:
            out.append((CLICK, 3, y))
            break
    avatar = avatar_cell(frame)
    if avatar is None:
        return out
    ai, aj = avatar
    for i, y in enumerate(ROW_ANCHORS):
        for j, x in enumerate(COL_ANCHORS):
            if (int(frame[y][x]) in (12, 14)
                    and abs(i - ai) <= 2 and abs(j - aj) <= 1):
                out.append((CLICK, x, y))
    return out


def grid(frame):
    palette = {3: "#", 5: "#", 8: "g", 9: "A", 10: ".", 11: "A",
               12: "c", 14: "Y", 15: "f"}
    return "/".join(
        "".join(palette.get(int(frame[y][x]), "?") for x in COL_ANCHORS)
        for y in ROW_ANCHORS
    )


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    start = edge_row(env.frame())
    route = bounded_replay_bfs(
        env.clone(),
        lambda node, path: edge_row(node.frame()) > start,
        actions,
        key_fn=lambda node: (arr(node.frame())[:63].tobytes(),
                             node.levels_completed),
        max_states=800, max_depth=12,
    )
    verified = env.clone()
    run_actions(verified, route or [])
    print("MACRO", route)
    print("VERIFY", edge_row(verified.frame()), verified.levels_completed,
          verified.terminal())
    repeated = env.clone()
    for step in range(30):
        frame = repeated.frame()
        ys = [y for y in ROW_ANCHORS if int(frame[y][3]) == 8]
        if not ys or repeated.terminal():
            print("REPEAT_STOP", step, edge_row(frame),
                  repeated.levels_completed, repeated.terminal(),
                  avatar_cell(frame))
            break
        repeated.step(CLICK, 3, ys[0])
        print("REPEAT", step + 1, edge_row(repeated.frame()),
              repeated.levels_completed, repeated.terminal(),
              avatar_cell(repeated.frame()),
              grid(repeated.frame()) if step % 2 else "")


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
