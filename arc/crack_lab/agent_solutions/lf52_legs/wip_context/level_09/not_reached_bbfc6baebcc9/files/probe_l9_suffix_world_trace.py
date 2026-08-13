"""Trace the verified level-9 suffix in observed scrolling coordinates."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, connected_components, safe_step


OPENING = (
    ((42, 18), (42, 30)), ((48, 24), (36, 24)),
    ((42, 24), (30, 24)), ((36, 24), (24, 24)),
    ((30, 24), (18, 24)), ((18, 24), (18, 36)),
    ((18, 36), (30, 36)), ((24, 36), (36, 36)),
    ((30, 36), (42, 36)), ((36, 36), (48, 36)),
    ((48, 42), (48, 30)), ((48, 30), (36, 30)),
    ((48, 36), (36, 36)), ((36, 30), (36, 42)),
)

PHASES = (
    ("reveal", (4,) * 9, ()),
    ("shuttle_left", (), (
        ((18, 58), (18, 46)), ((18, 52), (18, 40)),
        ((18, 46), (18, 34)), ((18, 40), (18, 28)),
        ((18, 34), (18, 22)), ((18, 28), (18, 16)),
        ((18, 22), (18, 10)), ((18, 16), (18, 4)),
    )),
    ("turn_left", (3,), (
        ((18, 16), (18, 4)), ((12, 4), (24, 4)),
    )),
    ("middle", (), (
        ((24, 4), (24, 16)), ((18, 4), (18, 16)),
        ((24, 16), (12, 16)), ((12, 16), (12, 28)),
        ((12, 28), (12, 40)), ((18, 10), (18, 22)),
        ((18, 16), (18, 28)), ((18, 22), (18, 34)),
        ((18, 28), (18, 40)), ((12, 40), (24, 40)),
        ((24, 40), (24, 52)),
    )),
    ("turn_right", (4,), (
        ((24, 46), (24, 58)), ((18, 28), (18, 40)),
        ((18, 34), (18, 46)), ((18, 40), (18, 52)),
        ((18, 46), (18, 58)), ((18, 58), (30, 58)),
        ((24, 58), (36, 58)),
    )),
    ("meet", (4,) * 5, (((36, 22), (36, 34)),)),
)


def normalize(action):
    return tuple(action) if isinstance(action, list) else action


def frame_key(node):
    return arr(node.frame())[1:, :].tobytes()


def logical(frame):
    pieces = []
    for blob in connected_components(frame, colors=(9, 12, 14, 15)):
        if blob.color in (9, 12, 14) and blob.size == (4, 4):
            if blob.color != 9 or blob.area == 12:
                pieces.append((blob.color, blob.top_left))
        elif blob.color == 15 and blob.size == (4, 4):
            pieces.append((15, (blob.bbox[0] + 1, blob.bbox[1])))
    return tuple(sorted(pieces))


def click_move(node, source, destination):
    safe_step(node, (6, source[1] + 1, source[0] + 1))
    safe_step(node, (6, destination[1] + 1, destination[0] + 1))


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = tuple(normalize(action)
                         for action in json.load(stream)["final_path"])
    for action in campaign:
        safe_step(env, action)
    node = env.clone()
    for source, destination in OPENING:
        click_move(node, source, destination)

    offset = 0
    print("trace_entry", logical(node.frame()), flush=True)
    for name, keys, moves in PHASES:
        changed_keys = []
        for action in keys:
            before = frame_key(node)
            safe_step(node, action)
            changed = frame_key(node) != before
            if changed:
                offset += 1 if action == 4 else -1 if action == 3 else 0
            changed_keys.append((action, changed, offset))
        traced = []
        for source, destination in moves:
            before = dict((position, color)
                          for color, position in logical(node.frame()))
            midpoint = ((source[0] + destination[0]) // 2,
                        (source[1] + destination[1]) // 2)
            world = tuple((row, col + 6 * offset)
                          for row, col in (source, midpoint, destination))
            colors = tuple(before.get(point) for point
                           in (source, midpoint, destination))
            click_move(node, source, destination)
            traced.append((colors, world))
        print("trace_phase", name, tuple(changed_keys), tuple(traced),
              int(node.levels_completed), logical(node.frame()), flush=True)


arena.run_program("lf52", probe)
