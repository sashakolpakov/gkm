"""Probe the visible cross-region peg-to-carrier transfer on level 9."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, connected_components, frame_delta, safe_step


FIRST_RELAY = (
    ((42, 18), (42, 30)), ((48, 24), (36, 24)),
    ((42, 24), (30, 24)), ((36, 24), (24, 24)),
    ((30, 24), (18, 24)), ((18, 24), (18, 36)),
    ((18, 36), (30, 36)), ((24, 36), (36, 36)),
    ((30, 36), (42, 36)), ((36, 36), (48, 36)),
    ((48, 42), (48, 30)), ((48, 30), (36, 30)),
    ((48, 36), (36, 36)), ((36, 30), (36, 42)),
)


def play_move(env, move):
    source, destination = move
    safe_step(env, (6, source[1] + 1, source[0] + 1))
    safe_step(env, (6, destination[1] + 1, destination[0] + 1))


def puzzle_delta(before, after):
    left = arr(before).copy()
    right = arr(after).copy()
    right[0] = left[0]
    delta = frame_delta(left, right)
    return delta["count"], delta["bbox"]


def pieces(frame):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(3, 9, 11, 12, 14, 15))
        if blob.color != 9 or blob.area >= 12
    )


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    for move in FIRST_RELAY:
        play_move(env, move)

    attached = env.clone()
    before_attach = attached.frame()
    safe_step(attached, (6, 17, 37))
    safe_step(attached, (6, 29, 37))
    print("implicit_bridge_attach",
          puzzle_delta(before_attach, attached.frame()),
          int(attached.levels_completed), pieces(attached.frame()))

    loaded = env.clone()
    for offset in range(9):
        peg_col = 52 - 6 * offset
        for order in ("incoming_to_carrier", "carrier_to_incoming"):
            child = loaded.clone()
            before = child.frame()
            incoming = (6, peg_col + 1, 13)
            carrier = (6, 23, 37)
            actions = (
                (incoming, carrier)
                if order == "incoming_to_carrier"
                else (carrier, incoming)
            )
            for action in actions:
                safe_step(child, action)
            print("loaded_cross", offset, peg_col, order,
                  puzzle_delta(before, child.frame()),
                  int(child.levels_completed), pieces(child.frame()))
        safe_step(loaded, 4)

    vertical = env.clone()
    for _ in range(6):
        safe_step(vertical, 4)
    before_vertical = vertical.frame()
    safe_step(vertical, (6, 23, 37))
    safe_step(vertical, (6, 23, 13))
    print("vertical_unload", puzzle_delta(before_vertical, vertical.frame()),
          int(vertical.levels_completed), pieces(vertical.frame()))
    vertical_pairs = (
        ((12, 16), (12, 28)),
        ((12, 22), (12, 10)),
        ((12, 16), (36, 22)),
        ((12, 22), (36, 22)),
    )
    for source, destination in vertical_pairs:
        child = vertical.clone()
        before = child.frame()
        safe_step(child, (6, source[1] + 1, source[0] + 1))
        safe_step(child, (6, destination[1] + 1, destination[0] + 1))
        print("vertical_pair", source, destination,
              puzzle_delta(before, child.frame()),
              int(child.levels_completed), pieces(child.frame()))

    # Unload the carried survivor, leaving an empty carrier at (36, 22).
    safe_step(env, (6, 23, 37))
    safe_step(env, (6, 11, 37))
    adjacent_bridge = env.clone()
    before_adjacent = adjacent_bridge.frame()
    safe_step(adjacent_bridge, (6, 17, 37))
    safe_step(adjacent_bridge, (6, 23, 37))
    print("adjacent_bridge_load",
          puzzle_delta(before_adjacent, adjacent_bridge.frame()),
          int(adjacent_bridge.levels_completed), pieces(adjacent_bridge.frame()))
    for steps in range(7):
        node = env.clone()
        for _ in range(steps):
            safe_step(node, 4)
        carrier_col = 22 + 6 * steps
        for order in ("peg_first", "carrier_first"):
            child = node.clone()
            before = child.frame()
            peg_click = (6, 53, 13)
            carrier_click = (6, carrier_col + 1, 37)
            actions = (
                (peg_click, carrier_click)
                if order == "peg_first"
                else (carrier_click, peg_click)
            )
            for action in actions:
                safe_step(child, action)
            print("cross", steps, carrier_col, order,
                  puzzle_delta(before, child.frame()),
                  int(child.levels_completed), pieces(child.frame()))
    for steps in (7, 8):
        node = env.clone()
        for _ in range(steps):
            safe_step(node, 4)
        for destination in ((6, 63, 37), (6, 62, 37), (6, 63, 38)):
            child = node.clone()
            before = child.frame()
            safe_step(child, (6, 53, 13))
            safe_step(child, destination)
            print("edge_cross", steps, destination,
                  puzzle_delta(before, child.frame()),
                  int(child.levels_completed), pieces(child.frame()))

    aligned = env.clone()
    for _ in range(6):
        safe_step(aligned, 4)
    local_destinations = (
        (12, 58), (18, 52), (18, 58), (24, 52), (24, 58),
    )
    for destination in local_destinations:
        child = aligned.clone()
        before = child.frame()
        safe_step(child, (6, 53, 13))
        safe_step(child, (6, destination[1] + 1, destination[0] + 1))
        print("aligned_local", destination,
              puzzle_delta(before, child.frame()),
              int(child.levels_completed), pieces(child.frame()))
    staged = aligned.clone()
    for source, destination in (
        ((12, 52), (12, 58)),
        ((12, 58), (36, 58)),
    ):
        safe_step(staged, (6, source[1] + 1, source[0] + 1))
        safe_step(staged, (6, destination[1] + 1, destination[0] + 1))
    print("aligned_two_stage", int(staged.levels_completed), pieces(staged.frame()))


arena.run_program("lf52", probe)
