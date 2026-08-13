"""Verify one abstractly planned level-9 route on a pristine clone."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _bridge_carrier_moves, _bridge_carrier_state, _lattice_step
from perception import arr, color_counts, connected_components, safe_step


MOVES = (
    ((42, 18), (42, 30)),
    ((48, 24), (36, 24)),
    ((42, 24), (30, 24)),
    ((36, 24), (24, 24)),
    ((30, 24), (18, 24)),
    ((18, 24), (18, 36)),
    ((18, 36), (30, 36)),
    ((24, 36), (36, 36)),
    ((30, 36), (42, 36)),
    ((36, 36), (48, 36)),
    ((48, 42), (48, 30)),
    ((48, 30), (36, 30)),
    ((48, 36), (36, 36)),
    ((36, 30), (36, 42)),
)

WORLD_MOVES = (
    ((18, 112), (18, 100)),
    ((18, 106), (18, 94)),
    ((18, 100), (18, 88)),
    ((18, 94), (18, 82)),
    ((18, 88), (18, 76)),
    ((18, 82), (18, 70)),
    ((18, 76), (18, 64)),
    ((18, 70), (18, 58)),
    ((18, 64), (18, 52)),
    ((12, 52), (24, 52)),
    ((24, 52), (24, 64)),
    ((18, 52), (18, 64)),
    ((24, 64), (12, 64)),
    ((12, 64), (12, 76)),
    ((12, 76), (12, 88)),
    ((18, 58), (18, 70)),
    ((18, 64), (18, 76)),
    ((18, 70), (18, 82)),
    ((18, 76), (18, 88)),
    ((12, 88), (24, 88)),
    ((24, 88), (24, 100)),
    ((24, 100), (24, 112)),
    ((18, 82), (18, 94)),
    ((18, 88), (18, 100)),
    ((18, 94), (18, 106)),
    ((18, 100), (18, 112)),
    ((18, 112), (30, 112)),
    ((24, 112), (36, 112)),
    ((36, 106), (36, 118)),
)


def peg_pixels(frame):
    return int((arr(frame) == 14).sum())


def ascii_map(frame):
    glyph = {0: " ", 1: ".", 2: "d", 3: "s", 5: "#", 7: "v",
             9: "X", 10: " ", 11: "B", 12: "C", 14: "O", 15: "T"}
    data = arr(frame)
    return "\n".join(
        "".join(glyph[int(value)] for value in data[row, :])
        for row in range(6, 60)
    )


def compact_components(frame):
    return tuple(
        (blob.color, blob.top_left, blob.size, blob.area)
        for blob in connected_components(frame, colors=(7, 9, 11, 12, 14, 15))
        if blob.color in (11, 12, 14, 15)
        or (blob.size == (4, 4) and blob.area >= 8)
    )


def selectable_summary(node):
    base = arr(node.frame())
    source_candidates = tuple(
        blob.top_left
        for blob in connected_components(node.frame(), colors=(7, 9, 14, 15))
        if blob.size == (4, 4) and blob.area >= 8
    )
    out = []
    for row, col in source_candidates:
        child = node.clone()
        safe_step(child, (6, col + 1, row + 1))
        after = arr(child.frame())
        if not (base[1:, :] != after[1:, :]).any():
            continue
        destinations = []
        for blob in connected_components(after, colors=(2,)):
            destinations.append(blob.bbox)
        out.append(((row, col), int(base[row + 1, col + 1]),
                    tuple(destinations)))
    return tuple(out)


def observe(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)

    node = env.clone()
    print("START", {"level": node.levels_completed,
                    "peg_pixels": peg_pixels(node.frame())})
    for index, (source, destination) in enumerate(MOVES, 1):
        safe_step(node, (6, source[1] + 1, source[0] + 1))
        safe_step(node, (6, destination[1] + 1, destination[0] + 1))
        print("MACRO", index, source, destination,
              {"level": node.levels_completed,
               "peg_pixels": peg_pixels(node.frame())})

    print("LOADED_COLORS", color_counts(node.frame()))
    for blob in connected_components(node.frame(), colors=(11, 12, 14)):
        print("LOADED_COMP", blob)
    print("LOADED_MAP\n" + ascii_map(node.frame()))

    base = arr(node.frame())
    print("LOADED_KEYS")
    for action in (1, 2, 3, 4, 7):
        child = node.clone()
        safe_step(child, action)
        delta = base[1:, :] != arr(child.frame())[1:, :]
        ys, xs = delta.nonzero()
        print(action, {"changed": int(delta.sum()),
                       "bbox": None if not len(ys) else
                       (int(ys.min() + 1), int(xs.min()),
                        int(ys.max() + 1), int(xs.max())),
                       "colors": color_counts(child.frame()),
                       "level": child.levels_completed})

    source_candidates = tuple(
        blob.top_left
        for blob in connected_components(node.frame(), colors=(7, 9, 14, 15))
        if blob.size == (4, 4) and blob.area >= 8
    )
    print("LOADED_SELECTABLES")
    for row, col in source_candidates:
        child = node.clone()
        safe_step(child, (6, col + 1, row + 1))
        after = arr(child.frame())
        delta = base[1:, :] != after[1:, :]
        if not delta.any():
            continue
        color2 = after == 2
        ys, xs = color2.nonzero()
        print((row, col), int(base[row + 1, col + 1]),
              {"dest_bbox": None if not len(ys) else
               (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())),
               "dest_pixels": int(color2.sum())})

    print("RIGHT_CONTEXTS")
    world = {name: set() for name in ("slots", "pegs", "fixed", "movable")}

    def absorb_world(candidate, offset):
        state = _bridge_carrier_state(candidate.frame())
        world["slots"].update((row, col + 6 * offset) for row, col in state[0])
        world["pegs"].update(
            (row, col + 6 * offset) for row, col in state[1]
            if (row, col) != (36, 22)
        )
        world["fixed"].update((row, col + 6 * offset) for row, col in state[3])
        world["movable"].update(
            (blob.top_left[0], blob.top_left[1] + 6 * offset)
            for blob in connected_components(candidate.frame(), colors=(9,))
            if blob.size == (4, 4) and blob.area == 12
        )

    absorb_world(node, 0)
    for count in range(1, 19):
        child = node.clone()
        for _ in range(count):
            safe_step(child, 4)
        base_child = arr(child.frame())
        available_keys = []
        for action in (1, 2, 3, 4):
            key_child = child.clone()
            safe_step(key_child, action)
            if (base_child[1:, :] != arr(key_child.frame())[1:, :]).any():
                available_keys.append(action)
        print("RIGHT", count, {"level": child.levels_completed,
                               "components": compact_components(child.frame()),
                               "keys": tuple(available_keys),
                               "persistent_moves": _bridge_carrier_moves(child.frame()),
                               "moves": selectable_summary(child)})
        if count in (5, 6, 14):
            bridge_state = _bridge_carrier_state(child.frame())
            print("BRIDGE_STATE", count, {
                "step": _lattice_step(bridge_state[0] | bridge_state[2]),
                "slots": tuple(sorted(bridge_state[0])),
                "pegs": tuple(sorted(bridge_state[1])),
                "carriers": tuple(sorted(bridge_state[2])),
                "bridges": tuple(sorted(bridge_state[3])),
            })
        if count <= 14:
            absorb_world(child, count)
    print("WORLD_GEOMETRY")
    for name, positions in world.items():
        by_row = {}
        for row, col in positions:
            by_row.setdefault(row, []).append(col)
        print(name, {row: tuple(sorted(set(cols)))
                     for row, cols in sorted(by_row.items())})

    print("FULL_CANDIDATE")
    stage2_actions = 0

    def keys(actions):
        nonlocal stage2_actions
        for action in actions:
            safe_step(node, action)
            stage2_actions += 1

    def world_move(index, offset):
        nonlocal stage2_actions
        source, destination = WORLD_MOVES[index]
        screen_source = (source[0], source[1] - 6 * offset)
        screen_destination = (destination[0], destination[1] - 6 * offset)
        safe_step(node, (6, screen_source[1] + 1, screen_source[0] + 1))
        safe_step(node, (6, screen_destination[1] + 1,
                         screen_destination[0] + 1))
        stage2_actions += 2
        print("WORLD_MACRO", index + 1, screen_source, screen_destination,
              {"stage2_actions": stage2_actions,
               "level": node.levels_completed,
               "terminal": node.terminal(),
               "peg_components": tuple(
                   blob.top_left for blob in connected_components(
                       node.frame(), colors=(14,)
                   ) if blob.area >= 8
               )})

    keys((4,) * 9)
    for index in range(0, 8):
        world_move(index, 9)
    keys((3,))
    for index in range(8, 21):
        world_move(index, 8)
    keys((4,))
    for index in range(21, 28):
        world_move(index, 9)
    keys((4,) * 5)
    world_move(28, 14)
    print("FULL_RESULT", {"level": node.levels_completed,
                          "terminal": node.terminal(),
                          "stage1_actions": 2 * len(MOVES),
                          "stage2_actions": stage2_actions})


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {"levels": levels, "moves": len(path), "error": str(error)})
