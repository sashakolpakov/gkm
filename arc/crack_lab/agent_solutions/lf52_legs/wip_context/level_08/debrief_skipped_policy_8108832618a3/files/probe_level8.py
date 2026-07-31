"""Compact clean-room probes for lf52 level 8."""

import json
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

from perception import arr, color_counts, connected_components, frame_delta


COLORS = (1, 5, 9, 10, 14)
LOAD_LEFT = [
    3, 3, 1, 1, 1, 1, 4,
    (6, 31, 49), (6, 31, 37),
    (6, 31, 43), (6, 31, 31),
    (6, 31, 37), (6, 31, 25),
    (6, 13, 25), (6, 25, 25),
    (6, 25, 25), (6, 37, 25),
    (6, 37, 25), (6, 49, 25),
    (6, 49, 25), (6, 49, 37),
    (6, 49, 37), (6, 37, 37),
    (6, 31, 25), (6, 31, 37),
    (6, 37, 37), (6, 25, 37),
    (6, 25, 37), (6, 13, 37),
]
UNLOAD_STAGE = LOAD_LEFT + [3, 2, 2, 2, 2, 1, 1, 1, 1, 4, 4, 4]
SHIFTED_BOARD = UNLOAD_STAGE + [
    (6, 13, 37), (6, 25, 37),
    (6, 25, 37), (6, 37, 37),
]
DEEPER = LOAD_LEFT + [3, 2, 2, 2, 2, 4, 4, 4, 2]
LOWER_BRIDGES_LEFT = [
    (6, 55, 43), (6, 43, 43),
    (6, 49, 43), (6, 37, 43),
    (6, 43, 43), (6, 31, 43),
    (6, 37, 43), (6, 25, 43),
]
TRANSFER_LOWER = (
    DEEPER + [3, 3, 3] + LOWER_BRIDGES_LEFT
    + [(6, 25, 37), (6, 25, 49)]
)
LOWER_BRIDGES_RIGHT = [
    (6, 25, 43), (6, 37, 43),
    (6, 31, 43), (6, 43, 43),
    (6, 37, 43), (6, 49, 43),
]
TRANSFER_UPPER = (
    TRANSFER_LOWER + [4, 4, 4, 4] + LOWER_BRIDGES_RIGHT
    + [(6, 49, 49), (6, 49, 37)]
)
UPPER_JUNCTION = TRANSFER_UPPER + [1]
FINAL_STAGE = UPPER_JUNCTION + [4]
CAPTURE_STAGE = FINAL_STAGE + [1]
LEVEL8_CANDIDATE = CAPTURE_STAGE + [(6, 55, 37), (6, 55, 25)]
LEVEL8_CANDIDATE_REVERSE = CAPTURE_STAGE + [(6, 55, 31), (6, 55, 43)]


def enter_level_8(env):
    with open("checkpoint.json") as f:
        checkpoint = json.load(f)
    for action in checkpoint["final_path"]:
        env.step(action)


def component_summary(frame):
    return {
        color: [(blob.bbox, blob.area) for blob in connected_components(frame, [color])]
        for color in COLORS
    }


def compact_grid(frame, cell=4):
    grid = arr(frame)
    symbols = {1: ".", 5: "5", 9: "9", 10: "A", 14: "E"}
    rows = []
    for r in range(0, grid.shape[0], cell):
        row = []
        for c in range(0, grid.shape[1], cell):
            block = grid[r:r + cell, c:c + cell]
            values, counts = np.unique(block, return_counts=True)
            value = int(values[counts.argmax()])
            row.append(symbols.get(value, "?"))
        rows.append("".join(row))
    return rows


def tile_summary(frame):
    grid = arr(frame)
    for r in range(18, 55, 6):
        cells = []
        for c in range(12, 55, 6):
            values, counts = np.unique(grid[r:r + 4, c:c + 4], return_counts=True)
            signature = "/".join(
                f"{int(value)}:{int(count)}" for value, count in zip(values, counts)
            )
            cells.append(f"({r},{c})={signature}")
        print("TILES", " ".join(cells))


def board_map(frame):
    grid = arr(frame)
    rows = []
    for r in range(0, 61, 6):
        row = []
        for c in range(0, 61, 6):
            block = grid[r:r + 4, c:c + 4]
            counts = {
                color: int(np.count_nonzero(block == color))
                for color in (1, 9, 12, 14, 15)
            }
            if counts[14] >= 8:
                char = "P"
            elif counts[12] >= 4:
                char = "C"
            elif counts[9] >= 8:
                char = "B"
            elif counts[15] >= 4:
                char = "F"
            elif counts[1] == 16:
                char = "."
            else:
                char = " "
            row.append(char)
        rows.append(f"{r:02d} {''.join(row)}")
    return rows


def map_probes(env):
    paths = {
        "initial": [],
        "loaded": LOAD_LEFT,
        "scroll2": LOAD_LEFT + [3, 2, 2],
        "scrolled": LOAD_LEFT + [3, 2, 2, 2, 2],
    }
    for name, path in paths.items():
        node = apply_path(env, path)
        print("MAP", name, {
            "pegs": peg_positions(node.frame()),
            "carriers": carrier_tiles(node.frame()),
        })
        for row in board_map(node.frame()):
            print(row)


def save_view_strip(env):
    from PIL import Image

    palette = np.array([
        (0, 0, 0), (235, 235, 235), (255, 70, 70), (70, 210, 90),
        (255, 220, 60), (150, 150, 150), (230, 70, 210), (255, 150, 50),
        (80, 210, 230), (120, 35, 95), (25, 45, 85), (60, 170, 210),
        (250, 205, 70), (100, 70, 180), (235, 75, 120), (55, 185, 145),
    ], dtype=np.uint8)
    paths = (
        [],
        LOAD_LEFT,
        LOAD_LEFT + [3, 2, 2, 2, 2],
        LOAD_LEFT + [3, 2, 2, 2, 2, 4, 4, 4, 4],
    )
    frames = [palette[arr(apply_path(env, path).frame())] for path in paths]
    divider = np.full((64, 2, 3), 255, dtype=np.uint8)
    strip = frames[0]
    for frame in frames[1:]:
        strip = np.concatenate((strip, divider, frame), axis=1)
    image = Image.fromarray(strip).resize((strip.shape[1] * 6, 64 * 6), Image.Resampling.NEAREST)
    image.save("probe_level8_views.png")
    print("SAVED probe_level8_views.png")


def branch_probes(env):
    scrolled = LOAD_LEFT + [3, 2, 2, 2, 2]
    new_bridges_left4 = [
        (6, 55, 49), (6, 43, 49),
        (6, 49, 49), (6, 37, 49),
        (6, 43, 49), (6, 31, 49),
        (6, 37, 49), (6, 25, 49),
    ]
    old_bridges_down5 = [
        (6, 31, 7), (6, 31, 19),
        (6, 31, 13), (6, 31, 25),
        (6, 31, 19), (6, 31, 31),
        (6, 31, 25), (6, 31, 37),
        (6, 31, 31), (6, 31, 43),
    ]
    bridge_loaded_bottom = (
        scrolled + new_bridges_left4 + old_bridges_down5
        + [4, (6, 31, 43), (6, 31, 55)]
    )
    paths = {
        "scroll2": LOAD_LEFT + [3, 2, 2],
        "scroll2_bottom_left1": LOAD_LEFT + [3, 2, 2, (6, 55, 61), (6, 43, 61)],
        "scroll2_old_down1": LOAD_LEFT + [3, 2, 2, (6, 31, 19), (6, 31, 31)],
        "scrolled": scrolled,
        "new_bridge_left1": scrolled + [(6, 55, 49), (6, 43, 49)],
        "new_bridges_left4": scrolled + new_bridges_left4,
        "bridge_loaded_bottom": bridge_loaded_bottom,
        "transferred_lower": TRANSFER_LOWER,
        "upper_junction": UPPER_JUNCTION,
        "final_stage": FINAL_STAGE,
    }
    for name, path in paths.items():
        node = apply_path(env, path)
        print("BRANCH", name, state_summary(node))
        base = arr(node.frame()).copy()
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            delta = frame_delta(base, child.frame())
            print("KEY_AT", name, action, {
                "changed": max(0, delta["count"] - 1),
                "bbox": delta["bbox"],
                "pegs": tuple(peg_positions(child.frame())),
                "carriers": carrier_tiles(child.frame()),
            })


def apply_path(env, path):
    clone = env.clone()
    for action in path:
        if isinstance(action, tuple):
            clone.step(*action)
        else:
            clone.step(action)
    return clone


def peg_positions(frame):
    return [
        blob.top_left
        for blob in connected_components(frame, [14])
        if blob.size == (4, 4)
    ]


def colored_pixels(frame, color):
    ys, xs = np.where(arr(frame) == color)
    return [(int(y), int(x)) for y, x in zip(ys, xs)]


def carrier_tiles(frame):
    grid = arr(frame)
    return tuple(
        (r, c, int(np.count_nonzero(grid[r:r + 4, c:c + 4] == 12)))
        for r in range(0, 61, 6)
        for c in range(0, 61, 6)
        if np.count_nonzero(grid[r:r + 4, c:c + 4] == 12)
    )


def movable_tiles(frame):
    grid = arr(frame)
    pieces = []
    for r in range(0, 61, 6):
        for c in range(0, 61, 6):
            block = grid[r:r + 4, c:c + 4]
            if np.count_nonzero(block == 14) >= 8:
                pieces.append(("peg", (r, c)))
            elif np.count_nonzero(block == 9) >= 8:
                pieces.append(("bridge", (r, c)))
    return tuple(pieces)


def legal_coordinate_moves(env):
    moves = []
    for kind, (r, c) in movable_tiles(env.frame()):
        selected = env.clone()
        selected.step(6, c + 1, r + 1)
        destinations = sorted({
            (y // 6 * 6, x // 6 * 6)
            for y, x in colored_pixels(selected.frame(), 2)
        })
        moves.extend((kind, (r, c), destination) for destination in destinations)
    return tuple(moves)


def abstract_geometry(frame):
    slots = {
        blob.top_left
        for blob in connected_components(frame, [1])
        if blob.size == (4, 4) and blob.area == 16
    }
    pieces = movable_tiles(frame)
    pegs = frozenset(position for kind, position in pieces if kind == "peg")
    bridges = frozenset(position for kind, position in pieces if kind == "bridge")
    carriers = frozenset((r, c) for r, c, _ in carrier_tiles(frame))
    fixed = set()
    grid = arr(frame)
    for r in range(0, 61, 6):
        for c in range(0, 61, 6):
            if np.count_nonzero(grid[r:r + 4, c:c + 4] == 15) >= 4:
                fixed.add((r, c))
    return frozenset(slots | pegs | bridges), pegs, bridges, carriers, frozenset(fixed)


def abstract_moves(state, destinations, fixed):
    pegs, bridges = state
    occupied = pegs | bridges
    for kind, source in (
        tuple(("peg", position) for position in sorted(pegs))
        + tuple(("bridge", position) for position in sorted(bridges))
    ):
        for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
            midpoint = (source[0] + dr, source[1] + dc)
            destination = (source[0] + 2 * dr, source[1] + 2 * dc)
            if (
                destination not in destinations
                or destination in occupied
                or midpoint not in occupied | fixed
            ):
                continue
            child_pegs = set(pegs)
            child_bridges = set(bridges)
            if kind == "peg":
                child_pegs.remove(source)
                child_pegs.add(destination)
                if midpoint in child_pegs:
                    child_pegs.remove(midpoint)
            else:
                child_bridges.remove(source)
                child_bridges.add(destination)
            yield (
                (frozenset(child_pegs), frozenset(child_bridges)),
                (source, destination),
            )


def abstract_solution(frame, require_carrier, max_states=500000):
    slots, pegs, bridges, carriers, fixed = abstract_geometry(frame)
    destinations = slots | carriers
    start = (pegs, bridges)
    queue = deque([(start, ())])
    seen = {start}
    while queue and len(seen) <= max_states:
        state, path = queue.popleft()
        state_pegs, _ = state
        if len(state_pegs) == 1 and (
            not require_carrier or next(iter(state_pegs)) in carriers
        ):
            return path, len(seen)
        for child, move in abstract_moves(state, destinations, fixed):
            if child not in seen:
                seen.add(child)
                queue.append((child, path + (move,)))
    return None, len(seen)


def abstract_first_peg_change(frame, max_states=500000):
    slots, pegs, bridges, carriers, fixed = abstract_geometry(frame)
    destinations = slots | carriers
    start = (pegs, bridges)
    queue = deque([(start, ())])
    seen = {start}
    while queue and len(seen) <= max_states:
        state, path = queue.popleft()
        state_pegs, _ = state
        if state_pegs != pegs:
            return path, state_pegs, len(seen)
        for child, move in abstract_moves(state, destinations, fixed):
            if child not in seen:
                seen.add(child)
                queue.append((child, path + (move,)))
    return None, pegs, len(seen)


def abstract_first_bridge_load(frame, max_states=500000):
    slots, pegs, bridges, carriers, fixed = abstract_geometry(frame)
    destinations = slots | carriers
    start = (pegs, bridges)
    queue = deque([(start, ())])
    seen = {start}
    while queue and len(seen) <= max_states:
        state, path = queue.popleft()
        _, state_bridges = state
        loaded = state_bridges & carriers
        if loaded:
            return path, loaded, len(seen)
        for child, move in abstract_moves(state, destinations, fixed):
            if child not in seen:
                seen.add(child)
                queue.append((child, path + (move,)))
    return None, frozenset(), len(seen)


def abstract_reachability(frame, max_states=500000):
    slots, pegs, bridges, carriers, fixed = abstract_geometry(frame)
    destinations = slots | carriers
    start = (pegs, bridges)
    queue = deque([start])
    seen = {start}
    peg_positions_seen = set(pegs)
    loaded_pegs = set(pegs & carriers)
    loaded_bridges = set(bridges & carriers)
    while queue and len(seen) <= max_states:
        state = queue.popleft()
        state_pegs, state_bridges = state
        peg_positions_seen.update(state_pegs)
        loaded_pegs.update(state_pegs & carriers)
        loaded_bridges.update(state_bridges & carriers)
        for child, _ in abstract_moves(state, destinations, fixed):
            if child not in seen:
                seen.add(child)
                queue.append(child)
    return {
        "states": len(seen),
        "peg_positions": tuple(sorted(peg_positions_seen)),
        "loaded_pegs": tuple(sorted(loaded_pegs)),
        "loaded_bridges": tuple(sorted(loaded_bridges)),
    }


def abstract_closure_paths(frame, max_states=500000):
    slots, pegs, bridges, carriers, fixed = abstract_geometry(frame)
    destinations = slots | carriers
    start = (pegs, bridges)
    queue = deque([(start, ())])
    seen = {start}
    out = []
    while queue and len(seen) <= max_states:
        state, path = queue.popleft()
        out.append((state, path))
        for child, move in abstract_moves(state, destinations, fixed):
            if child not in seen:
                seen.add(child)
                queue.append((child, path + (move,)))
    return out


def abstract_probe(env):
    stage = [3, 3, 1, 1, 1, 1, 4]
    node = apply_path(env, stage)
    print("ABSTRACT_GEOMETRY", abstract_geometry(node.frame()))
    for require_carrier in (False, True):
        solution, states = abstract_solution(node.frame(), require_carrier)
        print("ABSTRACT_SOLUTION", {
            "require_carrier": require_carrier,
            "states": states,
            "moves": solution,
        })
        if solution is not None:
            path = list(stage)
            for source, destination in solution:
                path.extend([
                    (6, source[1] + 1, source[0] + 1),
                    (6, destination[1] + 1, destination[0] + 1),
                ])
            replayed = apply_path(env, path)
            print("ABSTRACT_REPLAY", {
                "require_carrier": require_carrier,
                "path_len": len(path),
                "level": int(replayed.levels_completed),
                "state": state_summary(replayed) if replayed.levels_completed == 7 else None,
            })


def shifted_abstract_probe(env):
    node = apply_path(env, SHIFTED_BOARD)
    print("SHIFTED_ABSTRACT", {
        "state": state_summary(node),
        "reachability": abstract_reachability(node.frame()),
        "capture": abstract_solution(node.frame(), False),
        "bridge_load": abstract_first_bridge_load(node.frame()),
    })


def move_path_to_actions(moves):
    actions = []
    for source, destination in moves:
        actions.extend([
            (6, source[1] + 1, source[0] + 1),
            (6, destination[1] + 1, destination[0] + 1),
        ])
    return actions


def short_key_coordinate_frontier(env, max_key_depth=3, max_frames=200):
    start = apply_path(env, DEEPER)
    closure = abstract_closure_paths(start.frame())
    seen_frames = set()
    findings = []
    for _, coordinate_path in closure:
        if len(seen_frames) >= max_frames:
            break
        node = apply_path(start, move_path_to_actions(coordinate_path))
        queue = deque([(node, ())])
        while queue and len(seen_frames) < max_frames:
            parent, key_path = queue.popleft()
            if len(key_path) >= max_key_depth:
                continue
            for key_action in (1, 2, 3, 4):
                child = parent.clone()
                child.step(key_action)
                child_key = arr(child.frame())[1:].tobytes()
                if child_key in seen_frames:
                    continue
                seen_frames.add(child_key)
                child_key_path = key_path + (key_action,)
                if child.levels_completed > 7:
                    findings.append((coordinate_path, child_key_path, (), "reward", child))
                    continue
                capture, _ = abstract_solution(child.frame(), False)
                peg_change, _, _ = abstract_first_peg_change(child.frame())
                bridge_load, _, _ = abstract_first_bridge_load(child.frame())
                if capture is not None:
                    findings.append((coordinate_path, child_key_path, capture, "capture", child))
                elif peg_change is not None:
                    findings.append((coordinate_path, child_key_path, peg_change, "peg_change", child))
                elif bridge_load is not None:
                    findings.append((coordinate_path, child_key_path, bridge_load, "bridge_load", child))
                else:
                    queue.append((child, child_key_path))
    print("SHORT_KEY_FRONTIER", {
        "coordinate_states": len(closure),
        "key_frames": len(seen_frames),
        "findings": len(findings),
    })
    for coordinate_path, key_path, suffix, kind, child in findings[:30]:
        print("SHORT_KEY_FINDING", {
            "kind": kind,
            "coordinate_prefix": coordinate_path,
            "keys": key_path,
            "coordinate_suffix": suffix,
            "state": state_summary(child),
        })


def state_summary(env):
    return {
        "level": int(env.levels_completed),
        "pieces": movable_tiles(env.frame()),
        "carriers": carrier_tiles(env.frame()),
        "moves": legal_coordinate_moves(env),
    }


def context_probes(env):
    deeper = LOAD_LEFT + [3, 2, 2, 2, 2, 4, 4, 4, 2]
    deeper_left1 = deeper + [(6, 55, 43), (6, 43, 43)]
    deeper_left2 = deeper_left1 + [(6, 49, 43), (6, 37, 43)]
    deeper_left3 = deeper_left2 + [(6, 43, 43), (6, 31, 43)]
    deeper_left4 = deeper_left3 + [(6, 37, 43), (6, 25, 43)]
    aligned_deeper = (
        LOAD_LEFT + [3, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 2]
    )
    aligned_left4 = aligned_deeper + [
        (6, 55, 43), (6, 43, 43),
        (6, 49, 43), (6, 37, 43),
        (6, 43, 43), (6, 31, 43),
        (6, 37, 43), (6, 25, 43),
    ]
    probes = {
        "right_staged": [1, 4],
        "right_loaded": [1, 4, (6, 55, 49), (6, 55, 61)],
        "right_loaded_up1": [1, 4, (6, 55, 49), (6, 55, 61), 1],
        "right_loaded_up2": [1, 4, (6, 55, 49), (6, 55, 61), 1, 1],
        "left_staged": [3, 3, 1, 1, 1, 1, 4],
        "left_peg_advanced": [(6, 13, 25), (6, 25, 25)],
        "peg_over_movable": [
            (6, 31, 49), (6, 31, 37),
            (6, 31, 43), (6, 31, 31),
            (6, 31, 37), (6, 31, 25),
            (6, 13, 25), (6, 25, 25),
            (6, 25, 25), (6, 37, 25),
        ],
        "left_loaded": LOAD_LEFT,
        "left_loaded_out": LOAD_LEFT + [3, 2, 2, 2, 2],
        "unload_stage": UNLOAD_STAGE,
        "unloaded_shifted": SHIFTED_BOARD,
        "deeper1": deeper,
        "deeper_left1_down": deeper_left1 + [2],
        "deeper_left2_down": deeper_left2 + [2],
        "deeper_left3_down": deeper_left3 + [2],
        "lower_unload_stage": deeper_left4,
        "lower_unloaded": deeper_left4 + [(6, 25, 37), (6, 25, 49)],
        "aligned_lower_stage": aligned_left4,
        "aligned_lower_transfer": aligned_left4 + [(6, 25, 37), (6, 25, 49)],
        "verified_lower_transfer": TRANSFER_LOWER,
        "verified_upper_transfer": TRANSFER_UPPER,
        "final_stage": FINAL_STAGE,
        "capture_stage": CAPTURE_STAGE,
        "final_capture": LEVEL8_CANDIDATE,
        "final_capture_reverse": LEVEL8_CANDIDATE_REVERSE,
    }
    for name, path in probes.items():
        clone = apply_path(env, path)
        print("CONTEXT", name, path, state_summary(clone))


def key_search_after_load(env, max_states=500, max_depth=20):
    start = apply_path(env, LOAD_LEFT)
    frame_key = lambda node: arr(node.frame())[1:].tobytes()
    queue = deque([()])
    seen = {frame_key(start)}
    peg_examples = {}
    deepest = 0
    while queue and len(seen) <= max_states:
        path = queue.popleft()
        node = apply_path(start, path)
        deepest = max(deepest, len(path))
        pegs = tuple(peg_positions(node.frame()))
        peg_examples.setdefault(pegs, (path, arr(node.frame()).copy()))
        if len(path) >= max_depth:
            continue
        for action in (1, 2, 3, 4):
            child_path = path + (action,)
            child = apply_path(start, child_path)
            child_key = frame_key(child)
            if child_key not in seen:
                seen.add(child_key)
                queue.append(child_path)
    ranked = []
    for pegs, (path, frame) in peg_examples.items():
        distance = (
            abs(pegs[0][0] - pegs[1][0]) + abs(pegs[0][1] - pegs[1][1])
            if len(pegs) == 2 else 999
        )
        ranked.append((distance, len(path), pegs, path, frame))
    print("LOADED_KEY_SEARCH", {
        "states": len(seen),
        "deepest": deepest,
        "peg_configurations": len(peg_examples),
    })
    for distance, _, pegs, path, frame in sorted(ranked)[:30]:
        node = apply_path(start, path)
        print("PEG_CONFIG", {
            "distance": distance,
            "pegs": pegs,
            "key_path": path,
            "carriers": carrier_tiles(node.frame()),
            "moves": legal_coordinate_moves(node),
        })


def linear_carrier_probe(env):
    node = apply_path(env, LOAD_LEFT)
    node.step(3)
    for count in range(25):
        print("LINEAR_DOWN", count, {
            "level": int(node.levels_completed),
            "pegs": tuple(peg_positions(node.frame())),
            "pieces": movable_tiles(node.frame()),
            "carriers": carrier_tiles(node.frame()),
        })
        node.step(2)


def linear_right_probe(env):
    node = apply_path(env, LOAD_LEFT + [3, 2, 2, 2, 2])
    for count in range(16):
        print("LINEAR_RIGHT", count, {
            "level": int(node.levels_completed),
            "pegs": tuple(peg_positions(node.frame())),
            "pieces": movable_tiles(node.frame()),
            "carriers": carrier_tiles(node.frame()),
        })
        node.step(4)


def linear_deeper_probe(env):
    node = apply_path(env, LOAD_LEFT + [3, 2, 2, 2, 2, 4, 4, 4])
    for count in range(14):
        print("LINEAR_DEEPER", count, {
            "level": int(node.levels_completed),
            "pegs": tuple(peg_positions(node.frame())),
            "pieces": movable_tiles(node.frame()),
            "carriers": carrier_tiles(node.frame()),
        })
        node.step(2)


def linear_lower_probe(env):
    node = apply_path(env, TRANSFER_LOWER)
    for count in range(10):
        print("LINEAR_LOWER", count, {
            "level": int(node.levels_completed),
            "pegs": tuple(peg_positions(node.frame())),
            "pieces": movable_tiles(node.frame()),
            "carriers": carrier_tiles(node.frame()),
        })
        node.step(4)


def linear_upper_probe(env):
    node = apply_path(env, TRANSFER_UPPER)
    for count in range(14):
        print("LINEAR_UPPER", count, {
            "level": int(node.levels_completed),
            "pegs": tuple(peg_positions(node.frame())),
            "pieces": movable_tiles(node.frame()),
            "carriers": carrier_tiles(node.frame()),
        })
        node.step(1)


def key_state_search(env, max_states=1000, max_depth=60):
    start = env.clone()
    key = lambda node: arr(node.frame())[1:].tobytes()
    queue = deque([(start, ())])
    seen = {key(start)}
    signatures = {}
    deepest = 0
    while queue and len(seen) <= max_states:
        node, path = queue.popleft()
        deepest = max(deepest, len(path))
        signature = carrier_tiles(node.frame())
        signatures.setdefault(signature, path)
        if len(path) >= max_depth:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            child_key = key(child)
            if child_key not in seen:
                seen.add(child_key)
                queue.append((child, path + (action,)))
    print("KEY_SEARCH", {
        "states": len(seen),
        "deepest": deepest,
        "carrier_signatures": len(signatures),
    })
    for signature, path in sorted(
        signatures.items(), key=lambda item: (len(item[1]), item[1])
    ):
        print("CARRIER", signature, path)


def wrapped_key_graph(env, max_states=80, max_depth=10):
    prefix = LOAD_LEFT + [3, 2, 2, 2, 2]
    start = apply_path(env, prefix)
    key = lambda node: arr(node.frame())[1:].tobytes()
    queue = deque([(start, ())])
    seen = {key(start)}
    examples = {}
    while queue and len(seen) <= max_states:
        node, path = queue.popleft()
        signature = (
            tuple(peg_positions(node.frame())),
            carrier_tiles(node.frame()),
        )
        examples.setdefault(signature, (path, arr(node.frame()).copy()))
        if len(path) >= max_depth:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            child_key = key(child)
            if child_key not in seen:
                seen.add(child_key)
                queue.append((child, path + (action,)))
    print("WRAPPED_KEYS", {"states": len(seen), "examples": len(examples)})
    interesting = 0
    for signature, (path, frame) in sorted(
        examples.items(), key=lambda item: (len(item[1][0]), item[1][0])
    ):
        slots, pegs, bridges, carriers, fixed = abstract_geometry(frame)
        coordinate_moves = [
            move
            for _, move in abstract_moves((pegs, bridges), slots | carriers, fixed)
        ]
        peg_moves = [move for move in coordinate_moves if move[0] in pegs]
        solution, closure_states = abstract_solution(frame, False)
        peg_change, changed_pegs, _ = abstract_first_peg_change(frame)
        bridge_load, loaded_bridges, _ = abstract_first_bridge_load(frame)
        if peg_change is not None or bridge_load is not None or solution is not None:
            interesting += 1
            print("ALIGN_INTERESTING", {
                "key_path": path,
                "signature": signature,
                "peg_moves": peg_moves,
                "peg_change": peg_change,
                "changed_pegs": changed_pegs,
                "bridge_load": bridge_load,
                "loaded_bridges": loaded_bridges,
                "capture": solution,
                "closure_states": closure_states,
            })
    print("WRAPPED_INTERESTING", interesting)


def shifted_key_graph(env, max_states=80, max_depth=10):
    start = apply_path(env, SHIFTED_BOARD)
    key = lambda node: arr(node.frame())[1:].tobytes()
    queue = deque([(start, ())])
    seen = {key(start)}
    examples = {}
    while queue and len(seen) <= max_states:
        node, path = queue.popleft()
        signature = (
            tuple(peg_positions(node.frame())),
            carrier_tiles(node.frame()),
        )
        examples.setdefault(signature, (path, arr(node.frame()).copy()))
        if len(path) >= max_depth:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            child_key = key(child)
            if child_key not in seen:
                seen.add(child_key)
                queue.append((child, path + (action,)))
    print("SHIFTED_KEYS", {"states": len(seen), "examples": len(examples)})
    interesting = 0
    for signature, (path, frame) in sorted(
        examples.items(), key=lambda item: (len(item[1][0]), item[1][0])
    ):
        reachability = abstract_reachability(frame)
        novel_loaded = tuple(
            position
            for position in reachability["loaded_pegs"]
            if position != (36, 12)
        )
        capture, _ = abstract_solution(frame, False)
        bridge_load, loaded_bridges, _ = abstract_first_bridge_load(frame)
        if novel_loaded or capture is not None or bridge_load is not None:
            interesting += 1
            print("SHIFTED_INTERESTING", {
                "key_path": path,
                "signature": signature,
                "novel_loaded_pegs": novel_loaded,
                "bridge_load": bridge_load,
                "loaded_bridges": loaded_bridges,
                "capture": capture,
                "closure_states": reachability["states"],
            })
    print("SHIFTED_INTERESTING_COUNT", interesting)


def selection_probes(env):
    sources = {
        "peg_left": (6, 13, 25),
        "peg_right": (6, 55, 49),
        "color9_upper": (6, 31, 43),
        "color9_lower": (6, 31, 49),
        "fixed_bridge": (6, 19, 25),
        "empty_slot": (6, 25, 19),
        "carrier_bottom": (6, 19, 61),
    }
    for name, action in sources.items():
        clone = apply_path(env, [action])
        print("SELECT", name, {
            "action": action,
            "color2": colored_pixels(clone.frame(), 2),
            "color3_bbox": (
                min(colored_pixels(clone.frame(), 3)),
                max(colored_pixels(clone.frame(), 3)),
            ) if colored_pixels(clone.frame(), 3) else None,
        })


def coordinate_probes(env):
    base = arr(env.frame()).copy()
    probes = {
        "select_left": [(6, 13, 25)],
        "left_over_bridge": [(6, 13, 25), (6, 25, 25)],
        "left_invalid_down": [(6, 13, 25), (6, 13, 37)],
        "right_invalid_left": [(6, 55, 49), (6, 43, 49)],
        "right_invalid_up": [(6, 55, 49), (6, 55, 37)],
        "color9_jump": [(6, 31, 49), (6, 31, 37)],
    }
    for name, path in probes.items():
        clone = apply_path(env, path)
        print("COORD", name, {
            "path": path,
            "level": int(clone.levels_completed),
            "pegs": peg_positions(clone.frame()),
            "color9_tiles": [
                (r, c)
                for r in range(0, 61, 6)
                for c in range(0, 61, 6)
                if int(np.count_nonzero(arr(clone.frame())[r:r + 4, c:c + 4] == 9)) >= 8
            ],
            "delta": frame_delta(base, clone.frame()),
        })


def probe(env):
    enter_level_8(env)
    base = arr(env.frame()).copy()
    if len(sys.argv) > 1 and sys.argv[1] == "tiles":
        tile_summary(base)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "maps":
        map_probes(env)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "image":
        save_view_strip(env)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "branches":
        branch_probes(env)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "coords":
        coordinate_probes(env)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "select":
        selection_probes(env)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "keys":
        key_state_search(env)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "contexts":
        context_probes(env)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "loaded_keys":
        key_search_after_load(env)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "linear":
        linear_carrier_probe(env)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "right":
        linear_right_probe(env)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "deeper":
        linear_deeper_probe(env)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "lower":
        linear_lower_probe(env)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "upper":
        linear_upper_probe(env)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "wrapped_keys":
        wrapped_key_graph(env)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "shifted_keys":
        shifted_key_graph(env)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "one_key":
        short_key_coordinate_frontier(env)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "abstract":
        abstract_probe(env)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "shifted":
        shifted_abstract_probe(env)
        return
    print("BASE", {
        "level": int(env.levels_completed),
        "terminal": bool(env.terminal()),
        "actions": list(env.actions),
        "colors": color_counts(base),
    })
    for row in compact_grid(base):
        print("GRID", row)
    print("COMPONENTS", component_summary(base))
    for action in env.actions:
        clone = env.clone()
        clone.step(action)
        print("ACTION", int(action), {
            "level": int(clone.levels_completed),
            "delta": frame_delta(base, clone.frame()),
            "colors": color_counts(clone.frame()),
            "components": component_summary(clone.frame()),
        })


if __name__ == "__main__":
    arena.run_program("lf52", probe)
