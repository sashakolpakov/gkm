"""Compact clean-room probes for dc22 level 6.

The live arena is advanced only through the validated checkpoint prefix.
Every level-6 experiment runs on a clone.
"""
import json
import sys
from collections import deque

import numpy as np
from PIL import Image

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import color_counts, connected_components, frame_delta


def enter_level_6(env):
    with open("checkpoint.json") as checkpoint_file:
        checkpoint = json.load(checkpoint_file)
    for action in checkpoint["final_path"]:
        env.step(action)


def compact_components(frame):
    counts = color_counts(frame)
    background = max(counts, key=counts.get)
    return [
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, min_area=2)
        if blob.color != background
    ]


def coarse_grid(frame, cell=4):
    pixels = np.asarray(frame)
    rows = []
    for row in range(0, pixels.shape[0], cell):
        tokens = []
        for col in range(0, pixels.shape[1], cell):
            values, counts = np.unique(
                pixels[row:row + cell, col:col + cell], return_counts=True
            )
            dominant = int(values[np.argmax(counts)])
            tokens.append(f"{dominant:x}")
        rows.append("".join(tokens))
    return rows


def gameplay_delta(before, after):
    first = np.asarray(before)
    second = np.asarray(after)
    changed = np.argwhere(first[:62] != second[:62])
    if not len(changed):
        return None
    transitions = {}
    for row, col in changed:
        pair = (int(first[row, col]), int(second[row, col]))
        transitions[pair] = transitions.get(pair, 0) + 1
    return (
        len(changed),
        (
            int(changed[:, 0].min()),
            int(changed[:, 1].min()),
            int(changed[:, 0].max()),
            int(changed[:, 1].max()),
        ),
        tuple(sorted(transitions.items())),
    )


def rare_click_points(frame):
    pixels = np.asarray(frame)
    rare = {1, 6, 7, 8, 9, 10, 11, 12, 14, 15}
    points = []
    for row in range(0, 62, 2):
        for col in range(0, 64, 2):
            if rare.intersection(int(value) for value in np.unique(
                    pixels[row:row + 2, col:col + 2])):
                points.append((col + 1, row + 1))
    return points


def avatar_position(frame):
    avatars = connected_components(frame, colors=(14,), min_area=4)
    return avatars[0].top_left if avatars else None


def central_ring(frame):
    candidates = [
        blob
        for blob in connected_components(frame, colors=(8,), min_area=20)
        if blob.bbox[1] < 40
    ]
    return candidates[0].bbox if candidates else None


def apply_symbols(env, symbols):
    points = {
        "A": (55, 7),
        "B": (51, 25),
        "C": (50, 48),
        "X": (50, 18),
        "Y": (53, 35),
        "Z": (49, 35),
        "W": (53, 39),
        "Q": (29, 34),
        "V": (35, 57),
        "N": (50, 32),
        "O": (46, 36),
    }
    actions = {"U": 1, "D": 2, "L": 3, "R": 4}
    for symbol in symbols:
        if symbol == "_":
            continue
        if symbol in points:
            env.step(6, *points[symbol])
        else:
            env.step(actions[symbol])


def movement_reachability(root, max_states=1000):
    names = {1: "U", 2: "D", 3: "L", 4: "R"}
    queue = deque([(root.clone(), "")])
    key = lambda node: np.asarray(node.frame())[:62].tobytes()
    seen = {key(root)}
    paths = {}
    wins = []
    while queue and len(seen) < max_states:
        node, path = queue.popleft()
        position = avatar_position(node.frame())
        if position is not None and position not in paths:
            paths[position] = path
        if node.levels_completed > 5:
            wins.append(path)
            continue
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            child_key = key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            queue.append((child, path + names[action]))
    return paths, wins, len(seen)


def controlled_search(root, goal_name, max_states=8000, max_depth=100):
    controls = {
        "U": (1,),
        "D": (2,),
        "L": (3,),
        "R": (4,),
        "A": (6, 55, 7),
        "B": (6, 51, 25),
        "C": (6, 50, 48),
        "Y": (6, 53, 35),
    }
    queue = deque([""])
    key = lambda node: (
        int(node.levels_completed),
        np.asarray(node.frame())[:62].tobytes(),
    )
    seen = {key(root)}
    best_row = 64
    best_col = 0
    progress = []
    while queue and len(seen) < max_states:
        path = queue.popleft()
        node = root.clone()
        apply_symbols(node, path)
        position = avatar_position(node.frame())
        if node.levels_completed > 5:
            return path, len(seen), progress
        if position is not None:
            row, col = position
            if row < best_row or col > best_col:
                best_row = min(best_row, row)
                best_col = max(best_col, col)
                progress.append((len(seen), position, path))
            if goal_name == "top" and row <= 10:
                return path, len(seen), progress
            if goal_name == "right" and col >= 32:
                return path, len(seen), progress
        if len(path) >= max_depth or node.terminal():
            continue
        for symbol, action in controls.items():
            child = node.clone()
            child.step(*action)
            child_key = key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            queue.append(path + symbol)
    return None, len(seen), progress


def probe(env):
    enter_level_6(env)
    base = np.asarray(env.frame()).copy()
    if len(sys.argv) > 2 and sys.argv[1] == "clickafter":
        context = env.clone()
        apply_symbols(context, sys.argv[2])
        context_base = np.asarray(context.frame()).copy()
        points = set(rare_click_points(context_base))
        points.update(
            (col, row)
            for row in range(2, 62, 4)
            for col in range(2, 64, 4)
        )
        effects = {}
        for point in sorted(points):
            clone = context.clone()
            clone.step(6, *point)
            effect = gameplay_delta(context_base, clone.frame())
            if effect is not None:
                effects.setdefault(effect, []).append(point)
        print("CONTEXT_CLICK_EFFECTS", sys.argv[2])
        for effect, effect_points in effects.items():
            print(effect_points, "=>", effect)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "selectscan":
        context = env.clone()
        if len(sys.argv) > 2:
            apply_symbols(context, sys.argv[2])
        context_base = np.asarray(context.frame()).copy()
        baselines = {}
        for action in (1, 2, 3, 4):
            baseline = context.clone()
            baseline.step(action)
            baselines[action] = np.asarray(baseline.frame()).copy()
        points = set(rare_click_points(context_base))
        points.update(
            (col, row)
            for row in range(2, 62, 4)
            for col in range(2, 64, 4)
        )
        effects = {}
        for point in sorted(points):
            selected = context.clone()
            selected.step(6, *point)
            if gameplay_delta(context_base, selected.frame()) is not None:
                continue
            for action in (1, 2, 3, 4):
                child = selected.clone()
                child.step(action)
                effect = gameplay_delta(baselines[action], child.frame())
                if effect is not None:
                    effects.setdefault((action, effect), []).append(point)
        print("SELECT_FOLLOWUPS", sys.argv[2] if len(sys.argv) > 2 else "INITIAL")
        for effect, effect_points in effects.items():
            print(effect_points, "=>", effect)
        return
    if len(sys.argv) > 2 and sys.argv[1] == "regionclicks":
        context = env.clone()
        apply_symbols(context, sys.argv[2])
        context_base = np.asarray(context.frame()).copy()
        effects = {}
        for row in range(30, 45):
            for col in range(42, 61):
                clone = context.clone()
                clone.step(6, col, row)
                effect = gameplay_delta(context_base, clone.frame())
                if effect is not None:
                    effects.setdefault(effect, []).append((col, row))
        print("REGION_CLICK_EFFECTS")
        for effect, effect_points in effects.items():
            print(effect_points, "=>", effect)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "tiles":
        context = env.clone()
        if len(sys.argv) > 2:
            apply_symbols(context, sys.argv[2])
        pixels = np.asarray(context.frame())
        for name, bounds in (
                ("upper", (18, 4, 20, 12)),
                ("start", (46, 16, 52, 22)),
                ("middle", (30, 6, 38, 14)),
                ("lower", (56, 16, 62, 22)),
                ("right_top", (14, 44, 22, 60)),
                ("right_lower", (44, 44, 54, 60)),
                ("next_pad", (46, 30, 62, 40))):
            row0, col0, row1, col1 = bounds
            print(name, bounds)
            for row in pixels[row0:row1, col0:col1]:
                print("".join(f"{int(value):x}" for value in row))
        return
    if len(sys.argv) > 2 and sys.argv[1] == "detail":
        context = env.clone()
        sequence = sys.argv[2]
        apply_symbols(context, sequence[:-1])
        before = np.asarray(context.frame()).copy()
        apply_symbols(context, sequence[-1])
        after = np.asarray(context.frame()).copy()
        print("DETAIL", sequence[-1])
        changed = np.argwhere(before[:62] != after[:62])
        for row, col in changed:
            print(
                (int(row), int(col)),
                int(before[row, col]),
                "->",
                int(after[row, col]),
            )
        return
    if len(sys.argv) > 1 and sys.argv[1] == "checkers":
        for sequence in sys.argv[2:] or ("",):
            context = env.clone()
            apply_symbols(context, sequence)
            pixels = np.asarray(context.frame())
            checkers = []
            for row in range(0, 61):
                for col in range(0, 63):
                    tile = pixels[row:row + 2, col:col + 2]
                    if (
                        int(tile[0, 0]) == int(tile[1, 1])
                        and int(tile[0, 1]) == int(tile[1, 0])
                        and int(tile[0, 0]) != int(tile[0, 1])
                    ):
                        checkers.append(
                            (row, col, int(tile[0, 0]), int(tile[0, 1]))
                        )
            print("CHECKERS", sequence or "INITIAL", checkers)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "parities":
        right_paths = {
            1: "UULLLLLBUURLDDB",
            0: "UULLLLLBRBUUDDLBRBL",
        }
        common = (
            "RRRRRDD"
            "AAAAA"
            "LLLLDLDLLDLLU"
            "A"
            "UUUUUU"
            "B"
            "UUUUUUUUUUUUU"
        )
        for right_parity, right_path in right_paths.items():
            for upper_parity in (1, 0):
                sequence = (
                    right_path
                    + common
                    + ("" if upper_parity else "B")
                    + "LC"
                )
                clone = env.clone()
                apply_symbols(clone, sequence)
                print(
                    "PARITY",
                    (right_parity, upper_parity),
                    "level", clone.levels_completed,
                    "avatar", avatar_position(clone.frame()),
                    "terminal", clone.terminal(),
                    "checkers",
                    [
                        (blob.color, blob.bbox, blob.area)
                        for blob in connected_components(
                            clone.frame(), colors=(13,), min_area=1
                        )
                    ],
                )
        return
    if len(sys.argv) > 1 and sys.argv[1] == "goalcombos":
        prefix = (
            "UULLLLLBUURLDDB"
            "RRRRRDD"
            "AAAAA"
            "LLLLDLDLLDLLU"
            "A"
            "UUUUUU"
            "B"
            "UUUUUUUUUUUUU"
            "L"
        )
        for a_phase in range(6):
            for b_phase in range(2):
                clone = env.clone()
                apply_symbols(clone, prefix + "A" * a_phase + "B" * b_phase + "C")
                print(
                    "COMBO", (a_phase, b_phase),
                    "level", clone.levels_completed,
                    "avatar", avatar_position(clone.frame()),
                )
        return
    if len(sys.argv) > 2 and sys.argv[1] == "finalcombos":
        prefix = sys.argv[2]
        found = []
        fifteen = {}
        for a_phase in range(6):
            for b_phase in range(2):
                for c_phase in range(4):
                    clone = env.clone()
                    apply_symbols(
                        clone,
                        prefix
                        + "A" * a_phase
                        + "B" * b_phase
                        + "C" * c_phase,
                    )
                    if clone.levels_completed > 5:
                        found.append((a_phase, b_phase, c_phase))
                    count = color_counts(clone.frame()).get(15, 0)
                    fifteen.setdefault(count, []).append(
                        (a_phase, b_phase, c_phase)
                    )
        print("FINAL_COMBOS", found, "COLOR15", fifteen)
        return
    if len(sys.argv) > 2 and sys.argv[1] == "ringcycle":
        context = env.clone()
        apply_symbols(context, sys.argv[2])
        control = sys.argv[3] if len(sys.argv) > 3 else "Y"
        seen = {np.asarray(context.frame())[:62].tobytes(): 0}
        for phase in range(1, 13):
            before = np.asarray(context.frame()).copy()
            apply_symbols(context, control)
            frame = np.asarray(context.frame())
            key = frame[:62].tobytes()
            rings = [
                (blob.bbox, blob.area)
                for blob in connected_components(
                    frame, colors=(8,), min_area=4
                )
            ]
            print(
                "RING", phase,
                "repeat", seen.get(key),
                "level", context.levels_completed,
                "avatar", avatar_position(frame),
                "delta", gameplay_delta(before, frame),
                "color8", rings,
            )
            if key in seen:
                break
            seen[key] = phase
        return
    if len(sys.argv) > 2 and sys.argv[1] == "image":
        context = env.clone()
        apply_symbols(context, sys.argv[2])
        palette = np.asarray([
            (0, 0, 0), (30, 90, 200), (220, 50, 40), (40, 170, 70),
            (245, 210, 40), (150, 150, 150), (210, 50, 180),
            (245, 130, 30), (40, 190, 230), (120, 30, 40),
            (20, 130, 120), (210, 170, 20), (110, 60, 170),
            (245, 120, 170), (245, 245, 245), (150, 240, 60),
        ], dtype=np.uint8)
        image = Image.fromarray(palette[np.asarray(context.frame())])
        image.resize((512, 512), resample=Image.Resampling.NEAREST).save(
            sys.argv[3] if len(sys.argv) > 3 else "level6.png"
        )
        print("WROTE", sys.argv[3] if len(sys.argv) > 3 else "level6.png")
        return
    if len(sys.argv) > 1 and sys.argv[1] == "grid":
        print("GRID_2X2")
        for index, row in enumerate(coarse_grid(base, cell=2)):
            print(f"{index * 2:02d} {row}")
        return
    if len(sys.argv) > 1 and sys.argv[1] == "clicks":
        effects = {}
        for point in rare_click_points(base):
            clone = env.clone()
            clone.step(6, *point)
            effect = gameplay_delta(base, clone.frame())
            if effect is not None:
                effects.setdefault(effect, []).append(point)
        print("RARE_CLICK_EFFECTS")
        for effect, points in effects.items():
            print(points, "=>", effect)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "allclicks":
        effects = {}
        for row in range(2, 62, 4):
            for col in range(2, 64, 4):
                clone = env.clone()
                clone.step(6, col, row)
                effect = gameplay_delta(base, clone.frame())
                if effect is not None:
                    effects.setdefault(effect, []).append((col, row))
        print("SAMPLED_CLICK_EFFECTS")
        for effect, points in effects.items():
            print(points, "=>", effect)
        return
    if len(sys.argv) > 2 and sys.argv[1] == "sequence":
        points = {"A": (55, 7), "B": (51, 25), "C": (50, 48)}
        actions = {"U": 1, "D": 2, "L": 3, "R": 4}
        for sequence in sys.argv[2:]:
            clone = env.clone()
            prior = np.asarray(clone.frame()).copy()
            print("SEQUENCE", sequence)
            for symbol in sequence:
                if symbol in points:
                    clone.step(6, *points[symbol])
                else:
                    clone.step(actions[symbol])
                current = np.asarray(clone.frame()).copy()
                print(
                    symbol,
                    "avatar", avatar_position(current),
                    "level", clone.levels_completed,
                    "delta", gameplay_delta(prior, current),
                )
                prior = current
            print("FINAL_GRID_2X2")
            for index, row in enumerate(coarse_grid(clone.frame(), cell=2)):
                print(f"{index * 2:02d} {row}")
        return
    if len(sys.argv) > 1 and sys.argv[1] == "reach":
        sequence = sys.argv[2] if len(sys.argv) > 2 else ""
        clone = env.clone()
        apply_symbols(clone, sequence)
        paths, wins, states = movement_reachability(clone)
        print("REACH", sequence or "INITIAL", "states", states, "positions", len(paths))
        if paths:
            print(
                "BBOX",
                (
                    min(row for row, col in paths),
                    min(col for row, col in paths),
                    max(row for row, col in paths),
                    max(col for row, col in paths),
                ),
            )
            for position in sorted(paths):
                print(position, paths[position])
        print("WINS", wins[:3])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "reachmany":
        for sequence in sys.argv[2:]:
            clone = env.clone()
            apply_symbols(clone, "" if sequence == "-" else sequence)
            paths, wins, states = movement_reachability(clone, max_states=500)
            bbox = None
            if paths:
                bbox = (
                    min(row for row, col in paths),
                    min(col for row, col in paths),
                    max(row for row, col in paths),
                    max(col for row, col in paths),
                )
            extremes = sorted(
                paths.items(),
                key=lambda item: (
                    item[0][0], item[0][1], -len(item[1])
                ),
            )[:2]
            print(
                sequence, "states", states, "positions", len(paths),
                "bbox", bbox, "extremes", extremes, "wins", wins[:1],
            )
        return
    if len(sys.argv) > 2 and sys.argv[1] == "movegraph":
        root = env.clone()
        apply_symbols(root, sys.argv[2])
        names = {1: "U", 2: "D", 3: "L", 4: "R"}
        queue = deque([root.clone()])
        key = lambda node: np.asarray(node.frame())[:62].tobytes()
        seen = {key(root)}
        edges = set()
        while queue and len(seen) < 200:
            node = queue.popleft()
            source = avatar_position(node.frame())
            for action in (1, 2, 3, 4):
                child = node.clone()
                child.step(action)
                target = avatar_position(child.frame())
                edges.add((source, names[action], target))
                child_key = key(child)
                if child_key not in seen:
                    seen.add(child_key)
                    queue.append(child)
        for edge in sorted(edges):
            print(edge)
        return
    if len(sys.argv) > 2 and sys.argv[1] == "ringbfs":
        root = env.clone()
        apply_symbols(root, sys.argv[2])
        macros = {
            "U": ((1,), (6, 50, 32), (2,)),
            "L": ((3,), (6, 46, 36), (4,)),
            "D": ((2,), (6, 50, 40), (1,)),
            "R": ((4,), (6, 54, 36), (3,)),
        }
        key = lambda node: np.asarray(node.frame())[:62].tobytes()
        queue = deque([(root.clone(), "")])
        seen = {key(root)}
        rings = {central_ring(root.frame()): ""}
        wins = []
        exits = []
        while queue and len(seen) < 200:
            node, path = queue.popleft()
            for symbol, macro in macros.items():
                child = node.clone()
                for action in macro:
                    child.step(*action)
                    if child.levels_completed > 5:
                        wins.append(path + symbol)
                        break
                child_key = key(child)
                if child_key in seen:
                    continue
                seen.add(child_key)
                ring = central_ring(child.frame())
                rings.setdefault(ring, path + symbol)
                if color_counts(child.frame()).get(15, 0) > 2:
                    exits.append((path + symbol, ring))
                queue.append((child, path + symbol))
        print("RING_BFS", "states", len(seen), "wins", wins, "exits", exits)
        for ring, path in sorted(rings.items()):
            print(ring, path)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "cycles":
        for control, point in (("A", (55, 7)), ("B", (51, 25))):
            clone = env.clone()
            initial = np.asarray(clone.frame()).copy()
            seen = {initial[:62].tobytes(): 0}
            print("CONTROL", control)
            for phase in range(1, 21):
                clone.step(6, *point)
                current = np.asarray(clone.frame()).copy()
                key = current[:62].tobytes()
                pieces = [
                    (blob.color, blob.bbox, blob.area)
                    for blob in connected_components(
                        current, colors=(1, 6, 7, 9, 10, 11, 12), min_area=2
                    )
                ]
                print(
                    phase,
                    "repeat", seen.get(key),
                    "delta", gameplay_delta(initial, current),
                    "pieces", pieces,
                )
                if key in seen:
                    break
                seen[key] = phase
        return
    if len(sys.argv) > 2 and sys.argv[1] == "search":
        goal_name = sys.argv[2]
        prefix = sys.argv[3] if len(sys.argv) > 3 else ""
        max_states = int(sys.argv[4]) if len(sys.argv) > 4 else 8000
        max_depth = int(sys.argv[5]) if len(sys.argv) > 5 else 100
        root = env.clone()
        apply_symbols(root, prefix)
        path, states, progress = controlled_search(
            root, goal_name, max_states=max_states, max_depth=max_depth
        )
        print("SEARCH", goal_name, "states", states, "path", path)
        print("PROGRESS")
        for item in progress:
            print(item)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "outcomes":
        for sequence in sys.argv[2:]:
            clone = env.clone()
            last_effect = None
            stopped_at = None
            for index, symbol in enumerate(sequence):
                prior = np.asarray(clone.frame()).copy()
                apply_symbols(clone, symbol)
                last_effect = gameplay_delta(prior, clone.frame())
                if clone.terminal():
                    stopped_at = index + 1
                    break
            paths, wins, states = movement_reachability(clone, max_states=500)
            print(
                sequence,
                "steps", stopped_at or len(sequence),
                "avatar", avatar_position(clone.frame()),
                "level", clone.levels_completed,
                "terminal", clone.terminal(),
                "last_delta", last_effect,
                "reach", (states, len(paths)),
                "wins", wins[:1],
            )
        return
    print("LEVEL", env.levels_completed, "ACTIONS", env.actions)
    print("COUNTS", color_counts(base))
    print("COMPONENTS")
    for component in compact_components(base):
        print(component)
    print("COARSE")
    for row in coarse_grid(base):
        print(row)
    print("DIRECTION_DELTAS")
    for action in (1, 2, 3, 4):
        clone = env.clone()
        clone.step(action)
        print(action, frame_delta(base, clone.frame()))


if __name__ == "__main__":
    arena.run_program("dc22", probe)
