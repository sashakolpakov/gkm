"""Compact clean-room probes for sk48 level 5 via the documented harness."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import perception as p
import players


def compact_objects(frame):
    return [
        (o["color"], o["bbox"], o["area"])
        for o in p.object_candidates(frame)
        if o["color"] != 1
    ]


def symbolic_state(env):
    objects = p.object_candidates(env.frame())
    pieces = []
    for o in objects:
        if o["color"] in (8, 9) and o["bbox"][0] < 53:
            pieces.append((o["color"], o["bbox"]))
    bodies = [
        (o["color"], o["bbox"], o["area"])
        for o in objects
        if o["color"] in (0, 5, 6)
        and o["bbox"][0] < 53
        and o["area"] < 100
    ]
    return {
        "level": env.levels_completed,
        "pieces": pieces,
        "body": bodies,
    }


def brief_state(env):
    state = symbolic_state(env)
    avatar = next(
        bbox
        for color, bbox, area in state["body"]
        if color == 6 and area == 18
    )
    return avatar[:2], [(color, bbox[:2]) for color, bbox in state["pieces"]]


def compact_deltas(env):
    return {
        action: (delta["count"], delta["bbox"])
        for action, delta in p.action_deltas(env, env.actions).items()
    }


def coarse_grid(frame):
    data = p.arr(frame)
    rows = []
    for r in range(0, 54, 6):
        row = []
        for c in range(0, 60, 6):
            values, counts = np.unique(data[r:r + 6, c:c + 6], return_counts=True)
            row.append(str(int(values[int(np.argmax(counts))])))
        rows.append("".join(row))
    return rows


def empty_train(env, _path):
    data = p.arr(env.frame())
    avatar_top = int(np.where(data[:53, :11] == 6)[0].min())
    return not np.isin(data[avatar_top:avatar_top + 6], (8, 9)).any()


def piece_key(env):
    data = p.arr(env.frame())
    return np.where(np.isin(data, (6, 8, 9)), data, 0).tobytes()


def direct_empty_paths(staged):
    found = []
    approaches = [()] + [
        (direction,) * count
        for direction in (1, 2)
        for count in range(1, 7)
    ]
    shifts = [()] + [
        (direction,) * count
        for direction in (3, 4)
        for count in range(1, 8)
    ]
    for approach in approaches:
        for shift in shifts:
            for crossing in (1, 2):
                path = list(approach + shift + (crossing,))
                branch = p.replay(staged, path)
                if empty_train(branch, path):
                    found.append((path, brief_state(branch)))
                    if len(found) == 8:
                        return found
    return found


def lift_near_nine_paths(empty):
    found = []
    for extension in range(1, 6):
        for retraction in range(0, 6):
            for direction in (1, 2):
                path = [2] + [4] * extension + [3] * retraction + [direction]
                branch = p.replay(empty, path)
                state = brief_state(branch)
                if all(not (color == 9 and pos[0] == 25) for color, pos in state[1]):
                    found.append((path, state))
    return found[:8]


def park_above_paths(lifted):
    found = []
    for direction in (3, 4):
        for count in range(0, 8):
            path = [direction] * count + [2]
            branch = p.replay(lifted, path)
            state = brief_state(branch)
            if all(not (color == 9 and pos[0] == 25) for color, pos in state[1]):
                found.append((path, state))
    return found


def collect_one_eight_paths(cleared):
    found = []
    seen = set()
    for extension in range(1, 9):
        for retraction in range(1, 9):
            path = [4] * extension + [3] * retraction
            branch = p.replay(cleared, path)
            state = brief_state(branch)
            eights = tuple(pos for color, pos in state[1] if color == 8)
            if any(pos[1] < 36 for pos in eights) and eights not in seen:
                seen.add(eights)
                found.append((path, state))
    return found


def contact_eight_paths(staged):
    initial = tuple(
        pos for color, pos in brief_state(staged)[1] if color == 8
    )
    found = []
    seen = set()
    for first_retract in range(0, 5):
        for extension in range(1, 8):
            for second_retract in range(1, 8):
                path = (
                    [3] * first_retract
                    + [4] * extension
                    + [3] * second_retract
                )
                branch = p.replay(staged, path)
                state = brief_state(branch)
                eights = tuple(pos for color, pos in state[1] if color == 8)
                if eights != initial and eights not in seen:
                    seen.add(eights)
                    found.append((path, state, branch.levels_completed))
    return found


def contact_use_paths(staged):
    initial = tuple(
        pos for color, pos in brief_state(staged)[1] if color == 8
    )
    found = []
    seen = set()
    for first_retract in range(0, 5):
        for extension in range(0, 8):
            for second_retract in range(0, 6):
                path = (
                    [3] * first_retract
                    + [4] * extension
                    + [6]
                    + [3] * second_retract
                )
                branch = p.replay(staged, path)
                state = brief_state(branch)
                eights = tuple(pos for color, pos in state[1] if color == 8)
                if eights != initial and eights not in seen:
                    seen.add(eights)
                    found.append((path, state, branch.levels_completed))
    return found


def probe(env):
    for level in range(1, 5):
        getattr(players, f"play_level_{level}")(env)
    print("LEVEL", env.levels_completed + 1, "ACTIONS", env.actions)
    print("COLORS", p.color_counts(env.frame()))
    print("GRID", "/".join(coarse_grid(env.frame())))
    print("OBJECTS", compact_objects(env.frame()))
    print(
        "DELTAS",
        {
            action: {
                "count": delta["count"],
                "bbox": delta["bbox"],
            }
            for action, delta in p.action_deltas(env, env.actions).items()
        },
    )
    print("POST6_DELTAS", compact_deltas(p.replay(env, [6])))
    separated = p.replay(env, [1, 4])
    print("SEPARATED6_DELTAS", compact_deltas(p.replay(separated, [6])))
    sequences = [
        [3],
        [3, 3],
        [3, 3, 3],
        [3, 3, 3, 3],
        [1, 4],
        [1, 4, 4, 4],
        [1, 3, 4],
        [1, 6],
        [3, 6],
        [1, 4, 6],
        [2, 4, 6],
        [1, 4, 2],
        [1, 4, 2, 3],
        [1, 4, 2, 3, 1],
        [1, 3, 2],
        [1, 3, 2, 1, 3, 2],
        [1, 3, 2, 1, 3, 2, 1, 3, 2],
    ]
    for sequence in sequences:
        clone = p.replay(env, sequence)
        print("SEQ", "".join(map(str, sequence)), symbolic_state(clone))
    one_nine = p.replay(env, [1, 3, 2, 1, 3, 2])
    for extension in (3, 4, 5, 6):
        sequence = [1] + [4] * extension + [2]
        print("VERTICAL8", extension, brief_state(p.replay(one_nine, sequence)))
    clone = env.clone()
    players.reverse_row_train(
        clone,
        approach_lanes=4,
        stages=((5, 6, 6), (4, 6, 5), (2, 5, 4)),
        final_extension=4,
    )
    print("LEG reverse_row_train", symbolic_state(clone))
    for extension in range(0, 5):
        sequence = [1] + [4] * extension + [2]
        print("STAGED_VERTICAL8", extension, brief_state(p.replay(clone, sequence)))
    for sequence in ([1], [1, 1], [1, 1, 1], [1, 1, 1, 2, 2, 2]):
        branch = p.replay(clone, sequence)
        print("POST", "".join(map(str, sequence)), symbolic_state(branch))
    trace = env.clone()
    operations = (
        ("U4", 1, 4),
        ("E5", 4, 5),
        ("D1", 2, 1),
        ("U1", 1, 1),
        ("R5", 3, 5),
        ("D1", 2, 1),
        ("E6", 4, 6),
        ("R6", 3, 6),
    )
    for label, action, count in operations:
        for _ in range(count):
            trace.step(action)
        print("TRACE", label, brief_state(trace))
    trace = env.clone()
    players.move_vertical_lanes(trace, players.UP, 4)
    for index, (span, reach, compact) in enumerate(
        ((5, 6, 6), (4, 6, 5), (2, 5, 4)), start=1
    ):
        players.extend_tether(trace, span)
        players.move_vertical_lanes(trace, players.DOWN, 1)
        players.move_vertical_lanes(trace, players.UP, 1)
        players.retract_tether(trace, span)
        players.move_vertical_lanes(trace, players.DOWN, 1)
        players.extend_tether(trace, reach)
        players.retract_tether(trace, compact)
        print("STAGE", index, brief_state(trace))
    print("DIRECT_EMPTY", direct_empty_paths(clone))
    contact_ready = p.replay(clone, [3, 3, 4, 4, 4])
    print("CONTACT6_DELTAS", compact_deltas(p.replay(contact_ready, [6])))
    print("CONTACT_8", contact_eight_paths(clone))
    print("CONTACT_USE", contact_use_paths(clone))
    empty = p.replay(clone, [3, 3, 3, 1])
    for sequence in (
        [2],
        [2, 4],
        [2, 4, 1],
        [2, 4, 1, 3],
        [2, 4, 1, 3, 2],
        [2, 4, 4, 1, 3, 3, 2],
    ):
        print("CLEAR", sequence, brief_state(p.replay(empty, sequence)))
    print("LIFT_NINE", lift_near_nine_paths(empty))
    lifted = p.replay(empty, [2, 4, 4, 4, 1])
    print("PARK_ABOVE", park_above_paths(lifted))
    clear_path = [3, 3, 3, 1, 2, 4, 4, 4, 1, 3, 3, 3, 2]
    cleared = p.replay(clone, clear_path)
    print("CLEARED_ROW", brief_state(cleared))
    for sequence in (
        [4, 4, 4, 4],
        [4, 4, 4, 4, 3],
        [4, 4, 4, 4, 3, 3, 3, 3],
    ):
        print("COLLECT8", sequence, brief_state(p.replay(cleared, sequence)))
    print("COLLECT_ONE_8", collect_one_eight_paths(cleared))
    for sequence in (
        [6],
        [6] + [4] * 5 + [3] * 5,
        [4] * 5 + [6] + [3] * 5,
        [6] + [4] * 8 + [3] * 8,
        [4] * 8 + [6] + [3] * 8,
    ):
        branch = p.replay(cleared, sequence)
        print("EMPTY_USE", sequence, brief_state(branch))


levels, path, error = arena.run_program("sk48", probe)
print("END", levels, len(path), error)
