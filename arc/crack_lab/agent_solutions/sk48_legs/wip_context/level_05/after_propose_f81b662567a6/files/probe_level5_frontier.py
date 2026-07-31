"""Compact reproduction of the preserved level-5 frontier routes."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import perception as p
import players


def runs(spec):
    return tuple(action for action, count in spec for _ in range(count))


PREFIX = runs(
    (
        (4, 5), (1, 3), (2, 1), (3, 2), (2, 1), (6, 1),
        (1, 6), (2, 1), (4, 3), (6, 1), (1, 2), (4, 6), (2, 2),
        (2, 5), (3, 6), (4, 1), (3, 4), (1, 1), (2, 4),
        (6, 1), (3, 4), (4, 3), (6, 1), (4, 2), (6, 1), (1, 1),
    )
)
CONTROLLED_PAIR = (
    4, 4, 3, 3, 3, 3, 1, 1, 4, 4, 4, 4,
    3, 3, 3, 3, 3, 2,
)
LOWER_SPEC = (
    (1, 2), (2, 2), (4, 5), (1, 3), (2, 1), (3, 2), (2, 1),
    (1, 6), (2, 1), (4, 3), (1, 2), (4, 6), (2, 2),
)
LOWER_FINAL_EIGHT = runs(LOWER_SPEC)
SUFFIX = runs(
    (
        (2, 5), (3, 6), (4, 1), (3, 4), (1, 1), (2, 4),
        (6, 1), (3, 4), (4, 3), (6, 1), (4, 2), (6, 1), (1, 1),
    )
)
RECOLLECT_PAIR_LEFT = (
    2, 2, 3, 3, 1, 1, 1, 4, 4, 3, 3, 3, 3, 3, 3,
    1, 4, 4, 4, 3, 3, 3, 2,
)


def pieces(env):
    return tuple(
        (blob.color, blob.bbox[0], blob.bbox[1])
        for blob in p.connected_components(
            env.frame(), colors=(8, 9), min_area=4
        )
        if blob.bbox[0] < 53
    )


def avatar_row(env):
    data = p.arr(env.frame())
    return int(np.argwhere(data[:53, :11] == 6)[:, 0].min()) + 1


def controlled(env):
    before = set(pieces(env))
    found = []
    for action in (3, 4):
        branch = env.clone()
        branch.step(action)
        moved = before - set(pieces(branch))
        for row in sorted({piece[1] for piece in moved}):
            sequence = tuple(
                color
                for color, _row, _col in sorted(
                    (piece for piece in moved if piece[1] == row),
                    key=lambda piece: piece[2],
                )
            )
            if sequence:
                found.append((action, sequence))
    return tuple(found)


def lane_runs(env):
    data = p.arr(env.frame())
    row = avatar_row(env) + 1
    values = data[row]
    out = []
    start = 0
    for col in range(1, len(values) + 1):
        if col == len(values) or values[col] != values[start]:
            color = int(values[start])
            if color not in (4, 5):
                out.append((color, start, col - 1))
            start = col
    return tuple(out)


def threaded(env):
    data = p.arr(env.frame())
    row = avatar_row(env) + 1
    reach = 10
    while reach + 1 < 53 and int(data[row, reach + 1]) not in (4, 5):
        reach += 1
    return tuple(
        color
        for color, piece_row, col in sorted(
            (
                piece
                for piece in pieces(env)
                if piece[1] == avatar_row(env) and piece[2] <= reach
            ),
            key=lambda piece: piece[2],
        )
    ), reach


def apply(env, actions):
    for action in actions:
        env.step(action)


def report(label, env):
    print(
        label,
        "level", env.levels_completed,
        "avatar", avatar_row(env),
        "controlled", controlled(env),
        "threaded", threaded(env),
        "lane", lane_runs(env),
        "pieces", pieces(env),
    )


def brief(env):
    return (
        env.levels_completed,
        avatar_row(env),
        threaded(env),
        pieces(env),
    )


def run(env):
    for level in range(1, 5):
        getattr(players, f"play_level_{level}")(env)
    report("START", env)
    empty = env.clone()
    apply(empty, (1, 3, 2) * 3)
    report("EMPTY", empty)
    for extension in range(3, 8):
        sweep = empty.clone()
        apply(sweep, (1,) + (4,) * extension + (2,))
        report(f"EMPTY_SWEEP_{extension}", sweep)
    empty_below = env.clone()
    apply(empty_below, (2, 3, 1) * 3)
    report("EMPTY_BELOW", empty_below)
    for extension in range(3, 8):
        sweep = empty_below.clone()
        apply(sweep, (1,) + (4,) * extension + (2,))
        report(f"EMPTY_BELOW_SWEEP_{extension}", sweep)
    recollect = None
    suffix_root = None
    for label, actions in (
        ("PREFIX", PREFIX),
        ("CONTROLLED_PAIR", CONTROLLED_PAIR),
        ("LOWER_FINAL_EIGHT", LOWER_FINAL_EIGHT),
        ("SUFFIX", SUFFIX),
        ("RECOLLECT_PAIR_LEFT", RECOLLECT_PAIR_LEFT),
        ("ATTACH_PAIR", (4, 4, 4, 1)),
    ):
        apply(env, actions)
        report(label, env)
        if label == "CONTROLLED_PAIR":
            trace = env.clone()
            for action, count in LOWER_SPEC:
                apply(trace, (action,) * count)
                print("LOWER_TRACE", action, count, brief(trace))
            transfer = env.clone()
            for action, count in LOWER_SPEC[:10]:
                apply(transfer, (action,) * count)
            for shift in range(4):
                branch = transfer.clone()
                apply(branch, (4,) * shift)
                print("TRANSFER_SHIFT", shift, "READY", brief(branch))
                apply(branch, (1, 1))
                print("TRANSFER_SHIFT", shift, "LIFT", brief(branch))
                apply(branch, (2, 2))
                print("TRANSFER_SHIFT", shift, "DROP", brief(branch))
        if label == "RECOLLECT_PAIR_LEFT":
            recollect = env.clone()
        if label == "SUFFIX":
            suffix_root = env.clone()
    surplus_transfer = suffix_root.clone()
    apply(
        surplus_transfer,
        (3,) * 5 + (1, 4, 1, 2, 2, 3),
    )
    report("SURPLUS_PARK_EIGHT", surplus_transfer)
    apply(surplus_transfer, (1,) * 4 + (4,) * 7)
    report("SURPLUS_THREAD_NINE", surplus_transfer)
    assembly_root = None
    for descent in range(1, 4):
        apply(surplus_transfer, (2,))
        report(f"SURPLUS_DESCENT_{descent}", surplus_transfer)
        if descent == 2:
            assembly_root = surplus_transfer.clone()
    for leg_name, leg in (
        (
            "reverse",
            lambda node: players.reverse_row_train(
                node,
                approach_lanes=4,
                stages=((5, 6, 6), (4, 6, 5), (2, 5, 4)),
                final_extension=4,
            ),
        ),
        ("weave", players.weave_vertical_four_train),
        ("unweave", players.unweave_horizontal_pairs_to_vertical_heads),
    ):
        branch = assembly_root.clone()
        try:
            leg(branch)
            print("SURPLUS_LEG", leg_name, brief(branch))
        except IndexError:
            print("SURPLUS_LEG_INVALID", leg_name)
    assembly = assembly_root.clone()
    apply(assembly, (1,) * 2 + (3,) * 7 + (2,) * 4)
    report("ASSEMBLY_ANCHOR_TOP", assembly)
    apply(assembly, (1,) * 2 + (4, 2, 2, 3))
    report("ASSEMBLY_FIRST_EIGHT_PARKED", assembly)
    apply(
        assembly,
        (4,) * 7 + (3,) * 7 + (4,) * 3 + (3,) * 3,
    )
    report("ASSEMBLY_SECOND_EIGHT_PARKED", assembly)
    apply(assembly, (1,) * 4 + (4,) * 7)
    report("ASSEMBLY_ANCHOR_THREADED", assembly)
    for descent in range(1, 4):
        apply(assembly, (2,))
        report(f"ASSEMBLY_PUSH_NINE_{descent}", assembly)
    for approach in range(1, 4):
        for extension in range(4, 9):
            branch = recollect.clone()
            apply(branch, (1,) * approach + (4,) * extension + (2,) * (approach - 1))
            sequence, reach = threaded(branch)
            if sequence[:1] == (8,) or branch.levels_completed > recollect.levels_completed:
                print(
                    "THREAD_PAIR",
                    approach,
                    extension,
                    "level", branch.levels_completed,
                    "avatar", avatar_row(branch),
                    "threaded", (sequence, reach),
                    "pieces", pieces(branch),
                )
            if approach == 1 and extension in (6, 7):
                if extension == 7:
                    base = brief(branch)
                    for action in (1, 2, 3, 4, 6):
                        probe = branch.clone()
                        for count in range(1, 8):
                            probe.step(action)
                            current = brief(probe)
                            if current != base:
                                print("THREAD_7_PROBE", action, count, current)
                    cycle = branch.clone()
                    try:
                        apply(cycle, (3,) * 5 + (2,))
                        report("THREADED_PAIR_LEFT", cycle)
                        apply(cycle, LOWER_FINAL_EIGHT)
                        report("THREADED_PAIR_LOWER", cycle)
                        apply(cycle, SUFFIX)
                        report("THREADED_PAIR_SUFFIX", cycle)
                    except IndexError:
                        print("THREADED_PAIR_CYCLE_INVALID")
                        trace = branch.clone()
                        apply(trace, (3,) * 5 + (2,))
                        for action, count in LOWER_SPEC:
                            try:
                                apply(trace, (action,) * count)
                                print(
                                    "THREADED_PAIR_TRACE",
                                    action,
                                    count,
                                    brief(trace),
                                )
                            except IndexError:
                                print(
                                    "THREADED_PAIR_TRACE_INVALID",
                                    action,
                                    count,
                                )
                                break
                try:
                    apply(branch, LOWER_FINAL_EIGHT)
                    report(f"THREAD_{extension}_LOWER", branch)
                    apply(branch, SUFFIX)
                    report(f"THREAD_{extension}_SUFFIX", branch)
                except IndexError:
                    print(f"THREAD_{extension}_CYCLE_INVALID")
    for extension in range(1, 4):
        parked = recollect.clone()
        try:
            apply(
                parked,
                (1, 1) + (4,) * extension + (1, 2, 3) + (3,) * (extension - 1) + (2, 2),
            )
            report(f"PARK_NINE_{extension}", parked)
        except IndexError:
            print(f"PARK_NINE_{extension}_INVALID")
    parked = recollect.clone()
    try:
        apply(parked, (1,) * 3 + (4,) * 7 + (3,) * 6 + (2, 3, 2, 2))
        report("PARK_TOP_NINE", parked)
    except IndexError:
        print("PARK_TOP_NINE_INVALID")
    for retractions in range(5):
        branch = env.clone()
        try:
            apply(branch, (3,) * retractions)
            report(f"CYCLE_R{retractions}", branch)
            apply(branch, LOWER_FINAL_EIGHT)
            report(f"CYCLE_R{retractions}_LOWER", branch)
            apply(branch, SUFFIX)
            report(f"CYCLE_R{retractions}_SUFFIX", branch)
        except IndexError:
            print(f"CYCLE_R{retractions}_INVALID")


levels, path, error = arena.run_program("sk48", run)
print("END", levels, len(path), error)
