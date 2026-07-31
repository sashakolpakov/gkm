import json
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import perception as P
from legs import _bridge_carrier_moves


LOCAL_MACROS = (
    ((18, 18), (30, 18)),
    ((24, 18), (36, 18)),
    ((30, 18), (42, 18)),
    ((36, 18), (48, 18)),
    ((48, 12), (48, 24)),
    ((54, 24), (42, 24)),
    ((42, 18), (42, 30)),
    ((42, 24), (42, 36)),
    ((42, 30), (42, 42)),
    ((42, 36), (42, 48)),
)


def pieces(frame):
    blobs = P.connected_components(frame, colors=(1, 8, 12, 14))
    groups = []
    for color in (1, 12, 8, 14):
        groups.append(frozenset(
            blob.top_left for blob in blobs
            if blob.color == color and blob.size == (4, 4)
            and (color != 1 or blob.area == 16)
        ))
    return tuple(groups)


def click(env, cell):
    env.step(6, cell[1] + 1, cell[0] + 1)


def stage_local(env):
    with open("checkpoint.json") as handle:
        prefix = json.load(handle)["final_path"]
    for action in prefix:
        env.step(*(action if isinstance(action, list) else [action]))
    for source, destination in LOCAL_MACROS:
        click(env, source)
        click(env, destination)


def stage(env):
    stage_local(env)
    _, carriers, bridges, _ = pieces(env.frame())
    click(env, next(iter(bridges)))
    click(env, next(iter(carriers)))
    for _ in range(9):
        env.step(4)


def key(frame):
    data = P.arr(frame).copy()
    data[0, 0] = 0
    return data.tobytes()


def candidate_leaps(frame):
    holes, carriers, bridges, pegs = pieces(frame)
    occupied = bridges | pegs
    for source in sorted(occupied):
        for destination in sorted(holes | carriers):
            dr = destination[0] - source[0]
            dc = destination[1] - source[1]
            if (abs(dr), abs(dc)) not in ((12, 0), (0, 12)):
                continue
            midpoint = (source[0] + dr // 2, source[1] + dc // 2)
            if midpoint in occupied - {source}:
                yield source, midpoint, destination


def complete_moves(frame):
    holes, carriers, bridges, pegs = pieces(frame)
    static_bridges = frozenset(
        (blob.bbox[0] + 1, blob.bbox[1])
        for blob in P.connected_components(frame, colors=(15,))
        if blob.size == (4, 4)
    )
    occupied = bridges | pegs | static_bridges
    moves = []
    for source in sorted(bridges | pegs):
        for destination in sorted(holes | carriers):
            dr = destination[0] - source[0]
            dc = destination[1] - source[1]
            if (abs(dr), abs(dc)) not in ((12, 0), (0, 12)):
                continue
            midpoint = (source[0] + dr // 2, source[1] + dc // 2)
            if midpoint in occupied - {source}:
                moves.append((source, midpoint, destination))
    return tuple(moves)


def alignment_gap(frame):
    holes, carriers, bridges, pegs = pieces(frame)
    occupied = bridges | pegs
    candidates = []
    for source in occupied:
        for destination in holes | carriers:
            dr = destination[0] - source[0]
            dc = destination[1] - source[1]
            if (abs(dr), abs(dc)) not in ((12, 0), (0, 12)):
                continue
            midpoint = (source[0] + dr // 2, source[1] + dc // 2)
            others = occupied - {source}
            if others:
                candidates.append(min(
                    abs(midpoint[0] - piece[0])
                    + abs(midpoint[1] - piece[1])
                    for piece in others
                ))
    return min(candidates) if candidates else 10 ** 9


def ascii_map(frame):
    symbols = {
        0: "0", 1: ".", 5: "#", 8: "B", 9: "+", 10: " ",
        11: "c", 12: "C", 14: "P",
    }
    data = P.arr(frame)
    return "\n".join(
        "".join(symbols.get(int(data[row, col]), "?")
                for col in range(0, 64, 2))
        for row in range(0, 64, 2)
    )


def probe(env):
    stage(env)
    root = env.clone()
    base_level = root.levels_completed
    print("remote counts", P.color_counts(root.frame()))
    print("special blobs", [
        (blob.color, blob.bbox, blob.area)
        for blob in P.connected_components(root.frame(), min_area=3)
        if blob.color not in (0, 1, 5, 8, 9, 10, 11, 12, 14)
    ])
    print("remote map\n" + ascii_map(root.frame()))
    for horizontal_action in (3, 4):
        corridor = root.clone()
        corridor.step(1)
        corridor.step(1)
        for count in range(10):
            moves = complete_moves(corridor.frame())
            if moves:
                print(
                    "corridor macro", horizontal_action, count,
                    "moves", moves,
                    "C/B/P", sorted(pieces(corridor.frame())[1]),
                    sorted(pieces(corridor.frame())[2]),
                    sorted(pieces(corridor.frame())[3]),
                )
            corridor.step(horizontal_action)
    for outward_count in range(5):
        lower_corridor = root.clone()
        lower_corridor.step(1)
        lower_corridor.step(1)
        for _ in range(outward_count):
            lower_corridor.step(3)
        lower_corridor.step(2)
        lower_corridor.step(2)
        for return_count in range(13):
            moves = complete_moves(lower_corridor.frame())
            if moves:
                print(
                    "lower corridor macro", outward_count, return_count,
                    "moves", moves,
                    "C/B/P", sorted(pieces(lower_corridor.frame())[1]),
                    sorted(pieces(lower_corridor.frame())[2]),
                    sorted(pieces(lower_corridor.frame())[3]),
                )
            lower_corridor.step(4)
    for horizontal_action in (3, 4):
        corridor = root.clone()
        corridor.step(1)
        corridor.step(1)
        for horizontal_count in range(10):
            branch = corridor.clone()
            trace = []
            for down_count in range(1, 7):
                before = pieces(branch.frame())
                branch.step(2)
                after = pieces(branch.frame())
                moves = complete_moves(branch.frame())
                if after != before or moves:
                    trace.append((
                        down_count, sorted(after[1]), sorted(after[2]),
                        sorted(after[3]), moves,
                    ))
            if trace:
                print(
                    "corridor down", horizontal_action, horizontal_count,
                    trace,
                )
            corridor.step(horizontal_action)
    route = (1, 1, 4, 1)
    node = root.clone()
    for action in route:
        node.step(action)
    print(
        "staged pair", route,
        "pieces", tuple(sorted(group) for group in pieces(node.frame())),
        "candidates", tuple(candidate_leaps(node.frame())),
    )
    test = node.clone()
    before = pieces(test.frame())
    click(test, (30, 28))
    click(test, (18, 28))
    after = pieces(test.frame())
    entered = test.clone()
    print(
        "static bridge leap", after != before,
        "reward", test.levels_completed - base_level,
        "pieces", tuple(sorted(group) for group in after),
    )
    upper = entered.clone()
    upper.step(2)
    for _ in range(3):
        upper.step(3)
    for _ in range(2):
        upper.step(1)
    upper_macros = (
        ((30, 28), (18, 28)),
        ((18, 28), (18, 40)),
        ((18, 46), (18, 34)),
        ((18, 40), (18, 28)),
        ((18, 28), (30, 28)),
    )
    for macro in upper_macros:
        before = pieces(upper.frame())
        click(upper, macro[0])
        click(upper, macro[1])
        after = pieces(upper.frame())
        print(
            "upper macro", macro, "changed", after != before,
            "pegs", len(before[3]), "->", len(after[3]),
            "C/B/P", sorted(after[1]), sorted(after[2]), sorted(after[3]),
            "reward", upper.levels_completed - base_level,
        )
    descend = upper.clone()
    descend_trace = []
    for count in range(5):
        descend_trace.append((
            count, tuple(sorted(group) for group in pieces(descend.frame()))
        ))
        descend.step(2)
    print("upper descend", descend_trace)
    lower_root = upper.clone()
    lower_root.step(2)
    lower_root.step(2)
    for action in (3, 4):
        branch = lower_root.clone()
        for count in range(13):
            moves = _bridge_carrier_moves(branch.frame())
            current_pieces = pieces(branch.frame())
            if action == 4:
                print(
                    "lower travel", count,
                    "C", sorted(current_pieces[1]),
                    "P", sorted(current_pieces[3]),
                )
            if moves:
                print(
                    "lower dock", action, count, "moves", moves,
                    "C/B/P", sorted(pieces(branch.frame())[1]),
                    sorted(pieces(branch.frame())[2]),
                    sorted(pieces(branch.frame())[3]),
                )
            branch.step(action)
    trace = []
    for count in range(5):
        trace.append((
            count, tuple(sorted(group) for group in pieces(test.frame()))
        ))
        test.step(1)
    print("peg follows", trace)
    for action in (3, 4, 2):
        branch = test.clone()
        branch_trace = []
        for count in range(5):
            branch_trace.append((
                count, tuple(sorted(group) for group in pieces(branch.frame()))
            ))
            branch.step(action)
        print("empty carrier action", action, branch_trace)
    for horizontal_action in (3, 4):
        branch = entered.clone()
        route_trace = []
        branch.step(2)
        for count in range(6):
            route_trace.append((
                count, tuple(sorted(group) for group in pieces(branch.frame()))
            ))
            branch.step(horizontal_action)
        for _ in range(3):
            branch.step(2)
        route_trace.append((
            "down", tuple(sorted(group) for group in pieces(branch.frame()))
        ))
        print("split carrier", horizontal_action, route_trace)
    split = entered.clone()
    split.step(2)
    for horizontal_count in range(13):
        branch = split.clone()
        for _ in range(horizontal_count):
            branch.step(3)
        before = pieces(branch.frame())
        for _ in range(5):
            branch.step(1)
        after = pieces(branch.frame())
        if after != before:
            print(
                "peg carrier shaft", horizontal_count,
                "before C/B/P", sorted(before[1]), sorted(before[2]),
                sorted(before[3]),
                "after C/B/P", sorted(after[1]), sorted(after[2]),
                sorted(after[3]),
            )
    parked = pieces(node.frame())
    for horizontal_action in (3, 4):
        for horizontal_count in range(13):
            test = node.clone()
            for _ in range(horizontal_count):
                test.step(horizontal_action)
            trace = []
            for down_count in range(1, 9):
                before = pieces(test.frame())
                test.step(2)
                after = pieces(test.frame())
                if after != before:
                    trace.append((
                        down_count, sorted(after[2]), sorted(after[3]),
                        tuple(candidate_leaps(test.frame())),
                    ))
            if trace or pieces(test.frame()) != parked:
                print(
                    "hidden carrier", horizontal_action, horizontal_count,
                    "trace", trace,
                    "reward", test.levels_completed - base_level,
                )
    for source, midpoint, destination in candidate_leaps(node.frame()):
        child = node.clone()
        before = pieces(child.frame())
        click(child, source)
        click(child, destination)
        after = pieces(child.frame())
        print(
            "pair result", (source, midpoint, destination),
            "pegs", len(before[3]), "->", len(after[3]),
            "reward", child.levels_completed - base_level,
            "pieces", tuple(sorted(group) for group in after),
        )
    return
    scanner = root.clone()
    last_key = None
    for horizontal in range(12):
        current_key = key(scanner.frame())
        if current_key == last_key:
            break
        last_key = current_key
        if horizontal < 5:
            node = scanner.clone()
            trace = []
            for count in range(5):
                _, _, bridges, pegs = pieces(node.frame())
                trace.append((count, sorted(bridges), sorted(pegs)))
                node.step(1)
            print("up trace", horizontal, trace)
        scanner.step(3)
    return
    queue = deque([()])
    seen = {key(root.frame())}
    reports = 0
    best_gap = alignment_gap(root.frame())
    print("alignment", best_gap, "path", ())
    while queue and len(seen) < 600:
        path = queue.popleft()
        node = root.clone()
        for action in path:
            node.step(action)
        before = pieces(node.frame())
        for source, midpoint, destination in candidate_leaps(node.frame()):
            child = node.clone()
            click(child, source)
            click(child, destination)
            after = pieces(child.frame())
            if after != before:
                reports += 1
                print(
                    "success", path, (source, midpoint, destination),
                    "pegs", len(before[3]), "->", len(after[3]),
                    "reward", child.levels_completed - base_level,
                    "B", sorted(after[2]), "P", sorted(after[3]),
                )
        if len(path) >= 16:
            continue
        for action in (1, 2, 3, 4):
            child = root.clone()
            for replay_action in path + (action,):
                child.step(replay_action)
            child_key = key(child.frame())
            if child_key not in seen:
                seen.add(child_key)
                queue.append(path + (action,))
                gap = alignment_gap(child.frame())
                if gap < best_gap:
                    best_gap = gap
                    print("alignment", best_gap, "path", path + (action,))
    print("contexts", len(seen), "successful", reports, "best_gap", best_gap)


A.run_program("lf52", probe)
