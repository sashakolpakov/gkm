# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.

from collections import deque
from heapq import heappop, heappush
from math import gcd

from perception import arr, connected_components


def _peg_board(frame):
    """Return the slot lattice and occupied slots visible in a peg board."""
    holes = {
        blob.top_left
        for blob in connected_components(frame, colors=(1,))
        if blob.size == (4, 4) and blob.area == 16
    }
    pegs = {
        blob.top_left
        for blob in connected_components(frame, colors=(14,))
        if blob.size == (4, 4)
    }
    return holes | pegs, frozenset(pegs)


def _lattice_step(slots):
    coordinates = [
        sorted({position[axis] for position in slots})
        for axis in (0, 1)
    ]
    differences = [
        later - earlier
        for values in coordinates
        for earlier, later in zip(values, values[1:])
        if later > earlier
    ]
    step = 0
    for difference in differences:
        step = gcd(step, difference)
    return step


def _peg_solution(slots, start):
    """Find captures that leave the confirmed winning state of one peg."""
    step = _lattice_step(slots)
    if not step:
        return None
    queue = deque([(start, ())])
    seen = {start}
    directions = ((-step, 0), (step, 0), (0, -step), (0, step))
    while queue:
        pegs, path = queue.popleft()
        if len(pegs) == 1:
            return path
        for source in sorted(pegs):
            for dr, dc in directions:
                jumped = (source[0] + dr, source[1] + dc)
                destination = (source[0] + 2 * dr, source[1] + 2 * dc)
                if (
                    jumped in pegs
                    and destination in slots
                    and destination not in pegs
                ):
                    child = frozenset((pegs - {source, jumped}) | {destination})
                    if child not in seen:
                        seen.add(child)
                        queue.append((child, path + ((source, destination),)))
    return None


def solve_peg_solitaire(env):
    """Solve a visible orthogonal peg-solitaire board using coordinate clicks."""
    slots, pegs = _peg_board(env.frame())
    solution = _peg_solution(slots, pegs)
    if solution is None:
        return
    for source, destination in solution:
        env.step(6, source[1] + 1, source[0] + 1)
        env.step(6, destination[1] + 1, destination[0] + 1)


def _carrier_capture_macros(frame):
    """Return visible peg captures, including an empty bordered carrier."""
    slots, pegs = _peg_board(frame)
    carriers = {
        blob.top_left
        for blob in connected_components(frame, colors=(12,))
        if blob.size == (4, 4)
    }
    step = _lattice_step(slots | carriers)
    if not step:
        return ()
    destinations = slots | carriers
    macros = []
    for source in sorted(pegs):
        for dr, dc in ((-step, 0), (step, 0), (0, -step), (0, step)):
            jumped = (source[0] + dr, source[1] + dc)
            destination = (source[0] + 2 * dr, source[1] + 2 * dc)
            if (
                jumped in pegs
                and destination in destinations
                and destination not in pegs
            ):
                macros.append((
                    (6, source[1] + 1, source[0] + 1),
                    (6, destination[1] + 1, destination[0] + 1),
                ))
    return tuple(macros)


def solve_peg_solitaire_with_carrier(env, max_states=2000, max_depth=40):
    """Solve peg boards connected by a key-movable bordered carrier slot."""
    base_level = env.levels_completed

    def key(node):
        frame = arr(node.frame())
        return int(node.levels_completed), frame[1:, :].tobytes()

    queue = deque([(env.clone(), ())])
    seen = {key(env)}
    solution = None
    while queue and len(seen) < max_states:
        node, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        macros = tuple((action,) for action in (1, 2, 3, 4))
        macros += _carrier_capture_macros(node.frame())
        for macro in macros:
            child = node.clone()
            for action in macro:
                if isinstance(action, tuple):
                    child.step(*action)
                else:
                    child.step(action)
                if child.levels_completed > base_level:
                    solution = path + (macro,)
                    break
            if solution is not None:
                queue.clear()
                break
            child_key = key(child)
            if child_key not in seen:
                seen.add(child_key)
                queue.append((child, path + (macro,)))

    if solution is None:
        return
    for macro in solution:
        for action in macro:
            if isinstance(action, tuple):
                env.step(*action)
            else:
                env.step(action)


def _bridge_carrier_state(frame):
    """Return the puzzle-relevant geometry of a bridge/carrier peg board."""
    blobs = connected_components(
        frame, colors=(1, 3, 11, 12, 14, 15)
    )
    slots = {
        blob.top_left
        for blob in blobs
        if blob.color == 1 and blob.size == (4, 4) and blob.area == 16
    }
    pegs = {
        blob.top_left
        for blob in blobs
        if blob.color == 14 and blob.size == (4, 4)
    }
    slots |= {
        (blob.bbox[0] - 1, blob.bbox[1] - 1)
        for blob in blobs
        if blob.color == 1 and blob.size == (2, 2) and blob.area == 4
    }
    carriers = frozenset(
        (
            (blob.bbox[0] - 1, blob.bbox[1] - 1)
            if blob.size == (2, 2) else blob.top_left
        )
        for blob in blobs
        if blob.color == 12 and blob.size in ((2, 2), (4, 4))
    )
    bridges = frozenset(
        (blob.bbox[0] + 1, blob.bbox[1])
        for blob in blobs
        if blob.color == 15 and blob.size == (4, 4)
    )
    borders = frozenset(
        blob.top_left
        for blob in blobs
        if blob.color == 11 and blob.area >= 4
    )
    selected_blobs = [
        blob for blob in blobs
        if blob.color == 3 and blob.area >= 4
    ]
    selected = (
        (selected_blobs[0].bbox[0] + 1, selected_blobs[0].bbox[1] + 1)
        if selected_blobs else None
    )
    return (
        frozenset(slots), frozenset(pegs), carriers,
        bridges, borders, selected,
    )


def _bridge_carrier_moves_from_state(state):
    slots, pegs, carriers, bridges, _, _ = state
    step = _lattice_step(slots | carriers)
    if not step:
        return ()
    moves = []
    destinations = slots | carriers
    for source in sorted(pegs):
        for dr, dc in ((-step, 0), (step, 0), (0, -step), (0, step)):
            midpoint = (source[0] + dr, source[1] + dc)
            destination = (source[0] + 2 * dr, source[1] + 2 * dc)
            if destination not in destinations or destination in pegs:
                continue
            if midpoint in pegs:
                moves.append(("capture", source, destination))
            elif midpoint in bridges:
                moves.append(("bridge", source, destination))
    return tuple(moves)


def _bridge_carrier_moves(frame):
    """Return legal (kind, source, destination) peg moves."""
    return _bridge_carrier_moves_from_state(_bridge_carrier_state(frame))


def solve_bridge_carrier_peg_solitaire(
        env, max_align_states=120, max_macros=40, alignment_lookahead=24):
    """Solve peg boards joined by persistent bridges and a movable carrier."""
    base_level = env.levels_completed
    node = env.clone()
    solution = []
    forbidden = set()

    def bridge_score(state, move):
        """Dense progress: distance from a bridge landing to another peg."""
        pegs = state[1]
        _, source, destination = move
        others = pegs - {source}
        if not others:
            return 10 ** 9
        return min(
            abs(destination[0] - peg[0]) + abs(destination[1] - peg[1])
            for peg in others
        )

    def choose_state(state, blocked):
        moves = _bridge_carrier_moves_from_state(state)
        captures = [move for move in moves if move[0] == "capture"]
        if captures:
            return captures[0]
        bridges = [
            move for move in moves
            if (move[1], move[2]) not in blocked
        ]
        return min(bridges, key=lambda move: bridge_score(state, move)) if bridges else None

    def choose(frame, blocked):
        return choose_state(_bridge_carrier_state(frame), blocked)

    def alignment_priority(state, path):
        slots, pegs = state[:2]
        ordered_pegs = sorted(pegs)
        distances = [
            abs(first[0] - second[0]) + abs(first[1] - second[1])
            for index, first in enumerate(ordered_pegs)
            for second in ordered_pegs[index + 1:]
        ]
        pair_distance = min(distances) if distances else 10 ** 9
        return -len(pegs), pair_distance, -len(slots), len(path)

    def alignment_path(root, blocked):
        root_state = _bridge_carrier_state(root.frame())
        serial = 0
        queue = [(alignment_priority(root_state, ()), serial, ())]
        seen = {root_state}
        best_path = None
        best_score = None
        first_candidate_seen = None
        while queue and len(seen) < max_align_states:
            _, _, path = heappop(queue)
            if (
                first_candidate_seen is not None
                and len(seen) >= first_candidate_seen + alignment_lookahead
            ):
                return best_path
            for action in (4, 3, 2, 1):
                child_path = path + (action,)
                child = root.clone()
                for replay_action in child_path:
                    child.step(replay_action)
                if child.levels_completed > base_level:
                    return child_path
                state = _bridge_carrier_state(child.frame())
                if state in seen:
                    continue
                seen.add(state)
                move = choose_state(state, blocked)
                if move is not None:
                    if move[0] == "capture":
                        return child_path
                    score = bridge_score(state, move)
                    if best_score is None or score < best_score:
                        best_path, best_score = child_path, score
                    if first_candidate_seen is None:
                        first_candidate_seen = len(seen)
                    step = _lattice_step(state[0] | state[2])
                    if step and score <= step:
                        return child_path
                if len(child_path) < 16:
                    serial += 1
                    heappush(
                        queue,
                        (alignment_priority(state, child_path), serial, child_path),
                    )
        return best_path

    for _ in range(max_macros):
        if node.levels_completed > base_level:
            break
        move = choose(node.frame(), forbidden)
        if move is None:
            keys = alignment_path(node, forbidden)
            if keys is None:
                return
            for action in keys:
                node.step(action)
                solution.append(action)
            if node.levels_completed > base_level:
                break
            move = choose(node.frame(), forbidden)
            if move is None:
                return
        kind, source, destination = move
        before = _bridge_carrier_state(node.frame())
        clicks = (
            (6, source[1] + 1, source[0] + 1),
            (6, destination[1] + 1, destination[0] + 1),
        )
        for action in clicks:
            node.step(*action)
            solution.append(action)
        after = _bridge_carrier_state(node.frame())
        if after == before:
            return
        if kind == "bridge":
            forbidden.add((destination, source))
        else:
            forbidden.clear()

    if node.levels_completed <= base_level:
        return
    for action in solution:
        if isinstance(action, tuple):
            env.step(*action)
        else:
            env.step(action)


def observe_level_6(env):
    """Temporarily report compact level geometry and one-step action effects."""
    from perception import color_counts, frame_delta

    def relevant(node):
        return tuple(
            (blob.color, blob.top_left)
            for blob in connected_components(
                node.frame(), colors=(8, 12, 14)
            )
            if blob.area >= 3
        )

    print("L6", env.actions, relevant(env))
    for action in env.actions:
        child = env.clone()
        before = child.frame()
        level = child.levels_completed
        child.step(action)
        delta = frame_delta(before, child.frame())
        print(
            "A", action, (delta["count"], delta["bbox"]),
            child.levels_completed - level, relevant(child),
        )
    tests = {
        "peg-over-8": ((6, 19, 25), (6, 19, 13)),
        "8-over-peg": ((6, 19, 19), (6, 19, 31)),
    }
    for name, path in tests.items():
        child = env.clone()
        before = child.frame()
        for action in path:
            child.step(*action)
        print(
            "T", name, frame_delta(before, child.frame())["bbox"],
            child.levels_completed - env.levels_completed, relevant(child),
        )
    moves = (
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
    staged = env.clone()
    for source, destination in moves:
        staged.step(6, source[1] + 1, source[0] + 1)
        staged.step(6, destination[1] + 1, destination[0] + 1)
    print("ONE-LOADED", relevant(staged))
    staged.step(6, 23, 43)
    staged.step(6, 35, 43)
    print("STAGED", staged.levels_completed - env.levels_completed, relevant(staged))
    for action in (1, 2, 3, 4):
        child = staged.clone()
        before = child.frame()
        child.step(action)
        print(
            "SA", action, frame_delta(before, child.frame())["bbox"],
            child.levels_completed - env.levels_completed, relevant(child),
        )
    travel = staged.clone()
    for index in range(1, 25):
        before = travel.frame()
        travel.step(4)
        delta = frame_delta(before, travel.frame())
        print(
            "R4", index, delta["bbox"],
            travel.levels_completed - env.levels_completed, relevant(travel),
        )
        if travel.levels_completed > env.levels_completed or delta["count"] <= 1:
            holes = tuple(
                blob.top_left
                for blob in connected_components(travel.frame(), colors=(1,))
                if blob.size == (4, 4) and blob.area == 16
            )
            print("STOP", holes, relevant(travel))
            symbols = {
                0: "0", 1: ".", 5: "#", 8: "B", 9: "|",
                10: " ", 11: "C", 12: "c", 14: "P",
            }
            frame = travel.frame()
            print("MAP")
            for row in range(6, 61, 6):
                print(
                    f"{row:02}",
                    "".join(
                        symbols.get(int(frame[row + 1][column + 1]), "?")
                        for column in range(4, 59, 6)
                    ),
                )
            for action in (1, 2, 3, 4):
                child = travel.clone()
                before = child.frame()
                child.step(action)
                print(
                    "STOP-A", action,
                    frame_delta(before, child.frame())["bbox"],
                    relevant(child),
                )
            traces = {
                "upper": (1, 1) + (3,) * 7 + (1,) * 5,
                "lower": (1, 3, 3, 2) + (3,) * 5 + (2,) * 4,
            }
            for name, path in traces.items():
                child = travel.clone()
                points = []
                for action in path:
                    child.step(action)
                    bridge = tuple(
                        blob.top_left
                        for blob in connected_components(
                            child.frame(), colors=(8,)
                        )
                        if blob.area >= 3
                    )
                    points.append((bridge, relevant(child)))
                print("TRACE", name, tuple(points), relevant(child))
            lower = travel.clone()
            lower_moves = (
                ((42, 28), (42, 40)),
                ((42, 34), (42, 46)),
                ((42, 40), (42, 52)),
                ((42, 46), (42, 58)),
                ((42, 52), (54, 52)),
            )
            for source, destination in lower_moves:
                lower.step(6, source[1] + 1, source[0] + 1)
                lower.step(6, destination[1] + 1, destination[0] + 1)
                print("LOWER-M", relevant(lower))
            print(
                "LOWER-DONE",
                lower.levels_completed - env.levels_completed,
                relevant(lower),
            )
            attachments = {
                "upper-adjacent": (
                    (1, 1, 4),
                    ((6, 29, 31), (6, 29, 25)),
                ),
                "lower-adjacent": (
                    (1, 3, 3, 2),
                    ((6, 47, 43), (6, 53, 43)),
                ),
                "lower-leap": (
                    (1, 3, 3, 2),
                    ((6, 47, 43), (6, 59, 43)),
                ),
            }
            for name, (keys, clicks) in attachments.items():
                child = travel.clone()
                for action in keys:
                    child.step(action)
                for click in clicks:
                    child.step(*click)
                print("ATTACH", name, relevant(child))
            aligned = travel.clone()
            align_path = (1, 4, 4, 3, 2, 4)
            align_trace = []
            for action in align_path:
                aligned.step(action)
                row_slots = tuple(
                    blob.top_left
                    for blob in connected_components(
                        aligned.frame(), colors=(1,)
                    )
                    if (
                        blob.size == (4, 4)
                        and blob.area == 16
                        and blob.top_left[0] == 42
                    )
                )
                align_trace.append((relevant(aligned), row_slots))
            print("ALIGN", tuple(align_trace))
            aligned.step(6, 29, 43)
            aligned.step(6, 41, 43)
            print("ALIGN-LEAP", relevant(aligned))
            for name, click in (
                ("peg", (6, 29, 43)),
                ("bridge", (6, 35, 43)),
            ):
                selected = travel.clone()
                selected.step(*click)
                print(
                    "SELECT", name, color_counts(selected.frame()),
                    relevant(selected),
                )
                for action in (1, 2, 3, 4, 6):
                    child = selected.clone()
                    before = child.frame()
                    child.step(action)
                    print(
                        "SELECT-A", name, action,
                        frame_delta(before, child.frame())["bbox"],
                        color_counts(child.frame()), relevant(child),
                    )
            for name, clicks in {
                "peg-to-lower": ((6, 29, 43), (6, 53, 43)),
                "bridge-to-lower": ((6, 35, 43), (6, 53, 43)),
                "peg-to-top": ((6, 29, 43), (6, 23, 19)),
                "bridge-to-top": ((6, 35, 43), (6, 23, 19)),
            }.items():
                child = travel.clone()
                for click in clicks:
                    child.step(*click)
                print("DIRECT", name, relevant(child))
            around = travel.clone()
            around_path = (
                (1, 1) + (3,) * 5 + (2,) * 6 + (4,) * 6 + (1,) * 3
            )
            around_trace = []
            for action in around_path:
                around.step(action)
                around_trace.append(relevant(around))
            print(
                "AROUND", around.levels_completed - env.levels_completed,
                tuple(around_trace),
            )
            crossed = travel.clone()
            cross_keys = (1, 1) + (3,) * 5 + (2,) * 6 + (4,) * 2
            for action in cross_keys:
                crossed.step(action)
            print("CROSS-READY", relevant(crossed))
            crossed.step(6, 53, 43)
            crossed.step(6, 53, 55)
            print("CROSSED", relevant(crossed))
            for action in (1, 2, 3, 4):
                child = crossed.clone()
                child.step(action)
                print("CROSS-A", action, relevant(child))
            break
