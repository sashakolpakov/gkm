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


def _movable_bridge_board(frame):
    """Return visible slots, carriers, movable bridges, and pegs."""
    blobs = connected_components(frame, colors=(1, 8, 12, 14))
    slots = {
        blob.top_left
        for blob in blobs
        if blob.color == 1 and blob.size == (4, 4) and blob.area == 16
    }
    carriers = {
        blob.top_left
        for blob in blobs
        if blob.color == 12 and blob.size == (4, 4)
    }
    bridges = {
        blob.top_left
        for blob in blobs
        if blob.color == 8 and blob.size == (4, 4)
    }
    pegs = {
        blob.top_left
        for blob in blobs
        if blob.color == 14 and blob.size == (4, 4)
    }
    return slots, carriers, bridges, pegs


def _movable_bridge_solution(frame, max_states=20000):
    """Solve a visible relay board containing one reusable movable bridge."""
    slots, carriers, bridges, pegs = _movable_bridge_board(frame)
    if len(bridges) != 1:
        return None
    start = (frozenset(pegs), next(iter(bridges)))
    queue = deque([(start, ())])
    seen = {start}
    destinations = slots | carriers
    while queue and len(seen) <= max_states:
        (state_pegs, bridge), path = queue.popleft()
        if len(state_pegs) == 1 and next(iter(state_pegs)) in carriers:
            return path
        occupied = state_pegs | {bridge}
        pieces = tuple(("peg", peg) for peg in sorted(state_pegs))
        pieces += (("bridge", bridge),)
        for kind, source in pieces:
            for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                midpoint = (source[0] + dr, source[1] + dc)
                destination = (source[0] + 2 * dr, source[1] + 2 * dc)
                if (
                    destination not in destinations
                    or destination in occupied
                    or midpoint not in occupied
                    or (kind == "bridge" and midpoint not in state_pegs)
                ):
                    continue
                child_pegs = set(state_pegs)
                child_bridge = bridge
                if kind == "bridge":
                    child_bridge = destination
                else:
                    child_pegs.remove(source)
                    child_pegs.add(destination)
                    if midpoint in child_pegs:
                        child_pegs.remove(midpoint)
                child = (frozenset(child_pegs), child_bridge)
                if child not in seen:
                    seen.add(child)
                    queue.append((child, path + ((source, destination),)))
    return None


def _play_lattice_moves(env, moves):
    for source, destination in moves:
        env.step(6, source[1] + 1, source[0] + 1)
        env.step(6, destination[1] + 1, destination[0] + 1)


def solve_wrapped_bridge_carrier_peg_solitaire(env):
    """Solve a long relay whose carrier rails reveal successive peg regions."""
    local_solution = _movable_bridge_solution(env.frame())
    if local_solution is None:
        return
    _play_lattice_moves(env, local_solution)

    _, carriers, bridges, _ = _movable_bridge_board(env.frame())
    if len(carriers) != 1 or len(bridges) != 1:
        return
    _play_lattice_moves(env, ((next(iter(bridges)), next(iter(carriers))),))

    for action in (4,) * 9 + (1, 1, 3, 3, 1, 1):
        env.step(action)

    _play_lattice_moves(env, (
        ((30, 28), (18, 28)),
        ((18, 28), (18, 40)),
        ((30, 46), (18, 46)),
        ((18, 40), (18, 52)),
        ((18, 2), (18, 14)),
        ((18, 8), (18, 20)),
        ((18, 14), (18, 26)),
        ((18, 20), (18, 32)),
        ((18, 26), (18, 38)),
        ((18, 32), (18, 44)),
        ((18, 38), (30, 38)),
    ))
    for action in (2, 2, 4, 4, 1, 1, 1, 1):
        env.step(action)
    _play_lattice_moves(env, (
        ((18, 44), (18, 56)),
        ((18, 56), (30, 56)),
    ))

    env.step(2)
    env.step(2)
    _play_lattice_moves(env, (((30, 56), (30, 44)),))
    for action in (2, 2, 3, 3, 1, 1):
        env.step(action)
    _play_lattice_moves(env, (
        ((30, 44), (30, 32)),
        ((30, 32), (42, 32)),
    ))

    env.step(2)
    env.step(2)
    _play_lattice_moves(env, (
        ((42, 38), (42, 26)),
        ((42, 32), (42, 20)),
        ((42, 20), (54, 20)),
    ))
