# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.

from collections import deque
from heapq import heappop, heappush
from math import gcd

from perception import arr, bounded_replay_bfs, connected_components


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
        frame, colors=(1, 3, 8, 11, 12, 14, 15)
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
    bridges |= {
        blob.top_left
        for blob in blobs
        if blob.color == 8 and blob.size == (4, 4)
    }
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


def solve_movable_bridge_peg_solitaire(
        env, max_states=120000, max_depth=60,
        max_navigation_states=6000, max_navigation_depth=100):
    """Solve peg solitaire where a persistent bridge can leapfrog with pegs."""
    base_level = env.levels_completed
    slots, start_pegs, carriers, start_bridges, _, _ = (
        _bridge_carrier_state(env.frame())
    )
    destinations = slots | carriers
    start = (start_pegs, start_bridges)
    serial = 0
    queue = [(len(start_pegs), 0, serial, (), start)]
    seen = {start}
    while queue and len(seen) < max_states:
        _, _, _, path, state = heappop(queue)
        if len(path) >= max_depth:
            continue
        pegs, bridges = state
        if len(pegs) == 1 and pegs & carriers:
            clone = env.clone()
            for action in path:
                clone.step(*action)
            if clone.levels_completed > base_level:
                for action in path:
                    env.step(*action)
                return
            loaded_state = _bridge_carrier_state(clone.frame())
            loaded_macro = None
            loaded_pegs = loaded_state[1]
            for source in loaded_state[3]:
                for destination in loaded_state[2]:
                    midpoint = (
                        (source[0] + destination[0]) // 2,
                        (source[1] + destination[1]) // 2,
                    )
                    if (
                        midpoint in loaded_pegs
                        and (
                            abs(source[0] - destination[0])
                            + abs(source[1] - destination[1])
                        ) == 2 * _lattice_step(loaded_state[0] | loaded_state[2])
                    ):
                        loaded_macro = (
                            (6, source[1] + 1, source[0] + 1),
                            (6, destination[1] + 1, destination[0] + 1),
                        )
                        break
                if loaded_macro:
                    break
            if loaded_macro is None:
                continue
            for action in loaded_macro:
                clone.step(*action)
            staged_path = path + loaded_macro
            if clone.levels_completed > base_level:
                for action in staged_path:
                    env.step(*action)
                return

            key_path = bounded_replay_bfs(
                clone,
                goal_fn=lambda node, _: node.levels_completed > base_level,
                action_fn=lambda _: (1, 2, 3, 4),
                key_fn=lambda node: arr(node.frame())[1:].tobytes(),
                max_states=max_navigation_states,
                max_depth=max_navigation_depth,
            )
            if key_path is not None:
                solution = staged_path + tuple((action,) for action in key_path)
                for replay_action in solution:
                    env.step(*replay_action)
                return
            continue
        step = _lattice_step(destinations)
        occupied = pegs | bridges
        for source in sorted(occupied):
            for dr, dc in (
                    (-step, 0), (step, 0), (0, -step), (0, step)):
                midpoint = (source[0] + dr, source[1] + dc)
                destination = (source[0] + 2 * dr, source[1] + 2 * dc)
                if (
                    midpoint not in occupied
                    or destination not in destinations
                    or destination in occupied
                ):
                    continue
                next_pegs = set(pegs)
                next_bridges = set(bridges)
                if source in bridges:
                    next_bridges.remove(source)
                    next_bridges.add(destination)
                else:
                    next_pegs.remove(source)
                    next_pegs.add(destination)
                    if midpoint in pegs:
                        next_pegs.remove(midpoint)
                child_state = (
                    frozenset(next_pegs), frozenset(next_bridges)
                )
                if child_state in seen:
                    continue
                seen.add(child_state)
                macro = (
                    (6, source[1] + 1, source[0] + 1),
                    (6, destination[1] + 1, destination[0] + 1),
                )
                child_path = path + macro
                serial += 1
                heappush(queue, (
                    len(next_pegs), len(child_path), serial,
                    child_path, child_state,
                ))


def observe_level(env):
    """Print a compact symbolic frame and one-step cloned action deltas."""
    frame = arr(env.frame())
    print("bridge_state", _bridge_carrier_state(frame))
    print("bridge_moves", _bridge_carrier_moves(frame))
    state = _bridge_carrier_state(frame)
    click_moves = []
    for source in sorted(state[1] | state[3]):
        for destination in sorted(state[0] | state[2]):
            if source == destination:
                continue
            clone = env.clone()
            clone.step(6, source[1] + 1, source[0] + 1)
            clone.step(6, destination[1] + 1, destination[0] + 1)
            after_state = _bridge_carrier_state(clone.frame())
            if after_state[:4] != state[:4]:
                click_moves.append((
                    source, destination,
                    sorted(after_state[1]), sorted(after_state[3]),
                    clone.levels_completed - env.levels_completed,
                ))
    print("observed_click_moves", click_moves)
    symbols = {1: ".", 5: "5", 9: "9", 10: "A", 14: "E"}
    print("actions", env.actions)
    print("counts", {
        int(color): int((frame == color).sum())
        for color in sorted(set(frame.ravel()))
    })
    for r in range(0, 64, 4):
        row = []
        for c in range(0, 64, 4):
            block = frame[r:r + 4, c:c + 4]
            values, counts = __import__("numpy").unique(block, return_counts=True)
            value = int(values[counts.argmax()])
            row.append(symbols.get(value, format(value, "X")))
        print("".join(row))
    print("lattice6")
    for r in range(12, 61, 6):
        row = []
        for c in range(12, 61, 6):
            block = frame[r:r + 4, c:c + 4]
            values, counts = __import__("numpy").unique(block, return_counts=True)
            value = int(values[counts.argmax()])
            row.append({0: "o", 8: "X", 12: "C", 14: "P"}.get(
                value, symbols.get(value, format(value, "X"))
            ))
        print(r, "".join(row))
    print("blobs", [
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, min_area=3)
        if blob.color != 1
    ])
    for count in range(1, 13):
        clone = env.clone()
        for _ in range(count):
            clone.step(4)
        carriers = [
            blob.bbox
            for blob in connected_components(clone.frame(), colors=(12,))
            if blob.area >= 3
        ]
        print("rights", count, carriers)
    for path in (
        (1,), (2,), (3,), (4,), (6,),
        (4, 1), (4, 2), (4, 3), (4, 4), (4, 6),
        (4, 4, 4), (4, 4, 2), (4, 4, 6),
        ((6, 19, 19),), ((6, 19, 25),),
        ((6, 13, 49),), ((6, 25, 55),),
        ((6, 19, 25), (6, 19, 13)),
        ((6, 19, 25), (6, 19, 37)),
        ((6, 13, 49), (6, 25, 49)),
        ((6, 25, 55), (6, 25, 43)),
    ):
        clone = env.clone()
        before = arr(clone.frame()).copy()
        for action in path:
            clone.step(*action) if isinstance(action, tuple) else clone.step(action)
        after = arr(clone.frame())
        changed = __import__("numpy").argwhere(before != after)
        bbox = None if not len(changed) else (
            int(changed[:, 0].min()), int(changed[:, 1].min()),
            int(changed[:, 0].max()), int(changed[:, 1].max()),
        )
        transitions = {}
        for r, c in changed:
            pair = (int(before[r, c]), int(after[r, c]))
            transitions[pair] = transitions.get(pair, 0) + 1
        print(
            "path", path, "changed", len(changed), "bbox", bbox,
            "trans", sorted(transitions.items()),
            "reward", clone.levels_completed - env.levels_completed,
            "parts", [
                (blob.color, blob.bbox, blob.area)
                for blob in connected_components(after, colors=(2, 3, 8, 11, 12, 14))
                if blob.area >= 3
            ],
        )
