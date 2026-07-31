# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.

from collections import deque
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
    slots, pegs = _peg_board(frame)
    slots |= {
        (blob.bbox[0] - 1, blob.bbox[1] - 1)
        for blob in connected_components(frame, colors=(1,))
        if blob.size == (2, 2) and blob.area == 4
    }
    carriers = frozenset(
        (
            (blob.bbox[0] - 1, blob.bbox[1] - 1)
            if blob.size == (2, 2) else blob.top_left
        )
        for blob in connected_components(frame, colors=(12,))
        if blob.size in ((2, 2), (4, 4))
    )
    bridges = frozenset(
        (blob.bbox[0] + 1, blob.bbox[1])
        for blob in connected_components(frame, colors=(15,))
        if blob.size == (4, 4)
    )
    borders = frozenset(
        blob.top_left
        for blob in connected_components(frame, colors=(11,), min_area=4)
    )
    selected_blobs = connected_components(frame, colors=(3,), min_area=4)
    selected = (
        (selected_blobs[0].bbox[0] + 1, selected_blobs[0].bbox[1] + 1)
        if selected_blobs else None
    )
    return (
        frozenset(slots), frozenset(pegs), carriers,
        bridges, borders, selected,
    )


def _bridge_carrier_moves(frame):
    """Return legal (kind, source, destination) peg moves."""
    slots, pegs, carriers, bridges, _, _ = _bridge_carrier_state(frame)
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


def solve_bridge_carrier_peg_solitaire(
        env, max_align_states=120, max_macros=40, max_search_states=0):
    """Solve peg boards joined by persistent bridges and a movable carrier."""
    base_level = env.levels_completed

    if max_search_states:
        def reconstruct(path):
            child = env.clone()
            for macro in path:
                for action in macro:
                    if isinstance(action, tuple):
                        child.step(*action)
                    else:
                        child.step(action)
            return child

        def search_key(child):
            return (
                int(child.levels_completed),
                _bridge_carrier_state(child.frame()),
            )

        queue = deque([()])
        seen = {search_key(env)}
        solution = None
        while queue and len(seen) < max_search_states:
            path = queue.popleft()
            if len(path) >= max_macros:
                continue
            parent = reconstruct(path)
            macros = [((action,), None) for action in (1, 2, 3, 4)]
            macros += [
                (
                    (
                        (6, source[1] + 1, source[0] + 1),
                        (6, destination[1] + 1, destination[0] + 1),
                    ),
                    kind,
                )
                for kind, source, destination
                in _bridge_carrier_moves(parent.frame())
            ]
            for macro, _ in macros:
                child_path = path + (macro,)
                child = reconstruct(child_path)
                if child.levels_completed > base_level:
                    solution = child_path
                    queue.clear()
                    break
                key = search_key(child)
                if key not in seen:
                    seen.add(key)
                    queue.append(child_path)
        if solution is None:
            return
        for macro in solution:
            for action in macro:
                if isinstance(action, tuple):
                    env.step(*action)
                else:
                    env.step(action)
        return

    node = env.clone()
    solution = []
    forbidden = set()

    def choose(frame, blocked):
        moves = _bridge_carrier_moves(frame)
        captures = [move for move in moves if move[0] == "capture"]
        if captures:
            return captures[0]
        bridges = [
            move for move in moves
            if (move[1], move[2]) not in blocked
        ]
        return bridges[0] if bridges else None

    def alignment_path(root, blocked):
        queue = deque([()])
        seen = {_bridge_carrier_state(root.frame())}
        while queue and len(seen) < max_align_states:
            path = queue.popleft()
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
                if choose(child.frame(), blocked) is not None:
                    return child_path
                if len(child_path) < 16:
                    queue.append(child_path)
        return None

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
