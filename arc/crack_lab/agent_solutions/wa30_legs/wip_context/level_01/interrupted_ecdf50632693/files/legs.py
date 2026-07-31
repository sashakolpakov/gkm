# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.

import heapq
import itertools

import numpy as np

from perception import DIRS, DOWN, LEFT, RIGHT, UP, USE, connected_components


def plan_rimmed_blocks_to_targets(frame, max_states=250_000):
    """Plan the verified attach-and-carry mechanic on a 4-pixel cell grid.

    A carryable block has a uniform 4/3/0 rim and a 2x2 colour-9 centre.
    Target cells contain colour 2. Colour-7 cells are barriers. The returned
    path faces a block, attaches with USE, translates the rigid pair, and
    releases every block on a distinct target.
    """
    grid = np.asarray(frame)
    cell = 4
    rows, cols = grid.shape[0] // cell, grid.shape[1] // cell

    orange = connected_components(grid, colors=[14], min_area=1)
    if len(orange) != 1:
        return None
    player = (orange[0].bbox[0] // cell, orange[0].bbox[1] // cell)

    player_tile = grid[
        player[0] * cell : (player[0] + 1) * cell,
        player[1] * cell : (player[1] + 1) * cell,
    ]
    edge_zeros = {
        UP: int(np.count_nonzero(player_tile[0, :] == 0)),
        DOWN: int(np.count_nonzero(player_tile[-1, :] == 0)),
        LEFT: int(np.count_nonzero(player_tile[:, 0] == 0)),
        RIGHT: int(np.count_nonzero(player_tile[:, -1] == 0)),
    }
    facing = max(edge_zeros, key=edge_zeros.get)

    blocks = []
    targets = set()
    barriers = set()
    for row in range(rows):
        for col in range(cols):
            tile = grid[
                row * cell : (row + 1) * cell,
                col * cell : (col + 1) * cell,
            ]
            if np.any(tile == 2):
                targets.add((row, col))
            if np.any(tile == 7):
                barriers.add((row, col))
            rim = np.concatenate(
                (tile[0, :], tile[-1, :], tile[1:-1, 0], tile[1:-1, -1])
            )
            if (
                np.all(tile[1:3, 1:3] == 9)
                and len(np.unique(rim)) == 1
                and int(rim[0]) in (0, 3, 4)
            ):
                blocks.append((row, col))

    if not blocks or len(targets) < len(blocks):
        return None

    start = (player, facing, False, tuple(sorted(blocks)))
    target_order = tuple(sorted(targets))

    def heuristic(state):
        position, face, attached, packed_blocks = state
        assignment_distance = min(
            sum(
                abs(block[0] - target[0]) + abs(block[1] - target[1])
                for block, target in zip(packed_blocks, assignment)
            )
            for assignment in itertools.permutations(
                target_order, len(packed_blocks)
            )
        )
        unplaced = [block for block in packed_blocks if block not in targets]
        if attached:
            delta = DIRS[face]
            carried = (position[0] + delta[0], position[1] + delta[1])
            other_unplaced = len(unplaced) - int(carried in unplaced)
            use_cost = 1 + 2 * other_unplaced
            approach_cost = 0
        else:
            use_cost = 2 * len(unplaced)
            approach_cost = min(
                (
                    max(
                        0,
                        abs(position[0] - block[0])
                        + abs(position[1] - block[1])
                        - 1,
                    )
                    for block in unplaced
                ),
                default=0,
            )
        return assignment_distance + use_cost + approach_cost

    serial = itertools.count()
    queue = [(3 * heuristic(start), 0, next(serial), start)]
    parent = {start: None}
    parent_action = {}
    cost = {start: 0}
    actions = (UP, DOWN, LEFT, RIGHT, USE)

    def open_cell(position):
        row, col = position
        return 0 <= row < rows and 0 <= col < cols and position not in barriers

    def transition(state, action):
        position, face, attached, packed_blocks = state
        occupied = set(packed_blocks)

        if attached:
            face_delta = DIRS[face]
            carried = (
                position[0] + face_delta[0],
                position[1] + face_delta[1],
            )
            if carried not in occupied:
                return state
            if action == USE:
                return (position, face, False, packed_blocks)

            delta = DIRS[action]
            next_player = (position[0] + delta[0], position[1] + delta[1])
            next_carried = (carried[0] + delta[0], carried[1] + delta[1])
            other_blocks = occupied - {carried}
            if (
                not open_cell(next_player)
                or not open_cell(next_carried)
                or next_player in other_blocks
                or next_carried in other_blocks
            ):
                return state
            moved_blocks = tuple(sorted(other_blocks | {next_carried}))
            return (next_player, face, True, moved_blocks)

        if action == USE:
            delta = DIRS[face]
            faced_cell = (position[0] + delta[0], position[1] + delta[1])
            if faced_cell in occupied:
                return (position, face, True, packed_blocks)
            return state

        delta = DIRS[action]
        destination = (position[0] + delta[0], position[1] + delta[1])
        if open_cell(destination) and destination not in occupied:
            position = destination
        return (position, action, False, packed_blocks)

    goal = None
    while queue and len(parent) <= max_states:
        _, path_cost, _, state = heapq.heappop(queue)
        if path_cost != cost[state]:
            continue
        position, face, attached, packed_blocks = state
        if not attached and set(packed_blocks).issubset(targets):
            goal = state
            break
        for action in actions:
            child = transition(state, action)
            next_cost = path_cost + 1
            if child == state or next_cost >= cost.get(child, max_states + 1):
                continue
            parent[child] = state
            parent_action[child] = action
            cost[child] = next_cost
            priority = next_cost + 3 * heuristic(child)
            heapq.heappush(
                queue, (priority, next_cost, next(serial), child)
            )

    if goal is None:
        return None

    path = []
    state = goal
    while parent[state] is not None:
        path.append(parent_action[state])
        state = parent[state]
    path.reverse()
    return path


def place_rimmed_blocks_on_targets(env):
    """Execute a compact symbolic carry plan and stop at level completion."""
    path = plan_rimmed_blocks_to_targets(env.frame())
    if path is None:
        return False
    base_level = int(env.levels_completed)
    for action in path:
        if env.terminal() or env.levels_completed > base_level:
            break
        env.step(action)
    return env.levels_completed > base_level
