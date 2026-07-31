"""Build a floor-only model, find a five-piece connected formation, replay it."""
import sys
from collections import deque
from heapq import heappop, heappush

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from legs import MIRRORED_PAIR_ASCENT, MIRRORED_PAIR_MAZE_REUNION
from perception import connected_components

DIRS = {1: (-4, 0), 2: (4, 0), 3: (0, -4), 4: (0, 4)}


def shortest_positions(floor, start, size):
    q = deque([start])
    paths = {start: ()}
    while q:
        pos = q.popleft()
        for action, (dr, dc) in DIRS.items():
            nxt = (pos[0] + dr, pos[1] + dc)
            r, c = nxt
            if (nxt not in paths and 0 <= r <= 64 - size and 0 <= c <= 64 - size
                    and np.all(floor[r:r + size, c:c + size] == 5)):
                paths[nxt] = paths[pos] + (action,)
                q.append(nxt)
    return paths


def center(pos, size):
    return (pos[0] + (size - 1) / 2, pos[1] + (size - 1) / 2)


def adjacent(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 4


def connected(points):
    seen = {0}
    q = [0]
    while q:
        i = q.pop()
        for j in range(len(points)):
            if j not in seen and adjacent(points[i], points[j]):
                seen.add(j)
                q.append(j)
    return len(seen) == len(points)


def cell_center(pos, size):
    return (pos[0] + (size - 1) / 2, pos[1] + (size - 1) / 2)


def occupied_center(pos, size):
    p = cell_center(pos, size)
    return (int(p[0] * 2), int(p[1] * 2))


def heuristic(state):
    positions = state[:5]
    sizes = (4, 4, 2, 2, 2)
    pts = [cell_center(p, s) for p, s in zip(positions, sizes)]
    # Dense progress: remaining grid-cell gaps in a minimum spanning tree.
    used = {0}
    cost = 0
    while len(used) < 5:
        edge = min(
            (max(0, int((abs(pts[i][0] - pts[j][0])
                         + abs(pts[i][1] - pts[j][1])) / 4) - 1), j)
            for i in used for j in range(5) if j not in used
        )
        cost += edge[0]
        used.add(edge[1])
    return cost


def inspect(env):
    for action in MIRRORED_PAIR_ASCENT + MIRRORED_PAIR_MAZE_REUNION:
        env.step(action)
    frame = np.asarray(env.frame())
    movable = connected_components(frame, colors=(9, 10), min_area=2)
    large = [b.top_left for b in movable if b.color == 10]
    small = [b.top_left for b in movable if b.color == 9]
    floor = frame.copy()
    floor[np.isin(floor, (9, 10))] = 5

    reaches = [shortest_positions(floor, p, 2) for p in small]
    print("small_reach", [len(p) for p in reaches])

    start = tuple(large + small) + (0,)
    sizes = (4, 4, 2, 2, 2)

    def passable(pos, size):
        r, c = pos
        return (0 <= r <= 64 - size and 0 <= c <= 64 - size
                and np.all(floor[r:r + size, c:c + size] == 5))

    def move(state, action):
        pos = list(state[:5])
        mode = state[5]
        occupied = [occupied_center(p, s) for p, s in zip(pos, sizes)]
        indices = (0, 1) if mode == 0 else (mode + 1,)
        proposed = {}
        for i in indices:
            dr, dc = DIRS[action]
            if mode == 0 and i == 1 and action in (3, 4):
                dc = -dc
            target = (pos[i][0] + dr, pos[i][1] + dc)
            tc = occupied_center(target, sizes[i])
            other = {occupied[j] for j in range(5) if j not in indices}
            if passable(target, sizes[i]) and tc not in other:
                proposed[i] = target
            else:
                proposed[i] = pos[i]
        if len(indices) == 2:
            centers = [occupied_center(proposed[i], sizes[i]) for i in indices]
            if centers[0] == centers[1]:
                proposed = {i: pos[i] for i in indices}
        for i, target in proposed.items():
            pos[i] = target
        return tuple(pos) + (mode,)

    def is_goal(state):
        pts = [cell_center(p, s) for p, s in zip(state[:5], sizes)]
        return connected(pts)

    # Establish whether any one small group can geometrically bridge the pair.
    pair_q = deque([(large[0], large[1])])
    pair_seen = {(large[0], large[1]): ()}
    bridges = []
    while pair_q:
        pair = pair_q.popleft()
        template = tuple(pair) + tuple(small) + (0,)
        for marker_index, paths in enumerate(reaches):
            for marker_pos in paths:
                pts = [cell_center(pair[0], 4), cell_center(pair[1], 4),
                       cell_center(marker_pos, 2)]
                if connected(pts):
                    bridges.append((pair, marker_index, marker_pos,
                                    pair_seen[pair], paths[marker_pos]))
                    break
        for action in (1, 2, 3, 4):
            nxt_state = move(template, action)
            nxt = nxt_state[:2]
            if nxt not in pair_seen:
                pair_seen[nxt] = pair_seen[pair] + (action,)
                pair_q.append(nxt)
    print("pair_states", len(pair_seen), "bridges", bridges[:3])

    marker_nodes = list(reaches[0])
    marker_centers = {p: cell_center(p, 2) for p in marker_nodes}
    marker_edges = {
        p: [q for q in marker_nodes if adjacent(marker_centers[p], marker_centers[q])]
        for p in marker_nodes
    }
    chain = None
    for pair, pair_route in sorted(pair_seen.items(), key=lambda item: len(item[1])):
        left, right = [cell_center(p, 4) for p in pair]
        starts = [p for p in marker_nodes if adjacent(left, marker_centers[p])]
        goals = {p for p in marker_nodes if adjacent(right, marker_centers[p])}
        q = deque((p, (p,)) for p in starts)
        chain_seen = set(starts)
        while q:
            node, path = q.popleft()
            if node in goals and len(path) <= 3:
                chain = (pair, path, pair_route)
                break
            if len(path) >= 3:
                continue
            for nxt in marker_edges[node]:
                if nxt not in chain_seen:
                    chain_seen.add(nxt)
                    q.append((nxt, path + (nxt,)))
        if chain:
            break
    print("three_marker_chain", chain)

    # Remove the three selectable blockers and test whether the mirrored pair
    # can then reunite; this distinguishes bridge pieces from movable gates.
    free_q = deque([((large[0], large[1]), ())])
    free_seen = {(large[0], large[1])}
    free_reunion = None
    while free_q:
        pair, route = free_q.popleft()
        a, b = [cell_center(p, 4) for p in pair]
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) in (0, 4):
            free_reunion = (pair, route)
            break
        for action in (1, 2, 3, 4):
            out = []
            for i, pos in enumerate(pair):
                dr, dc = DIRS[action]
                if i == 1 and action in (3, 4):
                    dc = -dc
                target = (pos[0] + dr, pos[1] + dc)
                out.append(target if passable(target, 4) else pos)
            if occupied_center(out[0], 4) == occupied_center(out[1], 4):
                out = list(pair)
            nxt = tuple(out)
            if nxt not in free_seen:
                free_seen.add(nxt)
                free_q.append((nxt, route + (action,)))
    print("pair_without_markers", len(free_seen), free_reunion)
    if free_reunion:
        pair = (large[0], large[1])
        blocker_events = []
        pair_visited = {occupied_center(p, 4) for p in pair}
        marker_centers_initial = [occupied_center(p, 2) for p in small]
        for step, action in enumerate(free_reunion[1], 1):
            out = []
            for i, pos in enumerate(pair):
                dr, dc = DIRS[action]
                if i == 1 and action in (3, 4):
                    dc = -dc
                target = (pos[0] + dr, pos[1] + dc)
                if passable(target, 4):
                    if occupied_center(target, 4) in marker_centers_initial:
                        blocker_events.append(
                            (step, action, i, target,
                             marker_centers_initial.index(occupied_center(target, 4)))
                        )
                    out.append(target)
                else:
                    out.append(pos)
            pair = tuple(out)
            pair_visited.update(occupied_center(p, 4) for p in pair)
        print("blocker_events", blocker_events)
        parking = []
        for marker, paths in zip(small, reaches):
            options = sorted(
                ((route, target) for target, route in paths.items()
                 if occupied_center(target, 2) not in pair_visited),
                key=lambda item: len(item[0]),
            )
            parking.append(options)
        print("parking", [options[:3] for options in parking])
        if all(parking):
            test = env.clone()
            candidate = []
            parking_choice = (0, 1, 0)
            for marker_index, (marker, options) in enumerate(zip(small, parking)):
                click = (6, marker[1], marker[0])
                candidate.append(click)
                test.step(*click)
                for action in options[parking_choice[marker_index]][0]:
                    candidate.append(action)
                    test.step(action)
            click_large = (6, large[0][1], large[0][0])
            candidate.append(click_large)
            test.step(*click_large)
            for action in free_reunion[1]:
                candidate.append(action)
                test.step(action)
            print("parking_replay", test.levels_completed, candidate)
            print("parking_pieces", [
                (b.color, b.bbox) for b in connected_components(
                    test.frame(), colors=(1, 9, 10, 11), min_area=2
                )
            ])
            reunion_test = test.clone()
            reunion_candidate = list(candidate)
            current_markers = [
                options[parking_choice[i]][1] for i, options in enumerate(parking)
            ]
            targets = [(15, 31), (15, 35), (11, 35)]

            def graph_route(start_pos, target_pos, blocked):
                q = deque([start_pos])
                parents = {start_pos: None}
                while q:
                    node = q.popleft()
                    if node == target_pos:
                        route = []
                        while parents[node] is not None:
                            prev = parents[node]
                            dr, dc = node[0] - prev[0], node[1] - prev[1]
                            route.append(next(a for a, delta in DIRS.items()
                                              if delta == (dr, dc)))
                            node = prev
                        return tuple(reversed(route))
                    for nxt in marker_edges[node]:
                        if nxt not in parents and occupied_center(nxt, 2) not in blocked:
                            parents[nxt] = node
                            q.append(nxt)
                return None

            large_positions_final = free_reunion[0]
            large_centers = {
                occupied_center(p, 4) for p in large_positions_final
            }
            assembly_routes = []
            for i, target in enumerate(targets):
                blocked = set(large_centers)
                blocked.update(
                    occupied_center(pos, 2)
                    for j, pos in enumerate(current_markers) if j != i
                )
                route = graph_route(current_markers[i], target, blocked)
                assembly_routes.append(route)
                click = (6, current_markers[i][1], current_markers[i][0])
                candidate.append(click)
                test.step(*click)
                for action in route or ():
                    candidate.append(action)
                    test.step(action)
                current_markers[i] = target
            print("assembly_routes", assembly_routes)
            print("assembly_replay", test.levels_completed, candidate)
            print("assembly_pieces", [
                (b.color, b.bbox) for b in connected_components(
                    test.frame(), colors=(1, 9, 10, 11), min_area=2
                )
            ])

            # Jointly route the three small pieces around the now-stationary
            # reunited pair; single-piece shortest paths can block each other.
            marker_start = tuple(
                options[parking_choice[i]][1] for i, options in enumerate(parking)
            )

            def marker_goal(state):
                pts = [cell_center(p, 4) for p in large_positions_final]
                pts.extend(cell_center(p, 2) for p in state)
                return connected(pts)

            def marker_h(state):
                pts = [cell_center(p, 4) for p in large_positions_final]
                pts.extend(cell_center(p, 2) for p in state)
                used = {0}
                score = 0
                while len(used) < 5:
                    gap, j = min(
                        (max(0, int((abs(pts[i][0] - pts[k][0])
                                     + abs(pts[i][1] - pts[k][1])) / 4) - 1), k)
                        for i in used for k in range(5) if k not in used
                    )
                    score += gap
                    used.add(j)
                return score

            mh = [(2 * marker_h(marker_start), 0, marker_start)]
            mc = {marker_start: 0}
            mp = {marker_start: None}
            ma = {}
            marker_goal_state = None
            marker_expanded = 0
            while mh and marker_expanded < 500000:
                _, cost, state = heappop(mh)
                if cost != mc[state]:
                    continue
                marker_expanded += 1
                if marker_goal(state):
                    marker_goal_state = state
                    break
                occupied = {occupied_center(p, 2) for p in state} | large_centers
                for i, pos in enumerate(state):
                    for nxt in marker_edges[pos]:
                        nc = occupied_center(nxt, 2)
                        if nc in occupied:
                            continue
                        child = list(state)
                        child[i] = nxt
                        child = tuple(child)
                        new_cost = cost + 1
                        if new_cost < mc.get(child, 10 ** 9):
                            mc[child] = new_cost
                            mp[child] = state
                            dr, dc = nxt[0] - pos[0], nxt[1] - pos[1]
                            ma[child] = (i, next(a for a, d in DIRS.items()
                                                 if d == (dr, dc)))
                            heappush(mh, (new_cost + 2 * marker_h(child),
                                          new_cost, child))
            marker_moves = []
            cursor = marker_goal_state
            while cursor is not None and mp[cursor] is not None:
                marker_moves.append(ma[cursor])
                cursor = mp[cursor]
            marker_moves.reverse()
            print("joint_marker_search", marker_expanded, marker_goal_state,
                  len(marker_moves))

            # Continue from the interaction-tested staging clone.
            live_positions = list(marker_start)
            active = None
            for marker_index, action in marker_moves:
                if active != marker_index:
                    click = (6, live_positions[marker_index][1],
                             live_positions[marker_index][0])
                    candidate.append(click)
                    test.step(*click)
                    active = marker_index
                dr, dc = DIRS[action]
                live_positions[marker_index] = (
                    live_positions[marker_index][0] + dr,
                    live_positions[marker_index][1] + dc,
                )
                candidate.append(action)
                test.step(action)
                if test.levels_completed > 2:
                    break
            print("joint_marker_replay", test.levels_completed, candidate)

    heap = [(heuristic(start), 0, start)]
    costs = {start: 0}
    parents = {start: None}
    parent_actions = {}
    goal = None
    expanded = 0
    while heap and expanded < 250000:
        _, cost, state = heappop(heap)
        if cost != costs[state]:
            continue
        expanded += 1
        if is_goal(state):
            goal = state
            break
        successors = []
        for action in (1, 2, 3, 4):
            successors.append((move(state, action), action))
        for mode in range(4):
            if mode != state[5]:
                nxt = state[:5] + (mode,)
                p = state[mode + 1] if mode else state[0]
                successors.append((nxt, (6, p[1], p[0])))
        for nxt, action in successors:
            if nxt == state:
                continue
            new_cost = cost + 1
            if new_cost < costs.get(nxt, 10 ** 9):
                costs[nxt] = new_cost
                parents[nxt] = state
                parent_actions[nxt] = action
                heappush(heap, (new_cost + 2 * heuristic(nxt), new_cost, nxt))
    replay = []
    cursor = goal
    while cursor is not None and parents[cursor] is not None:
        replay.append(parent_actions[cursor])
        cursor = parents[cursor]
    replay.reverse()
    print("symbolic_search", "expanded", expanded, "cost", len(replay),
          "formation", goal, "dense", heuristic(goal) if goal else None)
    if not goal:
        return
    test = env.clone()
    for action in replay:
        if isinstance(action, tuple):
            test.step(*action)
        else:
            test.step(action)
    print("replay_level", test.levels_completed, "actions", replay)
    print("replay_pieces", [
        (b.color, b.bbox) for b in connected_components(
            test.frame(), colors=(1, 9, 10, 11), min_area=2
        )
    ])


if __name__ == "__main__":
    A.run_program("m0r0", inspect)
