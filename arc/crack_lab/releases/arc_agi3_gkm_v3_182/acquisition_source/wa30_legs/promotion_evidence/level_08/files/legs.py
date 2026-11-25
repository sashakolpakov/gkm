"""Reusable frame-based manipulation skills."""
from collections import deque
from perception import arr, DIRS, block_signatures


def fill_marked_cells(env, cell=4, cooperative=False, pickup_directions=(1,),
                      target_signature=None, persistent_targets=False, delivery_pairs=None):
    """Carry yellow pieces onto red cells, checking each delivery.

    Pickup directions default to above; narrow passages may require a side
    pickup. An optional target signature selects outlined destination cells.
    The carried piece retains its pickup offset. Plan translations
    of both occupied cells, avoiding stationary pieces and arena boundaries.
    Optional delivery pairs select a short ordered assistance task and replan
    around moving carriers after every translation.
    Persistent targets retain obscured destinations and measure progress by
    cargo occupancy instead of the currently visible marker colors.
    In cooperative mode, verify a delivery followed by yielding to other
    carriers, and commit only an alternative that completes the level.
    """
    base_level = env.levels_completed

    def observe(e):
        f = arr(e.frame())
        tiles = {(r, c): set(map(int, f[r*cell:(r+1)*cell,
                                         c*cell:(c+1)*cell].ravel()))
                 for r in range((f.shape[0]-1)//cell)
                 for c in range(f.shape[1]//cell)}
        avatar = next(p for p, colors in tiles.items() if 14 in colors)
        pieces = {p for p, colors in tiles.items() if 9 in colors and
                  (4 in colors or 3 in colors)}
        marks = ({p for p, colors in tiles.items() if 2 in colors}
                 if target_signature is None else
                 {p for p, colors in tiles.items()
                  if tuple(sorted(colors)) == target_signature})
        floor = {p for p, colors in tiles.items()
                 if colors <= {0, 1, 2, 3, 4, 9, 14}}
        return avatar, pieces, marks, floor

    def route(start, goal, floor, obstacles, offset=None):
        q = deque([(start, [])])
        seen = {start}
        while q:
            p, path = q.popleft()
            if p == goal:
                return path
            for action, (dr, dc) in DIRS.items():
                n = (p[0]+dr, p[1]+dc)
                footprint = {n, (n[0]+offset[0], n[1]+offset[1])} if offset else {n}
                if n not in seen and footprint <= floor and not footprint & obstacles:
                    seen.add(n)
                    q.append((n, path+[action]))
        return None

    _, _, target_cells, _ = observe(env)
    delivery_index = 0
    while not env.terminal() and env.levels_completed == base_level:
        if delivery_pairs is not None and delivery_index >= len(delivery_pairs):
            return
        avatar, pieces, marks, floor = observe(env)
        if persistent_targets:
            marks = target_cells - pieces
        if not marks:
            return
        candidates = []
        for source in pieces - target_cells:
            if delivery_pairs is not None and source != delivery_pairs[delivery_index][0]:
                continue
            for facing in pickup_directions:
                offset = DIRS[facing]
                approach = (source[0]-offset[0], source[1]-offset[1])
                first = route(avatar, approach, floor, pieces)
                if first is None:
                    continue
                for target in marks:
                    if delivery_pairs is not None and target != delivery_pairs[delivery_index][1]:
                        continue
                    destination = (target[0]-offset[0], target[1]-offset[1])
                    second = route(approach, destination, floor,
                                   pieces-{source}, offset)
                    if second is not None:
                        candidates.append((len(first)+len(second), source, target,
                                           first+[facing, 5]+second+[5], facing))
        if not candidates:
            raise RuntimeError('No reachable piece delivery')
        if cooperative:
            # Other carriers change the board during a route. Verify complete
            # delivery-and-yield alternatives rather than trusting static paths.
            import time
            for _, source, target, delivery, _ in sorted(candidates):
                started = time.monotonic()
                clone = env.clone()
                path = []
                # Move clear after dropping; USE there becomes a safe wait.
                for action in delivery + [2, 2, 2] + [5] * 72:
                    if clone.terminal() or clone.levels_completed != base_level:
                        break
                    clone.step(action)
                    path.append(action)
                time.sleep(max(0, len(path) / 290 - (time.monotonic()-started)))
                if clone.levels_completed > base_level:
                    for action in path:
                        env.step(action)
                    return
            raise RuntimeError('No verified cooperative delivery')
        _, source, target, path, facing = min(candidates)
        # Verify the complete delivery on a clone before committing it.
        clone = env.clone()
        if delivery_pairs is not None:
            # Moving carriers can enter a planned route. Recompute each
            # translation from the new frame while retaining the cargo offset.
            import time
            path = []
            delivery_started = time.monotonic()

            def advance(action):
                clone.step(action)
                path.append(action)
                time.sleep(max(0, len(path)/280 -
                               (time.monotonic()-delivery_started)))

            def navigate(goal, offset=None):
                for _ in range(64):
                    position, cargo, _, accessible = observe(clone)
                    if position == goal:
                        return
                    moves = route(position, goal, accessible, cargo, offset)
                    if not moves:
                        raise RuntimeError('No current route for the selected delivery')
                    advance(moves[0])
                raise RuntimeError('Selected delivery exceeded navigation bound')

            offset = DIRS[facing]
            navigate((source[0]-offset[0], source[1]-offset[1]))
            advance(facing)
            advance(5)
            if (0, 9) not in block_signatures(clone.frame(), cell).values():
                raise RuntimeError('Selected cargo was not picked up')
            navigate((target[0]-offset[0], target[1]-offset[1]), offset)
            advance(5)
        else:
            for action in path:
                clone.step(action)
        if clone.levels_completed == base_level:
            _, after_pieces, remaining, _ = observe(clone)
            if persistent_targets:
                remaining = target_cells - after_pieces
            if target in remaining or len(remaining) >= len(marks):
                raise RuntimeError('Delivery failed to fill a marked cell')
        delivery_index += 1
        for action in path:
            env.step(action)
            if env.terminal() or env.levels_completed != base_level:
                return


def handoff_across_barrier(env, cell=4):
    """Stage side-carried pieces across a divider for an autonomous carrier.

    Keep the avatar on reachable floor, preserve the pickup offset while
    routing cargo, and verify collection and final delivery on a clone.
    """
    import time
    started = time.monotonic()
    base_level = env.levels_completed
    trial = env.clone()
    path = []

    def observe(e):
        f = arr(e.frame())
        tiles = {(r, c): set(map(int, f[r*cell:(r+1)*cell,
                                         c*cell:(c+1)*cell].ravel()))
                 for r in range((f.shape[0]-1)//cell)
                 for c in range(f.shape[1]//cell)}
        avatar = next(p for p, s in tiles.items() if 14 in s)
        pieces = {p for p, s in tiles.items() if 9 in s and 4 in s}
        barrier = {p for p, s in tiles.items() if s == {1, 2}}
        return avatar, pieces, barrier, set(tiles)

    def step(action):
        trial.step(action)
        path.append(action)
        time.sleep(max(0, len(path)/290 - (time.monotonic()-started)))

    def route(start, goal, floor, obstacles, offset=None):
        q = deque([(start, [])])
        seen = {start}
        while q:
            p, moves = q.popleft()
            if p == goal:
                return moves
            for action, (dr, dc) in DIRS.items():
                n = (p[0]+dr, p[1]+dc)
                cargo = (n[0]+offset[0], n[1]+offset[1]) if offset else n
                if (n not in seen and n in floor and n not in obstacles
                        and cargo in cells and cargo not in obstacles):
                    seen.add(n)
                    q.append((n, moves+[action]))
        return None

    avatar, pieces, barrier, cells = observe(trial)
    f = arr(trial.frame())
    goals = {p for p in cells if set(map(int, f[p[0]*cell:(p[0]+1)*cell,
                                                       p[1]*cell:(p[1]+1)*cell].ravel())) == {2, 9}}
    # Pieces can obscure divider cells; close its observed continuous lines.
    divider = {p for p in cells
               if (sum(q[0] == p[0] for q in barrier) >= 3
                   and any(q[0] == p[0] and q[1] <= p[1] for q in barrier)
                   and any(q[0] == p[0] and q[1] >= p[1] for q in barrier))
               or (sum(q[1] == p[1] for q in barrier) >= 3
                   and any(q[1] == p[1] and q[0] <= p[0] for q in barrier)
                   and any(q[1] == p[1] and q[0] >= p[0] for q in barrier))}
    # Infer accessible staging sides from the divider and each pickup offset.
    while not trial.terminal() and trial.levels_completed == base_level:
        avatar, pieces, barrier, cells = observe(trial)
        barrier = divider
        floor = cells - barrier
        candidates = []
        for source in pieces - goals - barrier:
            for facing, offset in DIRS.items():
                approach = (source[0]-offset[0], source[1]-offset[1])
                if approach not in floor:
                    continue
                first = route(avatar, approach, floor, pieces)
                if first is None:
                    continue
                for target in barrier - pieces:
                    destination = (target[0]-offset[0], target[1]-offset[1])
                    if destination not in floor:
                        continue
                    second = route(approach, destination, floor,
                                   pieces-{source}, offset)
                    if second is not None:
                        candidates.append((len(first)+len(second), source, target,
                                           destination, first+[facing, 5]+second+[5]))
        if not candidates:
            break
        _, source, target, destination, delivery = min(candidates)
        for action in delivery:
            step(action)
        # Leave pickup range before waiting so USE cannot undo the handoff.
        for action, (dr, dc) in DIRS.items():
            n = (destination[0]+dr, destination[1]+dc)
            if n in floor and n not in pieces and n != target:
                step(action)
                break
        for _ in range(90):
            if trial.terminal() or trial.levels_completed != base_level:
                break
            _, remaining, _, _ = observe(trial)
            if target not in remaining:
                break
            step(5)
        else:
            raise RuntimeError('Carrier did not collect the staged piece')
        if len(path) > 350:
            raise RuntimeError('Handoff progress exceeded its bound')
    for _ in range(90):
        if trial.terminal() or trial.levels_completed != base_level:
            break
        step(5)
    if trial.levels_completed <= base_level:
        raise RuntimeError('Handoffs did not complete the level')
    for action in path:
        env.step(action)


def deliver_with_moving_carrier(env, cell=4, deliveries=3, max_steps=125):
    """Deliver cargo while yielding contested passages to another carrier.

    Remember obscured goals, replan around new deliveries, and hand off cargo
    when two blocked translations reveal a carrier interception. Search bounded
    source orders on clones and commit only a complete, reward-verified plan.
    """
    import time
    from itertools import islice, permutations
    from perception import block_signatures

    base_level = env.levels_completed

    def tiles(e):
        return {p: s for p, s in block_signatures(e.frame(), cell).items()
                if p[0] < (len(e.frame())-1)//cell}

    def observe(e):
        t = tiles(e)
        avatar = next(p for p, s in t.items() if 14 in s)
        pieces = {p for p, s in t.items() if 9 in s and (4 in s or 3 in s)}
        return avatar, pieces, t

    home, pieces, initial = observe(env)
    goals = {p for p, s in initial.items() if 2 in s}
    floor = {p for p, s in initial.items() if 5 not in s}

    def route(start, goal, obstacles, carrying=False):
        queue = deque([(start, [])])
        seen = {start}
        while queue:
            p, path = queue.popleft()
            if p == goal:
                return path
            for action, (dr, dc) in DIRS.items():
                n = p[0]+dr, p[1]+dc
                footprint = {n, (n[0]-1, n[1])} if carrying else {n}
                if n not in seen and footprint <= floor and not footprint & obstacles:
                    seen.add(n)
                    queue.append((n, path+[action]))
        return None

    sources = sorted(p for p in pieces-goals
                     if route(home, (p[0]+1, p[1]), pieces) is not None)
    orders = permutations(sources, min(deliveries, len(sources)))
    for sources_order in islice(orders, 60):
        trial = env.clone()
        path = []
        started = time.monotonic()

        def step(action):
            if (trial.terminal() or trial.levels_completed != base_level
                    or len(path) >= max_steps):
                return False
            trial.step(action)
            path.append(action)
            time.sleep(max(0, len(path)/290-(time.monotonic()-started)))
            return True

        def navigate(goal):
            for _ in range(65):
                avatar, pieces, t = observe(trial)
                if avatar == goal:
                    return True
                obstacles = pieces | {p for p, s in t.items()
                                      if 12 in s or s == (5, 9)}
                moves = route(avatar, goal, obstacles)
                if moves:
                    if not step(moves[0]):
                        return False
                else:
                    waits = [a for a, (dr, dc) in DIRS.items()
                             if (avatar[0]+dr, avatar[1]+dc) not in floor]
                    if not waits or not step(waits[0]):
                        return False
            return False

        for i, source in enumerate(sources_order):
            if source not in observe(trial)[1]:
                continue
            if not navigate((source[0]+1, source[1])):
                break
            if not step(1) or not step(5) or (0, 9) not in tiles(trial).values():
                break
            handed_off = False
            stalls = 0
            for _ in range(65):
                avatar, pieces, _ = observe(trial)
                options = []
                for target in goals-pieces:
                    moves = route(avatar, (target[0]+1, target[1]), pieces, True)
                    if moves is not None:
                        options.append((len(moves), target, moves))
                if not options:
                    break
                _, target, moves = min(options)
                if not moves:
                    step(5)
                    break
                if not step(moves[0]):
                    break
                stalls = stalls+1 if observe(trial)[0] == avatar else 0
                if stalls >= 2:
                    step(5)
                    handed_off = True
                    break
            if i+1 == len(sources_order):
                avatar = observe(trial)[0]
                retreat = home if handed_off else (avatar[0]+3, avatar[1])
                if retreat not in floor or not navigate(retreat):
                    break
        while step(5):
            pass
        if trial.levels_completed > base_level:
            for action in path:
                env.step(action)
            return
    raise RuntimeError('No verified delivery and carrier cooperation plan')


def clear_interfering_carrier(env, carrier_color=15, cell=4, max_steps=60,
                              settle_turns=0, anticipate_motion=False):
    """Approach a carrier, face it, and USE; verify removal before committing.

    Replan after every turn because the carrier may transport cargo while the
    avatar approaches. Walls, loose cargo, and the carrier block walking.
    Motion anticipation searches frame-based interception paths on clones.
    Optionally wait for several unchanged carrier/cargo observations before
    approaching, preventing a moving carrier from evading the facing action.
    """
    import time
    from perception import block_signatures

    trial = env.clone()
    path = []
    started = time.monotonic()
    base_level = env.levels_completed

    def step(action):
        trial.step(action)
        path.append(action)
        time.sleep(max(0, len(path)/280 - (time.monotonic()-started)))

    if anticipate_motion:
        # A moving actor can leave between facing and USE. Search short
        # interception paths, retaining orientation and observed time in the
        # beam. Wall-respecting distance guides movement through narrow gaps.
        def observe(e):
            tiles = block_signatures(e.frame(), cell)
            avatar = next(p for p, sig in tiles.items() if 14 in sig)
            carriers = {p for p, sig in tiles.items() if carrier_color in sig}
            return avatar, carriers, tiles

        clone_steps = 0
        while observe(trial)[1]:
            count = len(observe(trial)[1])
            beam = [(trial.clone(), [], None)]
            found = None
            for _ in range(max_steps - len(path)):
                candidates = []
                seen = set()
                for state, prefix, facing in beam:
                    avatar, carriers, tiles = observe(state)
                    for action in (5, 1, 2, 3, 4):
                        if action == 5:
                            if facing not in DIRS:
                                continue
                            dr, dc = DIRS[facing]
                            if (avatar[0]+dr, avatar[1]+dc) not in carriers:
                                continue
                        child = state.clone()
                        child.step(action)
                        clone_steps += 1
                        time.sleep(max(0, clone_steps/280 -
                                       (time.monotonic()-started)))
                        position, remaining, tiles = observe(child)
                        moves = prefix + [action]
                        direction = action if action != 5 else facing
                        if len(remaining) < count:
                            found = child, moves
                            break
                        key = (position, direction, tuple(
                            (p, sig) for p, sig in tiles.items()
                            if sig not in ((1,), (5,), (1, 7))))
                        if key in seen:
                            continue
                        seen.add(key)
                        floor = {p for p, sig in tiles.items()
                                 if p[0] < (len(child.frame())-1)//cell
                                 and sig != (5,)}
                        queue = deque([(position, 0)])
                        visited = {position}
                        distance = 99
                        while queue:
                            p, length = queue.popleft()
                            if any(abs(p[0]-q[0])+abs(p[1]-q[1]) <= 1
                                   for q in remaining):
                                distance = length+1
                                break
                            for dr, dc in DIRS.values():
                                n = p[0]+dr, p[1]+dc
                                if n in floor and n not in visited:
                                    visited.add(n)
                                    queue.append((n, length+1))
                        ready = (direction in DIRS and
                                 (position[0]+DIRS[direction][0],
                                  position[1]+DIRS[direction][1]) in remaining)
                        candidates.append((distance-0.4*ready,
                                           child, moves, direction))
                    if found:
                        break
                if found:
                    break
                candidates.sort(key=lambda item: item[0])
                beam = [(state, moves, direction)
                        for _, state, moves, direction in candidates[:30]]
                if not beam:
                    break
            if found is None:
                raise RuntimeError('No verified carrier interception within bound')
            trial, moves = found
            path.extend(moves)
        for action in path:
            env.step(action)
        return

    if settle_turns:
        previous = None
        stable = 0
        while len(path) + 2 < max_steps:
            signature = tuple((p, s) for p, s in block_signatures(trial.frame(), cell).items()
                              if carrier_color in s or s == (5, 9))
            if not any(carrier_color in s for _, s in signature):
                break
            stable = stable + 1 if signature == previous else 0
            if stable >= settle_turns:
                break
            previous = signature
            step(5)
        else:
            raise RuntimeError('Carrier did not settle within its bound')

    while len(path) + 2 <= max_steps:
        t = {p: s for p, s in block_signatures(trial.frame(), cell).items()
             if p[0] < (len(trial.frame())-1)//cell}
        carriers = {p for p, s in t.items() if carrier_color in s}
        if not carriers:
            for action in path:
                env.step(action)
            return
        avatar = next(p for p, s in t.items() if 14 in s)
        for facing, (dr, dc) in DIRS.items():
            if (avatar[0]+dr, avatar[1]+dc) in carriers:
                step(facing)
                step(5)
                break
        else:
            floor = {p for p, s in t.items()
                     if 5 not in s and not (9 in s and (4 in s or 3 in s))}
            queue = deque([(avatar, [])])
            seen = {avatar}
            route = None
            while queue:
                p, moves = queue.popleft()
                if any((p[0]+dr, p[1]+dc) in carriers for dr, dc in DIRS.values()):
                    route = moves
                    break
                for action, (dr, dc) in DIRS.items():
                    n = p[0]+dr, p[1]+dc
                    if n in floor and n not in carriers and n not in seen:
                        seen.add(n)
                        queue.append((n, moves+[action]))
            if not route:
                raise RuntimeError('No reachable carrier interaction')
            step(route[0])
        if trial.terminal() or trial.levels_completed != base_level:
            raise RuntimeError('Carrier interaction changed the level unexpectedly')
    raise RuntimeError('Carrier removal did not verify within its bound')


def yield_until_complete(env, max_steps=60, action=1):
    """Yield turns to autonomous carriers; commit only a reward-verified wait."""
    import time
    base_level = env.levels_completed
    trial = env.clone()
    path = []
    started = time.monotonic()
    for _ in range(max_steps):
        if trial.terminal() or trial.levels_completed > base_level:
            break
        trial.step(action)
        path.append(action)
        time.sleep(max(0, len(path)/280 - (time.monotonic()-started)))
    if trial.levels_completed <= base_level:
        raise RuntimeError('Yielding did not complete the level')
    for action in path:
        env.step(action)
