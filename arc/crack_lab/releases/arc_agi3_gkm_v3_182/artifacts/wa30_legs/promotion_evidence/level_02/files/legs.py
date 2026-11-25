"""Reusable frame-based manipulation skills."""
from collections import deque
from perception import arr, DIRS


def fill_marked_cells(env, cell=4, cooperative=False):
    """Carry yellow pieces from below onto red cells, checking each delivery.

    The carried piece retains its offset above the avatar. Plan translations
    of both occupied cells, avoiding stationary pieces and arena boundaries.
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
        marks = {p for p, colors in tiles.items() if 2 in colors}
        floor = {p for p, colors in tiles.items()
                 if colors <= {0, 1, 2, 3, 4, 9, 14}}
        return avatar, pieces, marks, floor

    def route(start, goal, floor, obstacles, carrying=False):
        q = deque([(start, [])])
        seen = {start}
        while q:
            p, path = q.popleft()
            if p == goal:
                return path
            for action, (dr, dc) in DIRS.items():
                n = (p[0]+dr, p[1]+dc)
                footprint = {n, (n[0]-1, n[1])} if carrying else {n}
                if n not in seen and footprint <= floor and not footprint & obstacles:
                    seen.add(n)
                    q.append((n, path+[action]))
        return None

    _, _, target_cells, _ = observe(env)
    while not env.terminal() and env.levels_completed == base_level:
        avatar, pieces, marks, floor = observe(env)
        if not marks:
            return
        candidates = []
        for source in pieces - target_cells:
            approach = (source[0]+1, source[1])
            first = route(avatar, approach, floor, pieces)
            if first is None:
                continue
            for target in marks:
                destination = (target[0]+1, target[1])
                second = route(approach, destination, floor, pieces-{source}, True)
                if second is not None:
                    candidates.append((len(first)+len(second), source, target,
                                       first+[1, 5]+second+[5]))
        if not candidates:
            raise RuntimeError('No reachable piece delivery')
        if cooperative:
            # Other carriers change the board during a route. Verify complete
            # delivery-and-yield alternatives rather than trusting static paths.
            import time
            for _, source, target, delivery in sorted(candidates):
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
        _, source, target, path = min(candidates)
        # Verify the complete delivery on a clone before committing it.
        clone = env.clone()
        for action in path:
            clone.step(action)
        if clone.levels_completed == base_level:
            _, _, remaining, _ = observe(clone)
            if target in remaining or len(remaining) >= len(marks):
                raise RuntimeError('Delivery failed to fill a marked cell')
        for action in path:
            env.step(action)
            if env.terminal() or env.levels_completed != base_level:
                return
