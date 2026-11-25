"""Small reusable skills built from public frame observations and actions."""
from functools import lru_cache

from perception import connected_components, safe_step


def solve_peg_jumps(env):
    """Read square sockets, plan orthogonal peg captures, verify, then execute."""
    def observe(arena):
        blobs = connected_components(arena.frame(), colors=(1, 14))
        sockets = {
            (b.bbox[1] + 2, b.bbox[0] + 2)
            for b in blobs
            if b.size == (4, 4) and b.area in (12, 16)
        }
        pegs = {
            (b.bbox[1] + 2, b.bbox[0] + 2)
            for b in blobs if b.color == 14 and b.size == (4, 4)
        }
        return sockets, frozenset(pegs)

    board, initial = observe(env)
    pitch = min(
        abs(a[axis] - b[axis])
        for a in board for b in board for axis in (0, 1)
        if a[axis] != b[axis]
    )
    jumps = []
    for x, y in sorted(board):
        for dx, dy in ((pitch, 0), (-pitch, 0), (0, pitch), (0, -pitch)):
            middle = (x + dx, y + dy)
            destination = (x + 2 * dx, y + 2 * dy)
            if middle in board and destination in board:
                jumps.append(((x, y), middle, destination))

    @lru_cache(None)
    def plan(pegs):
        if len(pegs) == 1:
            return ()
        for source, middle, destination in jumps:
            if source in pegs and middle in pegs and destination not in pegs:
                remaining = (pegs - {source, middle}) | {destination}
                suffix = plan(frozenset(remaining))
                if suffix is not None:
                    return ((source, destination),) + suffix
        return None

    path = plan(initial)
    if path is None:
        raise RuntimeError('No peg reduction path found')
    base_level = env.levels_completed
    clone = env.clone()
    for index, (source, destination) in enumerate(path):
        safe_step(clone, (6, *source))
        safe_step(clone, (6, *destination))
        if clone.levels_completed > base_level:
            path = path[:index + 1]
            break
        if len(observe(clone)[1]) != len(initial) - index - 1:
            raise RuntimeError('Observed jump disagrees with the capture model')
    if clone.levels_completed <= base_level:
        raise RuntimeError('Peg reduction did not complete the level')
    for source, destination in path:
        safe_step(env, (6, *source))
        safe_step(env, (6, *destination))
        if env.levels_completed > base_level:
            return
