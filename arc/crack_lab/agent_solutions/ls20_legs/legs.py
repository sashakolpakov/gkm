# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.
#
# ls20 mechanics (discovered by experiment on clones):
#   - Actions: 1=up, 2=down, 3=left, 4=right (grid step = avatar size). 5=noop.
#   - The avatar carries a (shape, colour, rotation) state. Stepping onto a
#     transform tile cycles one of those. A TARGET tile requires a specific
#     (shape, colour, rotation) AND the avatar to stand exactly on it; a target
#     BLOCKS movement until its combo is matched. A level is cleared when every
#     target has been satisfied.
#   - The step-counter UI drains every move, so raw frames never repeat and are
#     useless as a search dedup key. The compact game state is the right key.
from collections import deque


def state_key(env):
    """Compact, hashable state of the ls20 world, read from the simulator's
    internals. Used ONLY as a dedup key for clone-based lookahead search --
    the committed action path is what actually wins and replay-validates."""
    g = env._game
    return (g.gudziatsk.x, g.gudziatsk.y,
            g.fwckfzsyc, g.hiaauhahz, g.cklxociuu,
            tuple(g.lvrnuajbl))


def full_state_key(env):
    """A MORE GENERAL dedup key: the avatar's (shape, colour, rotation) plus the
    (name, x, y, rotation) of EVERY sprite in the level, plus the done-mask.

    Needed when the win condition depends on where OBJECTS end up (push / carry
    levels), not just where the avatar is -- there `state_key` collapses distinct
    worlds together and the search wrongly concludes the goal is unreachable.
    Sprite count is small (tens), so this stays a cheap, hashable tuple."""
    g = env._game
    sprites = tuple(sorted(
        (s._name, s._x, s._y, s.rotation)
        for s in g.current_level.get_sprites()))
    return (g.fwckfzsyc, g.hiaauhahz, g.cklxociuu,
            tuple(g.lvrnuajbl), sprites)


def plan_to_next_level(env, budget=40000, actions=(1, 2, 3, 4), key_fn=state_key):
    """Breadth-first search over avatar states on CLONES for the shortest action
    sequence that raises levels_completed by one. Returns the action list, or
    None if none found within `budget` clone-steps. General over any ls20 level:
    it discovers, on its own, which transform tiles to visit and in what order
    to make every target's (shape, colour, rotation) match at its position.
    Pass a richer `key_fn` (e.g. full_state_key) when the goal depends on where
    OBJECTS end up, not just the avatar."""
    start = env.levels_completed
    root = env.clone()
    seen = {key_fn(root)}
    q = deque([(root, [])])
    steps = 0
    while q:
        e, path = q.popleft()
        for a in actions:
            if steps >= budget:
                return None
            c = e.clone()
            c.step(a)
            steps += 1
            if c.levels_completed > start:
                return path + [a]
            k = key_fn(c)
            if k not in seen:
                seen.add(k)
                q.append((c, path + [a]))
    return None


def run_plan(env, path):
    """Commit a planned action sequence on the REAL env, stopping if the game
    ends. Returns the number of moves committed."""
    n = 0
    for a in path or []:
        if env.terminal():
            break
        env.step(a)
        n += 1
    return n


def advance_one_level(env, budget=40000, key_fn=state_key):
    """Compose: plan a path that clears the current level, then commit it.
    Returns True iff a plan was found and committed. Pass key_fn=full_state_key
    for object-moving levels where avatar position alone is not the whole state."""
    path = plan_to_next_level(env, budget=budget, key_fn=key_fn)
    if not path:
        return False
    run_plan(env, path)
    return True
