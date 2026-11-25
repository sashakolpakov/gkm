"""Small reusable skills learned from public frames and clone experiments."""
import numpy as np
from perception import arr


def complete_glyph_substitution(env, examples, slots, ink=5):
    """Match example glyph classes up to rotation/reflection and cycle answers.

    Examples contain (source row, source column, answer column); slots contain
    (source row, answer row, column). Glyphs occupy five by five pixels. Left
    and right select slots; up and down cycle the selected answer inversely.
    The dense progress measure is the number of answers in their mapped class.
    """
    def glyph(frame, row, col):
        return (arr(frame)[row:row + 5, col:col + 5] == ink).copy()

    def identity(g):
        return min(np.rot90(g[:, ::direction], turn).tobytes()
                   for direction in (1, -1) for turn in range(4))

    frame = env.frame()
    mapping = {identity(glyph(frame, row, source)): identity(glyph(frame, row, target))
               for row, source, target in examples}
    goals = [mapping[identity(glyph(frame, source, col))]
             for source, target, col in slots]
    # The dark cursor brackets align with the selected answer's column.
    dark_rows, dark_cols = np.where(arr(frame) == 0)
    selected = min(range(len(slots)),
                   key=lambda i: abs(slots[i][2] + 2 - float(np.mean(dark_cols))))
    work = env.clone()
    base_level = env.levels_completed
    path = []

    def advance(action):
        work.step(action)
        path.append(action)

    for index, (source, target, col) in enumerate(slots):
        right = (index - selected) % len(slots)
        left = (selected - index) % len(slots)
        for _ in range(min(right, left)):
            advance(4 if right <= left else 3)
        selected = index
        # Enumerate a bounded answer ring on a separate clone, then use its
        # shortest direction. Every completed slot is a verified subgoal.
        scan = work.clone()
        ring = []
        match = None
        for offset in range(32):
            g = glyph(scan.frame(), target, col)
            raw = g.tobytes()
            if raw in ring:
                break
            ring.append(raw)
            if identity(g) == goals[index]:
                match = offset
            scan.step(1)
            if scan.levels_completed > base_level:
                # The completing action can transition away from this board.
                if match is None:
                    match = offset + 1
                break
        if match is None:
            raise RuntimeError('No mapped glyph in the bounded answer ring')
        # If the scan transitioned, its discovered prefix is already a valid
        # completion; otherwise the repeated frame establishes the ring size.
        if scan.levels_completed > base_level:
            edits = [1] * (offset + 1)
        else:
            reverse = (len(ring) - match) % len(ring)
            edits = [1] * match if match <= reverse else [2] * reverse
        for action in edits:
            advance(action)
            if work.levels_completed > base_level:
                break
        if work.levels_completed > base_level:
            break
        if identity(glyph(work.frame(), target, col)) != goals[index]:
            raise RuntimeError('Glyph subgoal failed on clone')
    if work.levels_completed <= base_level:
        raise RuntimeError('Mapped glyphs did not complete the level')
    for action in path:
        env.step(action)
        if env.levels_completed > base_level:
            return
