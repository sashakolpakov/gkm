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
    frame = env.frame()
    mapping = {_glyph_identity(_glyph(frame, row, source, ink)):
               _glyph_identity(_glyph(frame, row, target, ink))
               for row, source, target in examples}
    goals = [mapping[_glyph_identity(_glyph(frame, source, col, ink))]
             for source, target, col in slots]
    complete_glyph_answers(env, [(target, col) for source, target, col in slots], goals, ink)


def _glyph(frame, row, col, ink):
    return (arr(frame)[row:row + 5, col:col + 5] == ink).copy()


def _glyph_identity(g):
    return min(np.rot90(g[:, ::direction], turn).tobytes()
               for direction in (1, -1) for turn in range(4))


def complete_glyph_expansion(env, examples, sources, answers, ink=5):
    """Expand each source glyph into its demonstrated sequence of glyph classes."""
    frame = env.frame()
    mapping = {_glyph_identity(_glyph(frame, row, source, ink)):
               [_glyph_identity(_glyph(frame, row, target, ink)) for target in targets]
               for row, source, targets in examples}
    goals = [goal for row, col in sources
             for goal in mapping[_glyph_identity(_glyph(frame, row, col, ink))]]
    if len(goals) != len(answers):
        raise RuntimeError('Expanded glyph sequence does not fit the answer slots')
    complete_glyph_answers(env, answers, goals, ink)


def complete_glyph_rewrite(env, examples, sources, answers, ink=5):
    """Rewrite a glyph sequence using demonstrated groups, longest match first.

    Each example pairs source coordinates with replacement coordinates.
    Glyph classes ignore rotation and reflection, as in substitution/expansion.
    """
    frame = env.frame()

    def sequence(coords):
        return tuple(_glyph_identity(_glyph(frame, row, col, ink))
                     for row, col in coords)

    rules = sorted([(sequence(source), sequence(target))
                    for source, target in examples],
                   key=lambda rule: len(rule[0]), reverse=True)
    remaining = sequence(sources)
    goals = []
    while remaining:
        for source, target in rules:
            if source and remaining[:len(source)] == source:
                goals.extend(target)
                remaining = remaining[len(source):]
                break
        else:
            raise RuntimeError('No demonstrated rule matches the glyph sequence')
    if len(goals) != len(answers):
        raise RuntimeError('Rewritten glyph sequence does not fit the answer slots')
    complete_glyph_answers(env, answers, goals, ink)


def complete_glyph_rule_synthesis(env, rules, sources, targets, ink=5):
    """Fit editable repeated-glyph rules to a fixed sequence pair.

    Each rule is used once. Its source and target groups each cycle as a
    unit, so a group can match only a run of one glyph class. Matched source
    and target prefixes provide dense progress while fitting the rules.
    """
    frame = env.frame()

    def sequence(coords):
        return tuple(_glyph_identity(_glyph(frame, r, c, ink))
                     for r, c in coords)

    source, target = sequence(sources), sequence(targets)

    def fits(i, j, remaining, assignments):
        if not remaining:
            if i == len(source) and j == len(target):
                yield assignments
            return
        for index in remaining:
            left, right = rules[index]
            a, b = source[i:i + len(left)], target[j:j + len(right)]
            if (len(a) != len(left) or len(b) != len(right)
                    or not a or not b or len(set(a)) != 1 or len(set(b)) != 1):
                continue
            yield from fits(i + len(left), j + len(right),
                            [k for k in remaining if k != index],
                            assignments | {index: (a[0], b[0])})

    slots = [group[0] for rule in rules for group in rule]
    base_level = env.levels_completed
    for assignment in fits(0, 0, list(range(len(rules))), {}):
        goals = [goal for index in range(len(rules))
                 for goal in assignment[index]]
        trial = env.clone()
        try:
            complete_glyph_answers(trial, slots, goals, ink)
        except RuntimeError:
            continue
        if trial.levels_completed > base_level:
            complete_glyph_answers(env, slots, goals, ink)
            return
    raise RuntimeError('No fitted glyph rule set completed the level')


def complete_glyph_answers(env, slots, goals, ink=5):
    """Set a cursor-selected glyph ring to target classes, verifying on a clone."""
    def glyph(frame, row, col):
        return _glyph(frame, row, col, ink)

    identity = _glyph_identity
    frame = env.frame()
    # The dark cursor brackets align with the selected answer's column.
    dark_rows, dark_cols = np.where(arr(frame) == 0)
    selected = min(range(len(slots)),
                   key=lambda i: abs(slots[i][1] + 2 - float(np.mean(dark_cols))))
    work = env.clone()
    base_level = env.levels_completed
    path = []

    def advance(action):
        work.step(action)
        path.append(action)

    for index, (target, col) in enumerate(slots):
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
