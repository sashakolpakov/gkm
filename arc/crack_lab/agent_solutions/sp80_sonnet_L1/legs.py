# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.
#
# sp80 world model (level 1): a spout (color 4) holds liquid (color 6) that,
# on ACTION5, pours straight down its column. A movable flat bar (color 9,
# avatar; ACTION1..4 = up/down/left/right, 4 cells per step) deflects the
# stream: liquid pooling on the bar spills off BOTH ends, falling in the
# 4-wide column just outside each bar edge. Cups (color 11) at the bottom
# have 4-wide openings; the level completes when the poured liquid lands in
# the cup openings. Pouring liquid that misses is a strike (~5 strikes =
# GAME_OVER), and a timer (top row, color 14) depletes 2 cells per step.

import numpy as np

SPOUT, LIQUID, BAR, CUP = 4, 6, 9, 11


def bbox(frame, color):
    """(ymin, ymax, xmin, xmax) of a colour, or None if absent."""
    ys, xs = np.where(np.asarray(frame) == color)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def cup_openings(frame, cup_color=CUP, max_opening=8):
    """Column ranges (x0, x1) of the narrow gaps at the cups' rim row —
    the slots liquid must fall into. Wide gaps (space between cups) are
    excluded."""
    f = np.asarray(frame)
    bb = bbox(f, cup_color)
    if bb is None:
        return []
    row = f[bb[0]]
    cols = np.where(row == cup_color)[0].tolist()
    runs = []
    for x in cols:
        if runs and x == runs[-1][1] + 1:
            runs[-1][1] = x
        else:
            runs.append([x, x])
    openings = []
    for (a0, a1), (b0, b1) in zip(runs, runs[1:]):
        gap = (a1 + 1, b0 - 1)
        if 0 < gap[1] - gap[0] + 1 <= max_opening:
            openings.append(gap)
    return openings


def move_bar_to_left_col(env, left, bar_color=BAR, max_steps=32):
    """Slide the avatar bar horizontally until its left edge is at `left`
    (ACTION3 = left, ACTION4 = right; the bar moves 4 cells per step)."""
    for _ in range(max_steps):
        bb = bbox(env.frame(), bar_color)
        if bb is None or bb[2] == left:
            return
        env.step(4 if bb[2] < left else 3)


def best_deflect_left_col(frame, bar_color=BAR, spout_color=SPOUT, spill=4):
    """Choose the bar left-edge column so that the spout's stream lands on
    the bar and the spill columns just outside each bar edge line up with
    as many cup openings as possible. Returns a left column reachable in
    4-cell steps from the bar's current position."""
    f = np.asarray(frame)
    bar = bbox(f, bar_color)
    sp = bbox(f, spout_color)
    ops = cup_openings(f)
    w = bar[3] - bar[2] + 1
    best, best_score = bar[2], -1
    for left in range(bar[2] % 4, 64 - w + 1, 4):
        right = left + w - 1
        if not (left <= sp[2] and sp[3] <= right):
            continue  # stream must hit the bar
        spills = [(left - spill, left - 1), (right + 1, right + spill)]
        score = sum(s in ops for s in spills)
        if score > best_score:
            best, best_score = left, score
    return best


def pour(env):
    """ACTION5: pour the spout's liquid. Only call once the deflector is
    aligned — a missed pour is a strike."""
    env.step(5)


def sense_align_trigger(env, sense_fn, align_fn, trigger_fn):
    """Read the world (sense_fn(frame) → target), move the tool to that target
    (align_fn(env, target)), then fire the irreversible action (trigger_fn(env))."""
    align_fn(env, sense_fn(env.frame()))
    trigger_fn(env)
