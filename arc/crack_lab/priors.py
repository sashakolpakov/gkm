"""Human perceptual priors as reusable 'mental legs' for ARC-AGI-3 games.

The games speak a uniform pixel/colour language with no goal labels — which is
exactly why a human cracks them fast and a blind searcher does not: humans bring
a SHIT-LOAD of preconceptions (what an avatar is, that blinking = salient, that a
framed region is a container/target, that a filling bar = progress). Those are
'mental legs' — perceptual primitives the cone search walks on. Hard-coding a
tint of them gives the search the goal-shaped heuristics it can't induce from
sparse reward alone.

This module is the prior library: cheap, game-agnostic detectors over frames /
frame-pairs. The cone search consumes them as (a) the objects legs bind to and
(b) progress heuristics that order discovery. None of them read the reward; they
are priors, not the goal.

Validated and used by the cracker:
  * avatar          — the rigidly-translating, occlusion-robust sprite (the thing
                      *I* control). [bfs_crack.detect_avatar_color]
  * movable_objects — small rigid sprites that translate (boxes/tiles).
  * container       — a colour ring around an interior colour (a target slot).
  * box_prior       — movable objects should approach the container (a heuristic).
  * salient_change  — cells that change between frames (blink / motion = important).
  * interaction_sites — cells where ACTION5/6 cause a frame change (where to act).
Planned (stubs): legend/target template matching, progress-bar/HUD reading,
symbol/pattern equality.
"""
from __future__ import annotations
from lab import arc
from typing import Dict, List, Optional, Tuple

Cell = Tuple[int, int]
WALL_LIKE_MIN_FRACTION = 0.20  # a colour covering >20% of the board reads as structure/wall


def _cells_by_color(frame) -> Dict[int, List[Cell]]:
    out: Dict[int, List[Cell]] = {}
    for y in range(len(frame)):
        for x in range(len(frame[0])):
            c = frame[y][x]
            if c != arc.BACKGROUND:
                out.setdefault(c, []).append((x, y))
    return out


def _cen(cells) -> Cell:
    return (round(sum(p[0] for p in cells) / len(cells)), round(sum(p[1] for p in cells) / len(cells)))


def structure_colours(frame) -> List[int]:
    """Colours that cover a large fraction of the board: walls/background/floor —
    the static scaffold a human ignores when looking for the 'pieces'."""
    n = len(frame) * len(frame[0])
    out = []
    for c, cells in _cells_by_color(frame).items():
        if len(cells) >= WALL_LIKE_MIN_FRACTION * n:
            out.append(c)
    return out


def movable_objects(frame, avatar_color: Optional[int] = None, max_size: int = 30) -> List[Tuple[int, Cell, int]]:
    """Small rigid sprites (candidate pushables/tiles): (colour, centroid, size),
    excluding the avatar and the big structure colours. A human sees these as
    'the pieces to move'."""
    skip = set(structure_colours(frame))
    if avatar_color is not None:
        skip.add(avatar_color)
    out = []
    for c in range(1, arc.NUM_COLORS):
        if c in skip:
            continue
        for comp in arc.connected_components(frame, c):
            if 1 <= len(comp) <= max_size:
                out.append((c, _cen(comp), len(comp)))
    return out


def containers(frame) -> List[Tuple[int, int, Cell]]:
    """A colour ring around a different interior colour reads as a container /
    target slot: (frame_colour, interior_colour, interior_centroid). Detected by
    an interior colour fully surrounded by another colour's cells."""
    out = []
    cbc = _cells_by_color(frame)
    cellset = {c: set(v) for c, v in cbc.items()}
    for ic, interior in cbc.items():
        # the ring colour = the modal colour just outside the interior's bbox border
        xs = [p[0] for p in interior]; ys = [p[1] for p in interior]
        x0, x1, y0, y1 = min(xs) - 1, max(xs) + 1, min(ys) - 1, max(ys) + 1
        border = {}
        for x in range(x0, x1 + 1):
            for y in (y0, y1):
                v = frame[y][x] if 0 <= y < len(frame) and 0 <= x < len(frame[0]) else None
                if v not in (None, ic, arc.BACKGROUND):
                    border[v] = border.get(v, 0) + 1
        for y in range(y0, y1 + 1):
            for x in (x0, x1):
                v = frame[y][x] if 0 <= y < len(frame) and 0 <= x < len(frame[0]) else None
                if v not in (None, ic, arc.BACKGROUND):
                    border[v] = border.get(v, 0) + 1
        if border:
            ring = max(border, key=border.get)
            if border[ring] >= 0.6 * (2 * (x1 - x0 + 1) + 2 * (y1 - y0 + 1)):
                out.append((ring, ic, _cen(interior)))
    return out


def box_prior(frame, box_color: int, target_color: int) -> float:
    """Heuristic mental leg: the movable boxes should approach the target. Lower
    is better. Used to ORDER discovery, not as the reward."""
    tgt = [(_cen(c)) for c in arc.connected_components(frame, target_color)]
    if not tgt:
        return 0.0
    tx, ty = tgt[0]
    boxes = [_cen(c) for c in arc.connected_components(frame, box_color)]
    return float(sum(abs(bx - tx) + abs(by - ty) for bx, by in boxes))


def salient_change(before, after) -> List[Cell]:
    """Cells that changed between two frames — motion / blinking. A human's eye
    snaps to these; they flag where the action and the interactables are."""
    return [(x, y) for y in range(len(before)) for x in range(len(before[0]))
            if before[y][x] != after[y][x]]


def nonstructure_change(before, after, avatar_color: Optional[int] = None) -> int:
    """Count of salient changes NOT explained by the avatar moving — i.e. the
    world reacted (a box moved, a tile toggled). A general progress proxy when no
    specific target is known."""
    skip = set(structure_colours(before))
    if avatar_color is not None:
        skip.add(avatar_color)
    n = 0
    for (x, y) in salient_change(before, after):
        if before[y][x] not in skip or after[y][x] not in skip:
            n += 1
    return n


# ---------------------------------------------------------------------------
# Avatar detection (game-agnostic perception): the controllable object is the
# colour whose component RIGIDLY TRANSLATES under some action — occlusion-robust
# (rejects flickering walls), direction-agnostic (non-canonical action maps ok).
# ---------------------------------------------------------------------------

def _shape(cells):
    mnx = min(p[0] for p in cells); mny = min(p[1] for p in cells)
    return frozenset((x - mnx, y - mny) for x, y in cells)


def avatar_centroid(frame, color, near=None):
    comps = arc.connected_components(frame, color)
    if not comps:
        return None
    if near is None:
        comp = max(comps, key=len)
    else:
        comp = min(comps, key=lambda c: abs(_cen(c)[0] - near[0]) + abs(_cen(c)[1] - near[1]))
    return _cen(comp)


def detect_avatar_color(make_env):
    """Returns (avatar_colour, available_actions, a_moving_centroid) or (None,...)."""
    base = make_env(); s0 = base.reset()
    acts = [a for a in (base.available_actions or [1, 2, 3, 4]) if 1 <= a <= 5]
    MIN_SPRITE = 4
    best = None
    for color in arc.extract_scene(s0.frame).colors_present():
        before_lookup = {}
        for c in arc.connected_components(s0.frame, color):
            if len(c) >= MIN_SPRITE:
                before_lookup.setdefault(_shape(c), []).append(_cen(c))
        for a in acts:
            e = make_env(); e.reset()
            for c in arc.connected_components(e.step(arc.GameAction(a)).frame, color):
                if len(c) < MIN_SPRITE:
                    continue
                sh, ce = _shape(c), _cen(c)
                if sh in before_lookup and all(ce != b for b in before_lookup[sh]):
                    if best is None or len(c) > best[0]:
                        best = (len(c), color, ce)
    if best is None:
        return None, acts, None
    return best[1], acts, best[2]
