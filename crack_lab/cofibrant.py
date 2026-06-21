"""Cofibrant object definitions: identities that LIFT along the renderer.

The renderer is a map  render : LogicalState -> Pixels  that is not injective on
object *shape* (wa30's avatar realises one logical cell as 4x3 OR 3x4 pixels). A
**cofibrant** object definition is one that lifts along `render`: it recovers the
logical object from noisy pixels because it is built only from data invariant
under the renderer's transformations -- the object's logical cell, colour-set, and
its ACTION-RESPONSE ROLE -- never its pixel shape. "I don't care how it
transforms" = the definition quotients out the rendering fibre.

Concretely:
  * the AVATAR is the colour whose logical cell translates by a consistent unit
    vector in response to the directional actions (a non-box that rotates is fine
    -- we read the cell, not the shape);
  * its move-rule is the verified action -> unit-vector map (possibly NON-STANDARD,
    e.g. wa30: 1=up, 2=down, 3=left, 4=right);
  * objects are tracked across frames by (colour, nearest previous cell), with the
    avatar disambiguated by continuity (it can move <=1 cell/step).

No hand-coded colour or direction: everything is read off interaction. See
SPEC_logical_cofibrant.md.
"""
from __future__ import annotations
import copy
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from logical_grid import Grid, LObject, components, objects

Cell = Tuple[int, int]


@dataclass
class AvatarModel:
    """The cofibrant avatar: a colour + a verified action->unit-vector map, learned
    from interaction at logical resolution. `confidence[a]` is the fraction of
    moves under action `a` that matched the modal vector (1.0 = perfectly clean)."""
    color: int
    vectors: Dict[int, Cell]            # action -> (dx, dy) in logical cells
    confidence: Dict[int, float] = field(default_factory=dict)
    score: float = 0.0                  # mean confidence over directional actions

    def predict_cell(self, cell: Cell, action: int) -> Cell:
        dx, dy = self.vectors.get(action, (0, 0))
        return (cell[0] + dx, cell[1] + dy)


def _avatar_cell(arr: np.ndarray, grid: Grid, color: int, near: Optional[Cell] = None) -> Optional[Cell]:
    comps = components(arr, color)
    if not comps:
        return None
    if near is not None:
        # pick the component whose majblock is nearest the previous cell (continuity)
        return min((grid.majblock(c) for c in comps),
                   key=lambda cc: abs(cc[0] - near[0]) + abs(cc[1] - near[1]))
    return grid.majblock(max(comps, key=len))


def identify_avatar(
    transitions: Sequence[Tuple[np.ndarray, int, np.ndarray]],
    grid: Grid,
    candidate_colors: Optional[Sequence[int]] = None,
    directional_actions: Sequence[int] = (1, 2, 3, 4),
) -> Optional[AvatarModel]:
    """Read the avatar off interaction. For each candidate colour, measure how
    consistently its single logical cell translates per action; the avatar is the
    colour with the cleanest non-zero directional response. Shape-invariant:
    only the majblock logical cell is used, so a rotating sprite is fine.

    A colour qualifies as avatar-like only if it stays a SINGLE component (an
    avatar is one piece); multi-piece colours (walls/boxes) are skipped."""
    if not transitions:
        return None
    arr0 = transitions[0][0]
    palette = candidate_colors if candidate_colors is not None else [c for c in range(1, 16) if (arr0 == c).any()]

    best: Optional[AvatarModel] = None
    for color in palette:
        # require single-component (one avatar) on the start frame
        if len(components(arr0, color)) != 1:
            continue
        per_action: Dict[int, Counter] = defaultdict(Counter)
        for before, a, after in transitions:
            bc = _avatar_cell(before, grid, color)
            if bc is None:
                continue
            ac = _avatar_cell(after, grid, color, near=bc)
            if ac is None:
                continue
            per_action[a][(ac[0] - bc[0], ac[1] - bc[1])] += 1
        vectors: Dict[int, Cell] = {}
        confidence: Dict[int, float] = {}
        for a, ctr in per_action.items():
            vec, n = ctr.most_common(1)[0]
            vectors[a] = vec
            confidence[a] = n / sum(ctr.values())
        # an avatar must show a non-zero modal vector on at least 2 directional actions
        nonzero_dirs = [a for a in directional_actions if vectors.get(a, (0, 0)) != (0, 0)]
        if len(nonzero_dirs) < 2:
            continue
        # the modal directional vectors must be DISTINCT (a real D-pad), not all equal
        if len({vectors[a] for a in nonzero_dirs}) < 2:
            continue
        score = float(np.mean([confidence[a] for a in directional_actions if a in confidence]))
        model = AvatarModel(color=color, vectors=vectors, confidence=confidence, score=score)
        if best is None or model.score > best.score:
            best = model
    return best


# ---------------------------------------------------------------------------
# THE GENERAL PRIMITIVE: the action ANCHOR -- the cofibrant object our actions
# act THROUGH, whatever it is. Not necessarily a mover: the effect channel may be
# MOVE (a navigated avatar), RECOLOUR/ACTIVITY (a cursor/toggled tile), or COUNT
# (something that appears/vanishes). The anchor is the object with the most
# action-DISTINCTIVE, CONSISTENT effect -- so future actions have a stable handle
# to be written relative to. Substrate-general; identify_avatar is the MOVE case.
# ---------------------------------------------------------------------------

@dataclass
class Anchor:
    """A PER-COMPONENT anchor: a specific component (colour + start cell), tracked
    across frames by continuity, that our actions steer. Identity is the component,
    NOT the colour -- so an avatar sharing a colour with walls (g50t) or a colour
    playing two roles (wa30's ring vs carrier) is isolated correctly."""
    color: int
    seed: Cell                            # the component's start cell = its identity tag
    size: int                             # renderer-stable component mass
    vectors: Dict[int, Cell]              # action -> modal logical displacement
    consistency: Dict[int, float]         # action -> fraction of MOVES in the modal direction
    moved: Dict[int, bool]                # action -> did the component ever move?
    distinctiveness: float                # #distinct move-vectors / #directional actions
    score: float

    def locate(self, arr, grid: Grid, prev_cell=None) -> Optional[Cell]:
        """Re-find the anchor in a new frame by continuity (this is how future
        actions stay anchored to it)."""
        if prev_cell is not None:
            return track_component(arr, grid, self.color, prev_cell)
        comps = components(arr, self.color)
        if len(comps) == 1:
            return grid.majblock(comps[0])
        compatible = [
            comp for comp in comps
            if abs(len(comp) - self.size) <= max(1, self.size // 4)
        ]
        if len(compatible) == 1:
            return grid.majblock(compatible[0])
        return None

    def __str__(self):
        eff = "  ".join(f"A{a}:{self.vectors[a]}" for a in sorted(self.vectors))
        return (f"anchor=colour {self.color} @component{self.seed} via 'move' "
                f"score={self.score:.2f} distinct={self.distinctiveness:.2f}\n   {eff}")


def _comp_cells(arr, grid: Grid, color) -> List[Cell]:
    return [grid.majblock(c) for c in components(arr, color)]


def track_component(arr, grid: Grid, color, prev_cell: Cell, max_jump: int = 3) -> Optional[Cell]:
    """The current logical cell of the tracked component: the same-colour
    component whose majblock is nearest `prev_cell`, within `max_jump` cells
    (an anchor moves <=1 cell/step; the slack absorbs probe repeats). None if the
    component is lost (occluded/merged) -- handled honestly by the caller."""
    cells = _comp_cells(arr, grid, color)
    if not cells:
        return None
    best = min(cells, key=lambda c: abs(c[0]-prev_cell[0]) + abs(c[1]-prev_cell[1]))
    return best if abs(best[0]-prev_cell[0]) + abs(best[1]-prev_cell[1]) <= max_jump else None


def _color_features(arr, grid: Grid, color, near=None):
    """Per-frame colour features (legacy colour-level channels; kept for the
    effect_summary the LLM reads)."""
    comps = components(arr, color)
    if not comps:
        return {"cell": None, "count": 0, "cells": frozenset()}
    if near is not None:
        cell = min((grid.majblock(c) for c in comps),
                   key=lambda cc: abs(cc[0]-near[0]) + abs(cc[1]-near[1]))
    else:
        cell = grid.majblock(max(comps, key=len))
    allcells = frozenset(grid.majblock(c) for c in comps)
    return {"cell": cell, "count": len(comps), "cells": allcells}


def score_seed(sequences, grid: Grid, color, seed: Cell, directional_actions=(1, 2, 3, 4)):
    """Track ONE component (colour+seed) through each action's probe sequence and
    summarise its per-action displacement. `sequences` = {action: [(before,after)]}
    contiguous from reset, so continuity holds within an action's run."""
    vectors, consistency, moved = {}, {}, {}
    for a, seq in sequences.items():
        disps: Counter = Counter()
        pos = seed
        for before, after in seq:
            bcell = track_component(before, grid, color, pos)
            if bcell is None:
                break
            acell = track_component(after, grid, color, bcell)
            if acell is None:
                break
            disps[(acell[0]-bcell[0], acell[1]-bcell[1])] += 1
            pos = acell
        nonzero = {v: n for v, n in disps.items() if v != (0, 0)}
        if nonzero:
            v = max(nonzero, key=nonzero.get)
            vectors[a] = v; consistency[a] = nonzero[v] / sum(nonzero.values()); moved[a] = True
        else:
            vectors[a] = (0, 0); consistency[a] = 1.0; moved[a] = False
    return vectors, consistency, moved


def identify_anchor(sequences, grid: Grid, candidate_colors=None,
                    directional_actions=(1, 2, 3, 4)) -> Optional[Anchor]:
    """Find the PER-COMPONENT action anchor: the single component whose directional
    actions produce DISTINCT, CONSISTENT moves (so actions can steer it). Tracks by
    continuity, shape-invariant (majblock). `sequences` = {action: [(before,after)]}.
    Returns None if no component is distinctly steerable (honest null)."""
    if not sequences:
        return None
    start = next(iter(sequences.values()))[0][0]
    palette = candidate_colors if candidate_colors is not None else [c for c in range(1, 16) if (start == c).any()]

    best: Optional[Anchor] = None
    for color in palette:
        for component in components(start, color):
            seed = grid.majblock(component)
            vectors, consistency, moved = score_seed(sequences, grid, color, seed, directional_actions)
            mv = [a for a in directional_actions if moved.get(a)]
            distinct = {vectors[a] for a in mv}
            if len(distinct) < 2:                          # actions can't steer it distinctly
                continue
            distinctiveness = len(distinct) / len(directional_actions)
            mc = float(np.mean([consistency[a] for a in mv])) if mv else 0.0
            score = distinctiveness * mc
            cand = Anchor(color=color, seed=seed, size=len(component),
                          vectors=vectors, consistency=consistency,
                          moved=moved, distinctiveness=distinctiveness, score=score)
            if best is None or cand.score > best.score:
                best = cand
    return best


# ---------------------------------------------------------------------------
# Action->avatar-displacement probe by direct intervention (no random walk)
# ---------------------------------------------------------------------------

def probe_avatar(make_env, grid: Grid, color: int, actions=(1, 2, 3, 4, 5),
                 name_map=None) -> Dict[int, Cell]:
    """Direct, deterministic read of the avatar's per-action logical displacement
    from the start state (one clone per action). Complements identify_avatar's
    statistical read; both are shape-invariant (majblock)."""
    from arcengine import ActionInput, GameAction as EA
    NAME = name_map or {0: "RESET", 1: "ACTION1", 2: "ACTION2", 3: "ACTION3", 4: "ACTION4", 5: "ACTION5"}
    e = make_env(); e.reset(); g0 = copy.deepcopy(e._env._game)
    base = np.asarray(g0.perform_action(ActionInput(id=EA.RESET), raw=True).frame[-1])
    c0 = _avatar_cell(base, grid, color)
    out: Dict[int, Cell] = {}
    for a in actions:
        gc = copy.deepcopy(g0); gc.perform_action(ActionInput(id=EA.RESET), raw=True)
        arr = np.asarray(gc.perform_action(ActionInput(id=EA[NAME[a]]), raw=True).frame[-1])
        c1 = _avatar_cell(arr, grid, color, near=c0)
        out[a] = (c1[0] - c0[0], c1[1] - c0[1]) if (c0 and c1) else (0, 0)
    return out
