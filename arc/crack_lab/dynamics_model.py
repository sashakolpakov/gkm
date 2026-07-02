"""The move-rule as a VERIFIED model over logical cells.

A `MoveRule` is a small, fully-symbolic hypothesis about a game's transition
function -- the thing a local LLM can propose and the fidelity gate verifies:

    vectors      : action -> (dx,dy) avatar displacement in logical cells
                   (may be NON-STANDARD, e.g. wa30: 1=up,2=down,3=left,4=right)
    wall_colors  : colours whose logical cells block the avatar
    box_colors   : colours the avatar can push (one cell per push, same vector)
    push         : whether pushing is active

Prediction is pure and runs on the cofibrant (logical) objects, where the
renderer noise is already quotiented out. `fidelity()` is the gate: exact-match
fraction of predicted (avatar cell + box cells) against held-out reality. We also
provide the two honest controls -- a data-only most-common-delta learner and a
constant "everything stays" floor.

See SPEC_logical_cofibrant.md (C3: structured rule -> near-exact).
"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import numpy as np

from logical_grid import Grid, components, objects

Cell = Tuple[int, int]
Vec = Tuple[int, int]


# ---------------------------------------------------------------------------
# Logical state: the cofibrant objects of one frame
# ---------------------------------------------------------------------------

@dataclass
class LState:
    avatar: Optional[Cell]
    boxes: List[Tuple[int, Cell]]          # (colour, cell) for each pushable component
    walls: Set[Cell]
    bounds: Tuple[int, int]                # (nx, ny) logical grid size

    @classmethod
    def of(cls, arr: np.ndarray, grid: Grid, avatar_color: int,
           box_colors: Sequence[int], wall_colors: Sequence[int],
           near: Optional[Cell] = None) -> "LState":
        H, W = arr.shape
        bounds = ((W - grid.phase[0] + grid.pitch - 1) // grid.pitch,
                  (H - grid.phase[1] + grid.pitch - 1) // grid.pitch)
        av_comps = components(arr, avatar_color)
        if not av_comps:
            avatar = None
        elif near is not None:
            avatar = min((grid.majblock(c) for c in av_comps),
                         key=lambda cc: abs(cc[0] - near[0]) + abs(cc[1] - near[1]))
        else:
            avatar = grid.majblock(max(av_comps, key=len))
        boxes = [(o.color, o.cell) for o in objects(arr, grid, list(box_colors))]
        walls = {o.cell for o in objects(arr, grid, list(wall_colors))}
        return cls(avatar=avatar, boxes=boxes, walls=walls, bounds=bounds)

    def box_cells(self) -> List[Tuple[int, Cell]]:
        return sorted(self.boxes)


@dataclass
class MoveRule:
    vectors: Dict[int, Vec]
    wall_colors: FrozenSet[int] = frozenset()
    box_colors: FrozenSet[int] = frozenset()
    push: bool = True
    name: str = ""

    def _oob(self, cell: Cell, bounds: Tuple[int, int]) -> bool:
        return not (0 <= cell[0] < bounds[0] and 0 <= cell[1] < bounds[1])

    def predict(self, st: LState, action: int) -> Tuple[Optional[Cell], List[Tuple[int, Cell]]]:
        """Predict (avatar cell, sorted box cells) after `action`. Pure."""
        boxes = list(st.boxes)
        if st.avatar is None:
            return None, sorted(boxes)
        dx, dy = self.vectors.get(action, (0, 0))
        if (dx, dy) == (0, 0):
            return st.avatar, sorted(boxes)
        tgt = (st.avatar[0] + dx, st.avatar[1] + dy)
        if self._oob(tgt, st.bounds) or tgt in st.walls:
            return st.avatar, sorted(boxes)                      # blocked by wall/edge
        # is there a pushable box at the target cell?
        idx = next((i for i, (_, c) in enumerate(boxes) if c == tgt), None)
        if idx is not None:
            if not self.push:
                return st.avatar, sorted(boxes)                  # solid box, no push -> stay
            col, _ = boxes[idx]
            btgt = (tgt[0] + dx, tgt[1] + dy)
            occupied = {c for j, (_, c) in enumerate(boxes) if j != idx}
            if self._oob(btgt, st.bounds) or btgt in st.walls or btgt in occupied:
                return st.avatar, sorted(boxes)                  # push blocked -> stay
            boxes[idx] = (col, btgt)
            return tgt, sorted(boxes)                            # push succeeds
        return tgt, sorted(boxes)                                # free move


# ---------------------------------------------------------------------------
# The fidelity gate
# ---------------------------------------------------------------------------

@dataclass
class Fidelity:
    joint: float
    avatar: float
    box: float
    n: int

    def __str__(self) -> str:
        return (f"joint {self.joint*100:.0f}%  avatar {self.avatar*100:.0f}%  "
                f"box {self.box*100:.0f}%  (n={self.n})")


def _box_truth(arr, grid, box_colors) -> List[Tuple[int, Cell]]:
    return sorted((o.color, o.cell) for o in objects(arr, grid, list(box_colors)))


def fidelity(rule: MoveRule, transitions, grid: Grid, avatar_color: int) -> Fidelity:
    """Exact-match fraction over held-out transitions. Avatar tracked with
    continuity (near=previous cell)."""
    av_ok = box_ok = joint = n = 0
    for before, a, after in transitions:
        st = LState.of(before, grid, avatar_color, rule.box_colors, rule.wall_colors)
        if st.avatar is None:
            continue
        n += 1
        p_av, p_box = rule.predict(st, a)
        r_av = LState.of(after, grid, avatar_color, rule.box_colors, rule.wall_colors,
                         near=st.avatar).avatar
        r_box = _box_truth(after, grid, rule.box_colors)
        a_ok = (p_av == r_av); b_ok = (p_box == r_box)
        av_ok += a_ok; box_ok += b_ok; joint += (a_ok and b_ok)
    d = max(1, n)
    return Fidelity(joint/d, av_ok/d, box_ok/d, n)


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

def constant_floor(transitions, grid: Grid, avatar_color: int,
                   box_colors, wall_colors) -> Fidelity:
    """'Everything stays' baseline -- exposes box-rarely-moves inflation."""
    rule = MoveRule(vectors={}, box_colors=frozenset(box_colors),
                    wall_colors=frozenset(wall_colors), push=False, name="constant")
    return fidelity(rule, transitions, grid, avatar_color)


@dataclass
class DataRule:
    """Data-only baseline: most-common avatar delta per action + most-common box
    displacement per (action, box-offset-from-avatar). No wall/push schema -- pure
    lookup. This is the control the LLM/structured rule must beat."""
    av_delta: Dict[int, Vec]
    box_disp: Dict[Tuple[int, Vec], Vec]
    box_colors: FrozenSet[int]

    @classmethod
    def learn(cls, transitions, grid: Grid, avatar_color: int, box_colors) -> "DataRule":
        avd: Dict[int, Counter] = defaultdict(Counter)
        bxd: Dict[Tuple[int, Vec], Counter] = defaultdict(Counter)
        for before, a, after in transitions:
            sb = LState.of(before, grid, avatar_color, box_colors, [])
            sa = LState.of(after, grid, avatar_color, box_colors, [], near=sb.avatar)
            if sb.avatar and sa.avatar:
                avd[a][(sa.avatar[0]-sb.avatar[0], sa.avatar[1]-sb.avatar[1])] += 1
            if sb.avatar:
                after_boxes = sa.boxes
                for col, bc in sb.boxes:
                    if not after_boxes:
                        continue
                    nc = min((c for cc, c in after_boxes if cc == col),
                             key=lambda c: abs(c[0]-bc[0])+abs(c[1]-bc[1]), default=bc)
                    rel = (bc[0]-sb.avatar[0], bc[1]-sb.avatar[1])
                    bxd[(a, rel)][(nc[0]-bc[0], nc[1]-bc[1])] += 1
        return cls(av_delta={a: c.most_common(1)[0][0] for a, c in avd.items()},
                   box_disp={k: c.most_common(1)[0][0] for k, c in bxd.items()},
                   box_colors=frozenset(box_colors))

    def fidelity(self, transitions, grid: Grid, avatar_color: int) -> Fidelity:
        av_ok = box_ok = joint = n = 0
        for before, a, after in transitions:
            sb = LState.of(before, grid, avatar_color, self.box_colors, [])
            if sb.avatar is None:
                continue
            n += 1
            d = self.av_delta.get(a, (0, 0))
            p_av = (sb.avatar[0]+d[0], sb.avatar[1]+d[1])
            pb = []
            for col, bc in sb.boxes:
                rel = (bc[0]-sb.avatar[0], bc[1]-sb.avatar[1])
                bd = self.box_disp.get((a, rel), (0, 0))
                pb.append((col, (bc[0]+bd[0], bc[1]+bd[1])))
            r_av = LState.of(after, grid, avatar_color, self.box_colors, [], near=sb.avatar).avatar
            r_box = _box_truth(after, grid, self.box_colors)
            a_ok = (p_av == r_av); b_ok = (sorted(pb) == r_box)
            av_ok += a_ok; box_ok += b_ok; joint += (a_ok and b_ok)
        dd = max(1, n)
        return Fidelity(joint/dd, av_ok/dd, box_ok/dd, n)


# ---------------------------------------------------------------------------
# Fit a structured rule from data (the non-LLM way to get a MoveRule), so the
# LLM-proposed rule has an honest data-fit counterpart to be compared against.
# ---------------------------------------------------------------------------

def fit_structured_rule(train, grid: Grid, avatar_color: int,
                        dynamic_colors, push: bool = True) -> MoveRule:
    """Fit a MoveRule from data: vectors = avatar's modal per-action logical delta
    (continuity-tracked); then PARTITION the non-background `dynamic_colors` into
    (walls, boxes) by choosing the wall-subset that MAXIMISES train fidelity (the
    data choosing the obstacle map). The LLM proposes the same fields directly, so
    this is the structured rule's honest data-fit counterpart."""
    avd: Dict[int, Counter] = defaultdict(Counter)
    for before, a, after in train:
        sb = LState.of(before, grid, avatar_color, dynamic_colors, [])
        sa = LState.of(after, grid, avatar_color, dynamic_colors, [], near=sb.avatar)
        if sb.avatar and sa.avatar:
            avd[a][(sa.avatar[0]-sb.avatar[0], sa.avatar[1]-sb.avatar[1])] += 1
    vectors = {a: c.most_common(1)[0][0] for a, c in avd.items()}
    best = None
    from itertools import combinations
    cand = list(dynamic_colors)
    subsets = [frozenset()] + [frozenset(s) for r in range(1, len(cand)+1) for s in combinations(cand, r)]
    for ws in subsets:                                   # ws = walls; boxes = the rest
        boxes = frozenset(c for c in cand if c not in ws)
        rule = MoveRule(vectors=vectors, wall_colors=ws, box_colors=boxes,
                        push=push, name=f"structured/walls={sorted(ws)}")
        f = fidelity(rule, train, grid, avatar_color).joint
        if best is None or f > best[0]:
            best = (f, rule)
    return best[1]
