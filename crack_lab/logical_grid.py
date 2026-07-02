"""Logical-cell substrate: quotient out the renderer.

ARC-AGI-3 keyboard games render a coarse NxN *logical* grid up into the 64x64
pixel frame (wa30: 16x16 at pitch 4). A sprite occupies ~one logical cell but its
pixels animate / rotate (wa30's avatar is a non-box: 4x3 facing up/down, 3x4
facing left/right). Tracking pixels inherits that noise; tracking the *logical
cell* a component lands in does not.

This module is the renderer-noise-killing layer for everything downstream:
  detect_pitch / detect_phase   : recover the logical grid geometry from a frame
  object_cell (majblock)        : map a pixel component -> the logical cell that
                                  holds the MAJORITY of its pixels (shape-invariant)
  objects                       : (colour, logical_cell, size) for every component
  to_logical                    : the whole frame as an NxN symbolic grid

No game semantics here (no 'box', no 'avatar', no reward) -- pure geometry, so it
is reusable across games and substrates. See SPEC_logical_cofibrant.md.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

Cell = Tuple[int, int]
BACKGROUND = 0


# ---------------------------------------------------------------------------
# Grid geometry: pitch (block size) and phase (origin offset within a block)
# ---------------------------------------------------------------------------

def _boundaries(arr: np.ndarray):
    """All colour-change boundary coordinates: xs (vertical edges), ys (horizontal)."""
    H, W = arr.shape
    xs: List[int] = []; ys: List[int] = []
    for y in range(H):
        for x in range(1, W):
            if arr[y][x] != arr[y][x - 1]:
                xs.append(x)
    for x in range(W):
        for y in range(1, H):
            if arr[y][x] != arr[y - 1][x]:
                ys.append(y)
    return xs, ys


def detect_pitch(arr: np.ndarray, candidates: Sequence[int] = (16, 8, 4, 2)) -> int:
    """The logical block size, by RESIDUE-CONCENTRATION DROP-ON-DOUBLING.

    At the true pitch P, colour-change boundaries align to a dominant residue mod
    P on each axis; DOUBLING to 2P splits adjacent cells into two sub-residues and
    sharply drops that alignment. So we score each candidate by
        conc(P) - conc(2P)        (best axis)
    and pick the maximiser. This is robust to (a) origin offsets, (b) 1px sprite
    detail, and (c) MIXED phases (wa30's avatar sits off the container's grid
    phase) -- all of which fool a plain modal-gap or fixed-threshold detector."""
    xs, ys = _boundaries(arr)
    if not xs and not ys:
        return 1

    def conc(coords, P):
        if not coords:
            return 0.0
        res = Counter(c % P for c in coords)
        return res.most_common(1)[0][1] / len(coords)

    def best_axis(P):
        return max(conc(xs, P), conc(ys, P))

    best = (0.0, 1)
    for P in candidates:
        if arr.shape[0] % P or arr.shape[1] % P:
            continue
        score = best_axis(P) - best_axis(2 * P)
        if score > best[0] + 1e-9 or (abs(score - best[0]) <= 1e-9 and P > best[1]):
            best = (score, P)
    return best[1] if best[0] > 0.1 else 1


def detect_phase(arr: np.ndarray, pitch: int) -> Cell:
    """The grid origin offset in [0,pitch): the residue mod pitch where most
    colour-change boundaries land on each axis. (wa30 -> (0,3): the play area is
    shifted off pixel 0.)"""
    if pitch <= 1:
        return (0, 0)
    H, W = arr.shape
    cx: Counter = Counter(); cy: Counter = Counter()
    for y in range(H):
        for x in range(1, W):
            if arr[y][x] != arr[y][x - 1]:
                cx[x % pitch] += 1
    for x in range(W):
        for y in range(1, H):
            if arr[y][x] != arr[y - 1][x]:
                cy[y % pitch] += 1
    px = cx.most_common(1)[0][0] % pitch if cx else 0
    py = cy.most_common(1)[0][0] % pitch if cy else 0
    return (px, py)


@dataclass(frozen=True)
class Grid:
    """The recovered logical geometry of a game's frames."""
    pitch: int
    phase: Cell

    @classmethod
    def infer(cls, arr: np.ndarray) -> "Grid":
        p = detect_pitch(arr)
        return cls(pitch=p, phase=detect_phase(arr, p))

    def cell_of_pixel(self, x: int, y: int) -> Cell:
        return ((x - self.phase[0]) // self.pitch, (y - self.phase[1]) // self.pitch)

    def majblock(self, cells: Sequence[Cell]) -> Cell:
        """The logical cell holding the MAJORITY of a component's pixels. This is
        the shape-invariant object-id: a sprite that rotates/animates within ~one
        cell keeps the same majblock."""
        blocks: Counter = Counter()
        for (x, y) in cells:
            blocks[self.cell_of_pixel(x, y)] += 1
        return blocks.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Connected components (8-connected) over a colour set
# ---------------------------------------------------------------------------

def components(arr: np.ndarray, colors) -> List[List[Cell]]:
    if isinstance(colors, int):
        colors = {colors}
    mask = np.isin(arr, list(colors))
    seen = np.zeros_like(mask, bool)
    H, W = arr.shape
    out: List[List[Cell]] = []
    for y in range(H):
        for x in range(W):
            if mask[y][x] and not seen[y][x]:
                stack = [(x, y)]; seen[y][x] = True; cells: List[Cell] = []
                while stack:
                    cx, cy = stack.pop(); cells.append((cx, cy))
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < W and 0 <= ny < H and mask[ny][nx] and not seen[ny][nx]:
                                seen[ny][nx] = True; stack.append((nx, ny))
                out.append(cells)
    return out


@dataclass(frozen=True)
class LObject:
    """A logical object: a colour at a logical cell. `size` is its pixel count
    (for size-based heuristics); its identity downstream is (colour, cell)."""
    color: int
    cell: Cell
    size: int


def objects(arr: np.ndarray, grid: Grid, colors: Optional[Sequence[int]] = None) -> List[LObject]:
    """Every non-background component, mapped to its majblock logical cell."""
    palette = colors if colors is not None else [c for c in range(1, 16) if (arr == c).any()]
    out: List[LObject] = []
    for c in palette:
        for comp in components(arr, c):
            out.append(LObject(color=int(c), cell=grid.majblock(comp), size=len(comp)))
    return out


def to_logical(arr: np.ndarray, grid: Grid) -> np.ndarray:
    """The frame as an NxN symbolic grid: each logical cell = the dominant
    non-background colour of its block (background if the block is empty)."""
    H, W = arr.shape
    p = grid.pitch; ox, oy = grid.phase
    nx = (W - ox + p - 1) // p
    ny = (H - oy + p - 1) // p
    out = np.zeros((ny, nx), dtype=int)
    for j in range(ny):
        for i in range(nx):
            block = arr[oy + j * p: oy + (j + 1) * p, ox + i * p: ox + (i + 1) * p]
            vals = [int(v) for v in block.flatten() if v != BACKGROUND]
            out[j][i] = Counter(vals).most_common(1)[0][0] if vals else BACKGROUND
    return out


def render_logical(lg: np.ndarray, mark: Optional[Cell] = None) -> str:
    """ASCII of a logical grid; '.' = background, hex digit = colour; mark in []."""
    rows = []
    for j in range(lg.shape[0]):
        cells = []
        for i in range(lg.shape[1]):
            ch = "." if lg[j][i] == BACKGROUND else format(int(lg[j][i]), "x")
            cells.append(f"[{ch}]" if mark == (i, j) else f" {ch} ")
        rows.append("".join(cells))
    return "\n".join(rows)
