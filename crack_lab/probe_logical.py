"""Reconnaissance ONLY (scratch): look at wa30's real frame at the pixel level so
we can decide the logical-cell quantizer + cofibrant object identity. Prints:
  - colour histogram of the start frame
  - the avatar (colour 14) footprint shape before/after each action (is it rigid?
    does it transform? how many *pixels* does it move?)
  - candidate logical grid pitch (the period at which colour-change boundaries
    repeat in x and y) -- if the renderer scales an NxN logical grid up to 64x64.
"""
from __future__ import annotations
import copy, sys
from collections import Counter
import numpy as np
from lab import make_env
from arcengine import ActionInput, GameAction as EA

NAME = {0: "RESET", 1: "ACTION1", 2: "ACTION2", 3: "ACTION3", 4: "ACTION4", 5: "ACTION5"}
GAME = sys.argv[1] if len(sys.argv) > 1 else "wa30"
AV = int(sys.argv[2]) if len(sys.argv) > 2 else 14


def arr_of(fd):
    return np.asarray(fd.frame[-1])


def shape_of(arr, color):
    ys, xs = np.where(arr == color)
    if len(xs) == 0:
        return None, None, None
    mnx, mny = int(xs.min()), int(ys.min())
    cells = frozenset((int(x) - mnx, int(y) - mny) for x, y in zip(xs, ys))
    return (mnx, mny), (int(xs.max()) - mnx + 1, int(ys.max()) - mny + 1), cells


def grid_pitch(arr):
    """Crude logical-pitch detector: for each axis, the most common spacing
    between consecutive 'colour changes' along the middle scanlines."""
    H, W = arr.shape
    def pitch(line):
        changes = [i for i in range(1, len(line)) if line[i] != line[i - 1]]
        gaps = [b - a for a, b in zip(changes, changes[1:])]
        return Counter(gaps)
    cx = Counter()
    for y in range(0, H, 4):
        cx += pitch(arr[y, :])
    cy = Counter()
    for x in range(0, W, 4):
        cy += pitch(arr[:, x])
    return cx.most_common(5), cy.most_common(5)


def main():
    e = make_env(GAME)(); e.reset(); g0 = copy.deepcopy(e._env._game)
    fd = g0.perform_action(ActionInput(id=EA.RESET), raw=True)
    a0 = arr_of(fd)
    print(f"=== {GAME} start frame {a0.shape} ===")
    hist = Counter(int(v) for v in a0.flatten())
    print("colour histogram:", dict(sorted(hist.items(), key=lambda t: -t[1])))
    px, py = grid_pitch(a0)
    print("x pitch (gap:count):", px)
    print("y pitch (gap:count):", py)

    pos0, dim0, sh0 = shape_of(a0, AV)
    print(f"\navatar(colour {AV}) start: corner={pos0} bbox(w,h)={dim0} npix={len(sh0) if sh0 else 0}")

    for a in (1, 2, 3, 4, 5):
        gc = copy.deepcopy(g0)
        gc.perform_action(ActionInput(id=EA.RESET), raw=True)
        fd1 = gc.perform_action(ActionInput(id=EA[NAME[a]]), raw=True)
        a1 = arr_of(fd1)
        pos1, dim1, sh1 = shape_of(a1, AV)
        if pos1 is None:
            print(f"ACTION{a}: avatar VANISHED (transforms colour?)")
            continue
        dpix = (pos1[0] - pos0[0], pos1[1] - pos0[1]) if pos0 else None
        same_shape = (sh0 == sh1)
        changed = int((a1 != a0).sum())
        print(f"ACTION{a}: corner_delta(px)={dpix} bbox={dim1} npix={len(sh1)} "
              f"rigid_shape={same_shape} changed_cells={changed}")


if __name__ == "__main__":
    main()
