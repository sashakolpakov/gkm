"""Schema-free, data-driven dynamics learning + FIDELITY GATE (no hand-coded
physics, no 'Sokoban' assumption). Learn wa30's transition rule purely from
observed (frame, action, next-frame) data, then verify it predicts held-out
transitions. If faithful -> a fast model to plan in; if not -> definitive that
L2 is beyond reach under the constraints.

Rule form (general, not push-specific): the avatar translates by a learned
per-action delta; every OTHER movable object's displacement is a learned
function of (action, the object's offset from the avatar). No assumption about
WHY it moves — just what the data says. Walls = cells the avatar never enters.
"""
from __future__ import annotations
import copy, random, sys
from collections import defaultdict, Counter
import numpy as np
from lab import arc, make_env
from arcengine import ActionInput, GameAction as EA
import priors

NAME = {0: "RESET", 1: "ACTION1", 2: "ACTION2", 3: "ACTION3", 4: "ACTION4", 5: "ACTION5"}


def step(g, a):
    gc = copy.deepcopy(g)
    return gc, gc.perform_action(ActionInput(id=EA[NAME[a]]), raw=True)


def arr_of(fd):
    return np.asarray(fd.frame[-1])


def comps(arr, colors):
    """8-connected components over a set of colours; return list of (centroid, frozenset cells)."""
    mask = np.isin(arr, list(colors))
    seen = np.zeros_like(mask, bool); H, W = arr.shape; out = []
    for y in range(H):
        for x in range(W):
            if mask[y][x] and not seen[y][x]:
                st = [(x, y)]; seen[y][x] = True; cells = []
                while st:
                    cx, cy = st.pop(); cells.append((cx, cy))
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < W and 0 <= ny < H and mask[ny][nx] and not seen[ny][nx]:
                                seen[ny][nx] = True; st.append((nx, ny))
                # ANCHOR = min-corner: exact under translation (no centroid rounding noise)
                ax = min(p[0] for p in cells); ay = min(p[1] for p in cells)
                out.append(((ax, ay), frozenset(cells)))
    return out


def main():
    N_TRANS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    e = make_env("wa30")(); s = e.reset(); g0 = e._env._game
    a0 = arr_of(g0.perform_action(ActionInput(id=EA.RESET), raw=True) if False else
               copy.deepcopy(g0).perform_action(ActionInput(id=EA.RESET), raw=True))
    struct = set(priors.structure_colours(a0.tolist()))
    avatar = 14
    box_colors = set(int(v) for v in np.unique(a0) if v not in (0,) and v not in struct and v != avatar and v != 7)
    print(f"avatar={avatar} box_colors={sorted(box_colors)} structure={sorted(struct)}")

    # ---- collect transitions (random walks) ----
    trans = []  # (before_arr, action, after_arr)
    g = copy.deepcopy(g0)
    fd = g.perform_action(ActionInput(id=EA.RESET), raw=True)
    rng = random.Random(0)
    for _ in range(N_TRANS):
        before = arr_of(fd)
        a = rng.choice([1, 2, 3, 4, 5])
        fd = g.perform_action(ActionInput(id=EA[NAME[a]]), raw=True)
        after = arr_of(fd)
        trans.append((before, a, after))
        if str(fd.state).endswith(("GAME_OVER", "WIN")):
            g = copy.deepcopy(g0); fd = g.perform_action(ActionInput(id=EA.RESET), raw=True)

    split = int(len(trans) * 0.7)
    train, test = trans[:split], trans[split:]

    # ---- learn avatar per-action delta + box-displacement rule ----
    av_delta = {}
    box_rule = defaultdict(Counter)   # (action, rel_offset_bucket) -> Counter(box_displacement)
    def cen(arr, colors):
        c = comps(arr, colors); return c
    for before, a, after in train:
        bav = comps(before, {avatar}); aav = comps(after, {avatar})
        if bav and aav:
            d = (aav[0][0][0] - bav[0][0][0], aav[0][0][1] - bav[0][0][1])
            av_delta.setdefault(a, Counter())[d] += 1
        # boxes: match before->after by nearest centroid (same colour set)
        bb = comps(before, box_colors); ab = comps(after, box_colors)
        avc = bav[0][0] if bav else (0, 0)
        for (bc, _bcells) in bb:
            # nearest after-box
            if not ab:
                continue
            nc = min(ab, key=lambda t: abs(t[0][0] - bc[0]) + abs(t[0][1] - bc[1]))[0]
            disp = (nc[0] - bc[0], nc[1] - bc[1])
            rel = (bc[0] - avc[0], bc[1] - avc[1])
            box_rule[(a, rel)][disp] += 1
    av_delta = {a: c.most_common(1)[0][0] for a, c in av_delta.items()}
    box_rule = {k: c.most_common(1)[0][0] for k, c in box_rule.items()}
    print(f"learned avatar deltas: {av_delta}")
    print(f"learned box-rule entries: {len(box_rule)}")

    # ---- fidelity on held-out: predict avatar + box centroids ----
    ok = 0; tot = 0
    for before, a, after in test:
        bav = comps(before, {avatar}); aav = comps(after, {avatar})
        if not bav or not aav:
            continue
        tot += 1
        pred_av = (bav[0][0][0] + av_delta.get(a, (0, 0))[0], bav[0][0][1] + av_delta.get(a, (0, 0))[1])
        real_av = aav[0][0]
        avc = bav[0][0]
        bb = comps(before, box_colors)
        pred_boxes = []
        for (bc, _c) in bb:
            rel = (bc[0] - avc[0], bc[1] - avc[1])
            d = box_rule.get((a, rel), (0, 0))
            pred_boxes.append((bc[0] + d[0], bc[1] + d[1]))
        real_boxes = sorted(t[0] for t in comps(after, box_colors))
        if pred_av == real_av and sorted(pred_boxes) == real_boxes:
            ok += 1
    print(f"\nFIDELITY (held-out, exact avatar+box centroids): {ok}/{tot} ({100*ok//max(1,tot)}%)")


if __name__ == "__main__":
    main()
