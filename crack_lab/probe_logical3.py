"""Reconnaissance #3 (scratch): which OBJECT-ID scheme makes wa30's logical
dynamics faithful? Tests several ways to map a pixel component -> a logical cell,
detects the grid PHASE, and breaks fidelity into avatar-only vs box-only so we see
where the 49% is lost.

Object-id schemes (all at pitch P, phase (px,py)):
  centroid : round((cen-phase)/P)              [current, ~53%]
  majblock : the PxP block (offset by phase) holding the MOST of the component's px
  minsnap  : round((min_corner-phase)/P)
"""
from __future__ import annotations
import copy, random, sys
from collections import defaultdict, Counter
import numpy as np
from lab import make_env
from arcengine import ActionInput, GameAction as EA
import priors

NAME = {0: "RESET", 1: "ACTION1", 2: "ACTION2", 3: "ACTION3", 4: "ACTION4", 5: "ACTION5"}
GAME = sys.argv[1] if len(sys.argv) > 1 else "wa30"
P = int(sys.argv[2]) if len(sys.argv) > 2 else 4
N = int(sys.argv[3]) if len(sys.argv) > 3 else 400


def arr_of(fd):
    return np.asarray(fd.frame[-1])


def comps(arr, colors):
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
                out.append(cells)
    return out


def detect_phase(arr, P):
    """Grid phase = the (px,py) in [0,P) where most colour-change boundaries land."""
    H, W = arr.shape
    cx = Counter(); cy = Counter()
    for y in range(H):
        for x in range(1, W):
            if arr[y][x] != arr[y][x - 1]:
                cx[x % P] += 1
    for x in range(W):
        for y in range(1, H):
            if arr[y][x] != arr[y - 1][x]:
                cy[y % P] += 1
    return (cx.most_common(1)[0][0] % P, cy.most_common(1)[0][0] % P)


def make_pos(scheme, P, phase):
    px, py = phase
    def centroid(cells):
        cx = sum(p[0] for p in cells) / len(cells); cy = sum(p[1] for p in cells) / len(cells)
        return (int(round((cx - px) / P)), int(round((cy - py) / P)))
    def minsnap(cells):
        return (int(round((min(p[0] for p in cells) - px) / P)),
                int(round((min(p[1] for p in cells) - py) / P)))
    def majblock(cells):
        blocks = Counter()
        for (x, y) in cells:
            blocks[((x - px) // P, (y - py) // P)] += 1
        return blocks.most_common(1)[0][0]
    return {"centroid": centroid, "minsnap": minsnap, "majblock": majblock}[scheme]


def main():
    e = make_env(GAME)(); e.reset(); g0 = copy.deepcopy(e._env._game)
    fd0 = g0.perform_action(ActionInput(id=EA.RESET), raw=True)
    a0 = arr_of(fd0)
    struct = set(priors.structure_colours(a0.tolist()))
    avatar = 14
    box_colors = set(int(v) for v in np.unique(a0)
                     if v not in (0,) and v not in struct and v != avatar and v != 7)
    phase = detect_phase(a0, P)
    print(f"avatar={avatar} box_colors={sorted(box_colors)} struct={sorted(struct)} pitch={P} phase={phase}")

    trans = []
    g = copy.deepcopy(g0); fd = g.perform_action(ActionInput(id=EA.RESET), raw=True)
    rng = random.Random(0)
    for _ in range(N):
        before = arr_of(fd); a = rng.choice([1, 2, 3, 4, 5])
        fd = g.perform_action(ActionInput(id=EA[NAME[a]]), raw=True)
        trans.append((before, a, arr_of(fd)))
        if str(fd.state).endswith(("GAME_OVER", "WIN")):
            g = copy.deepcopy(g0); fd = g.perform_action(ActionInput(id=EA.RESET), raw=True)
    split = int(len(trans) * 0.7); train, test = trans[:split], trans[split:]

    for scheme in ("centroid", "minsnap", "majblock"):
        pos = make_pos(scheme, P, phase)
        def cells_pos(arr, colors):
            return [pos(c) for c in comps(arr, colors)]
        # avatar displacement distribution + learn deltas
        av_delta = defaultdict(Counter); box_rule = defaultdict(Counter)
        for before, a, after in train:
            ba = comps(before, {avatar}); aa = comps(after, {avatar})
            if ba and aa:
                b = pos(ba[0]); n = pos(aa[0])
                av_delta[a][(n[0] - b[0], n[1] - b[1])] += 1
            avc = pos(ba[0]) if ba else (0, 0)
            bb = comps(before, box_colors); ab = [pos(c) for c in comps(after, box_colors)]
            for bc_cells in bb:
                bc = pos(bc_cells)
                if not ab:
                    continue
                nc = min(ab, key=lambda t: abs(t[0] - bc[0]) + abs(t[1] - bc[1]))
                box_rule[(a, (bc[0] - avc[0], bc[1] - avc[1]))][(nc[0] - bc[0], nc[1] - bc[1])] += 1
        av_d = {a: c.most_common(1)[0][0] for a, c in av_delta.items()}
        box_r = {k: c.most_common(1)[0][0] for k, c in box_rule.items()}
        # eval: avatar-only, box-only, joint
        av_ok = box_ok = joint_ok = tot = 0
        for before, a, after in test:
            ba = comps(before, {avatar}); aa = comps(after, {avatar})
            if not ba or not aa:
                continue
            tot += 1
            avc = pos(ba[0])
            pav = (avc[0] + av_d.get(a, (0, 0))[0], avc[1] + av_d.get(a, (0, 0))[1])
            rav = pos(aa[0]); a_ok = (pav == rav)
            pb = []
            for bc_cells in comps(before, box_colors):
                bc = pos(bc_cells); d = box_r.get((a, (bc[0] - avc[0], bc[1] - avc[1])), (0, 0))
                pb.append((bc[0] + d[0], bc[1] + d[1]))
            rb = sorted(cells_pos(after, box_colors)); b_ok = (sorted(pb) == rb)
            av_ok += a_ok; box_ok += b_ok; joint_ok += (a_ok and b_ok)
        print(f"\n[{scheme}] avatar-only {100*av_ok//max(1,tot)}%  box-only {100*box_ok//max(1,tot)}%  "
              f"joint {100*joint_ok//max(1,tot)}%  (n={tot})")
        print(f"   learned avatar deltas: {av_d}")


if __name__ == "__main__":
    main()
