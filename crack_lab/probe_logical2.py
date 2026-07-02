"""Reconnaissance #2 (scratch): test the LOGICAL-CELL hypothesis on wa30.

Claim: if we track each object's position as a logical cell = round(centroid/pitch)
instead of pixels, the avatar (which rotates 4x3<->3x4) moves by a CLEAN, consistent
unit vector per action, and a logical-cell dynamics model hits much higher fidelity
than the 49% pixel model -- WITHOUT caring about the avatar's pixel shape.

Reports, over a random walk:
  - per-action avatar LOGICAL displacement distribution (should be ~1 vector each)
  - logical-cell fidelity: predict avatar+box logical cells on held-out transitions
    using LLM-free learned per-action deltas, at PIXEL vs LOGICAL resolution (control).
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
PITCH = int(sys.argv[2]) if len(sys.argv) > 2 else 4
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
                cx = sum(p[0] for p in cells) / len(cells)
                cy = sum(p[1] for p in cells) / len(cells)
                out.append(((cx, cy), frozenset(cells)))
    return out


def logical(cen, pitch):
    return (int(round(cen[0] / pitch)), int(round(cen[1] / pitch)))


def main():
    e = make_env(GAME)(); e.reset(); g0 = copy.deepcopy(e._env._game)
    fd0 = g0.perform_action(ActionInput(id=EA.RESET), raw=True)
    a0 = arr_of(fd0)
    struct = set(priors.structure_colours(a0.tolist()))
    avatar = 14
    box_colors = set(int(v) for v in np.unique(a0)
                     if v not in (0,) and v not in struct and v != avatar and v != 7)
    print(f"avatar={avatar} box_colors={sorted(box_colors)} struct={sorted(struct)} pitch={PITCH}")

    # random walk
    trans = []
    g = copy.deepcopy(g0); fd = g.perform_action(ActionInput(id=EA.RESET), raw=True)
    rng = random.Random(0)
    for _ in range(N):
        before = arr_of(fd); a = rng.choice([1, 2, 3, 4, 5])
        fd = g.perform_action(ActionInput(id=EA[NAME[a]]), raw=True)
        trans.append((before, a, arr_of(fd)))
        if str(fd.state).endswith(("GAME_OVER", "WIN")):
            g = copy.deepcopy(g0); fd = g.perform_action(ActionInput(id=EA.RESET), raw=True)

    # per-action avatar LOGICAL displacement distribution
    print("\n--- avatar logical displacement per action (cell vectors) ---")
    disp = defaultdict(Counter)
    for before, a, after in trans:
        bc = comps(before, {avatar}); ac = comps(after, {avatar})
        if bc and ac:
            bl = logical(bc[0][0], PITCH); al = logical(ac[0][0], PITCH)
            disp[a][(al[0] - bl[0], al[1] - bl[1])] += 1
    for a in sorted(disp):
        print(f"  ACTION{a}: {dict(disp[a].most_common())}")

    # fidelity control: pixel-centroid vs logical-cell
    for mode in ("pixel", "logical"):
        split = int(len(trans) * 0.7); train, test = trans[:split], trans[split:]
        def pos(cen):
            return logical(cen, PITCH) if mode == "logical" else (round(cen[0]), round(cen[1]))
        av_delta = defaultdict(Counter); box_rule = defaultdict(Counter)
        for before, a, after in train:
            bav = comps(before, {avatar}); aav = comps(after, {avatar})
            if bav and aav:
                d = (pos(aav[0][0])[0] - pos(bav[0][0])[0], pos(aav[0][0])[1] - pos(bav[0][0])[1])
                av_delta[a][d] += 1
            avc = pos(bav[0][0]) if bav else (0, 0)
            bb = comps(before, box_colors); ab = comps(after, box_colors)
            for (bc, _c) in bb:
                if not ab:
                    continue
                nc = min(ab, key=lambda t: abs(t[0][0] - bc[0]) + abs(t[0][1] - bc[1]))[0]
                rel = (pos(bc)[0] - avc[0], pos(bc)[1] - avc[1])
                disp_b = (pos(nc)[0] - pos(bc)[0], pos(nc)[1] - pos(bc)[1])
                box_rule[(a, rel)][disp_b] += 1
        av_delta = {a: c.most_common(1)[0][0] for a, c in av_delta.items()}
        box_rule = {k: c.most_common(1)[0][0] for k, c in box_rule.items()}
        ok = tot = 0
        for before, a, after in test:
            bav = comps(before, {avatar}); aav = comps(after, {avatar})
            if not bav or not aav:
                continue
            tot += 1
            avc = pos(bav[0][0])
            pav = (avc[0] + av_delta.get(a, (0, 0))[0], avc[1] + av_delta.get(a, (0, 0))[1])
            rav = pos(aav[0][0])
            pb = []
            for (bc, _c) in comps(before, box_colors):
                rel = (pos(bc)[0] - avc[0], pos(bc)[1] - avc[1])
                d = box_rule.get((a, rel), (0, 0))
                pb.append((pos(bc)[0] + d[0], pos(bc)[1] + d[1]))
            rb = sorted(pos(t[0]) for t in comps(after, box_colors))
            if pav == rav and sorted(pb) == rb:
                ok += 1
        print(f"\nFIDELITY [{mode}] held-out exact avatar+box: {ok}/{tot} ({100*ok//max(1,tot)}%)")


if __name__ == "__main__":
    main()
