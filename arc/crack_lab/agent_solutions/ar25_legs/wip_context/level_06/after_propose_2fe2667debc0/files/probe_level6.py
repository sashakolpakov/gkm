import importlib.util
import os
import sys
from collections import Counter
from collections import deque

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from perception import action_deltas, color_counts, connected_components, frame_delta, replay


assert not G._workspace_taint_reason(os.getcwd())
spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


def probe(env):
    solver.solve(env)
    print("LEVEL", env.levels_completed + 1, "ACTIONS", tuple(env.actions))
    print("COLORS", color_counts(env.frame()))
    blobs = connected_components(env.frame(), colors=(4, 5, 9, 10, 11), min_area=2)
    print("BLOBS", [(b.color, b.bbox, b.area) for b in blobs])
    deltas = action_deltas(env, actions=tuple(env.actions))
    print("DELTAS", {a: (d["count"], d["bbox"]) for a, d in deltas.items()})
    f = np.asarray(env.frame())
    chars = {0: ".", 4: "A", 5: "p", 9: " ", 10: "x", 11: "#"}
    print("MAP3")
    for r in range(0, 63, 3):
        row = ""
        for c in range(0, 63, 3):
            vals, counts = np.unique(f[r:r+3, c:c+3], return_counts=True)
            row += chars[int(vals[np.argmax(counts)])]
        print(row.rstrip())
    for a in env.actions:
        child = env.clone()
        before = np.asarray(env.frame())
        child.step(a)
        after = np.asarray(child.frame())
        ys, xs = np.where(before != after)
        transitions = Counter((int(before[y, x]), int(after[y, x])) for y, x in zip(ys, xs))
        print("ACT", a, "L", child.levels_completed, "TRANS", sorted(transitions.items()))
    for seq in ([a] * 12 for a in env.actions):
        child = replay(env, seq)
        d = frame_delta(env.frame(), child.frame())
        print("REP", seq[0], child.levels_completed, d["count"], d["bbox"])
    for a in (1, 2, 3, 4):
        for seq in ([5, a] * 8, [5] + [a] * 12):
            child = replay(env, seq)
            d = frame_delta(env.frame(), child.frame())
            print("CTX", tuple(seq[:3]), len(seq), child.levels_completed, d["count"], d["bbox"])
    def sig(node):
        bs = connected_components(node.frame(), colors=(4, 5, 10), min_area=4)
        return [(b.color, b.bbox, b.area) for b in bs]
    tests = [
        [5], [5, 1], [5, 2], [5, 3], [5, 4],
        [5, 3, 3, 3], [5, 4, 4, 4],
        [5, 5], [5, 5, 1], [5, 5, 2], [5, 5, 3], [5, 5, 4],
        [2, 2, 2, 5], [2, 2, 2, 5, 3], [2, 2, 2, 5, 4],
    ]
    for seq in tests:
        child = replay(env, seq)
        print("SIG", tuple(seq), sig(child))

    target = np.asarray(env.frame()) == 11
    def search_piece(prefix, color):
        root = replay(env, prefix)
        q = deque([(root, [])])
        seen = set()
        scored = []
        while q and len(seen) < 240:
            node, path = q.popleft()
            nf = np.asarray(node.frame())
            mask = nf == color
            mask[63, :] = False
            key = mask.tobytes()
            if key in seen:
                continue
            seen.add(key)
            scored.append((int(np.count_nonzero(mask & target)),
                           int(np.count_nonzero(mask)), tuple(path)))
            if len(path) >= 16:
                continue
            for action in (1, 2, 3, 4):
                child = node.clone()
                child.step(action)
                q.append((child, path + [action]))
        print("SEARCH", color, len(seen), sorted(scored, reverse=True)[:12])

    search_piece([5], 4)
    search_piece([5, 5], 5)
    candidate = [5] + [4] * 5 + [5] + [2] * 4 + [3] * 10 + [5]
    node = replay(env, candidate)
    print("CAND_PRE", node.levels_completed, sig(node))
    for i in range(22):
        if node.levels_completed > env.levels_completed:
            break
        node.step(2)
        print("SCAN", i + 1, node.levels_completed)
    for prefix, action in (([5], 4), ([5, 5], 3), ([5, 5], 2)):
        before_node = replay(env, prefix)
        before = np.asarray(before_node.frame()).copy()
        before_node.step(action)
        after = np.asarray(before_node.frame())
        ys, xs = np.where(before != after)
        tr = Counter((int(before[y, x]), int(after[y, x])) for y, x in zip(ys, xs))
        print("MOVE_TRANS", tuple(prefix), action, sorted(tr.items()))
    for n in range(14):
        st = replay(env, [5] + [4] * n)
        sf = np.asarray(st.frame())
        composite = ((sf == 4) | (sf == 10))
        composite[:9, :] = False
        print("C1", n, int(np.count_nonzero(composite & target)),
              int(np.count_nonzero(composite)))
    for down in (1, 4):
        for left in (10, 15):
            seq = [5] + [4] * 5 + [5] + [2] * down + [3] * left + [5] + [2] * 22
            st = replay(env, seq)
            print("EXACT", down, left, st.levels_completed)
    hits = []
    for n in range(14):
        for down in (1, 4):
            for left in (10, 15):
                seq = [5] + [4] * n + [5] + [2] * down + [3] * left + [5] + [2] * 22
                st = replay(env, seq)
                if st.levels_completed > env.levels_completed:
                    hits.append((n, down, left, len(seq)))
    print("GRID_HITS", hits)
    for uses in range(0, 9):
        root = replay(env, [5] * uses)
        effects = []
        for action in (1, 2, 3, 4):
            child = root.clone()
            before = np.asarray(root.frame()).copy()
            child.step(action)
            after = np.asarray(child.frame())
            ys, xs = np.where(before != after)
            tr = Counter((int(before[y, x]), int(after[y, x])) for y, x in zip(ys, xs)
                         if x < 63)
            effects.append((action, len(ys), sorted(tr.items())))
        print("MODES", uses, effects)
    root3 = replay(env, [5, 5, 5])
    q = deque([(root3, [])])
    seen3 = set()
    scored3 = []
    while q and len(seen3) < 320:
        node, path = q.popleft()
        nf = np.asarray(node.frame())
        key = nf[:63, :63].tobytes()
        if key in seen3:
            continue
        seen3.add(key)
        colored = (nf == 4) | (nf == 5)
        colored[63, :] = False
        scored3.append((int(np.count_nonzero(colored & target)),
                        int(np.count_nonzero(colored)), tuple(path)))
        if len(path) >= 18:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            q.append((child, path + [action]))
    print("SEARCH3", len(seen3), sorted(scored3, reverse=True)[:20])
    c3_paths = [
        [2] * 12 + [3] * 5,
        [2] * 6 + [3] * 11,
        [2] * 6 + [3] * 5,
        [2] * 3 + [3] * 9,
        [3] * 5,
    ]
    hits3 = []
    for n in (3, 4, 5, 6):
        for down in (1, 4):
            for left in (10, 15):
                for p3 in c3_paths:
                    seq = ([5] + [4] * n + [5] + [2] * down + [3] * left
                           + [5] + p3 + [5] + [2] * 22)
                    st = replay(env, seq)
                    if st.levels_completed > env.levels_completed:
                        hits3.append((n, down, left, tuple(p3), len(seq)))
    print("HITS3", hits3)
    exhaustive_hits = []
    for down in (1, 4):
        for left in (10, 15):
            for _, _, p3 in scored3:
                seq = ([5] + [4] * 5 + [5] + [2] * down + [3] * left
                       + [5] + list(p3) + [5] + [2] * 22)
                st = replay(env, seq)
                if st.levels_completed > env.levels_completed:
                    exhaustive_hits.append((down, left, p3, len(seq)))
                    break
    print("EXHAUSTIVE_HITS", exhaustive_hits)


levels, path, err = A.run_program("ar25", probe)
print("END", levels, len(path), err)
