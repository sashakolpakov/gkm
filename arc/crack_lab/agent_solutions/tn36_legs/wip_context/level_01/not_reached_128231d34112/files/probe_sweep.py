"""Clean chunked click sweep: <=CHUNK clone-probes per fresh env.

usage: python probe_sweep.py XLO XHI YLO YHI STRIDE
"""
import sys, hashlib
import numpy as np
sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

XLO, XHI, YLO, YHI, STRIDE = (int(v) for v in sys.argv[1:6])
CHUNK = 30
BASE_HASH = None


def cls(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m = (a != b); m[1, :] = False
    ys, xs = np.where(m)
    if not len(ys):
        return "-"
    return f"{len(ys)}@({ys.min()},{xs.min()})-({ys.max()},{xs.max()})"


pts = [(x, y) for x in range(XLO, XHI + 1, STRIDE) for y in range(YLO, YHI + 1, STRIDE)]
chunks = [pts[i:i + CHUNK] for i in range(0, len(pts), CHUNK)]
print(f"{len(pts)} points in {len(chunks)} chunks")

for ci, chunk in enumerate(chunks):
    def run(env, chunk=chunk, ci=ci):
        global BASE_HASH
        base = np.asarray(env.frame()).copy()
        hh = hashlib.md5(base.tobytes()).hexdigest()[:6]
        if BASE_HASH is None:
            BASE_HASH = hh
        elif hh != BASE_HASH:
            print(f"  !! chunk {ci} base hash {hh} != {BASE_HASH} (env not fresh)")
        for x, y in chunk:
            c = env.clone()
            c.step(6, x, y)
            k = cls(base, c.frame())
            if k != "-":
                print(f"  ({x:2d},{y:2d}) {k} lvl={c.levels_completed}")
    A.run_program("tn36", run)
print("done")
