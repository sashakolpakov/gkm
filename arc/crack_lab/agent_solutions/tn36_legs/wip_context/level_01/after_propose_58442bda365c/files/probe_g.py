"""Decisive sibling-clone isolation test, budget row INCLUDED."""
import sys
import numpy as np
sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A


def budget(f):
    row = np.asarray(f)[1, 1:62]
    return int((row == 9).sum())


def glyphs(f):
    f = np.asarray(f)
    return tuple(int(f[44, c]) for c in (21, 26, 31, 36, 41))


def st(tag, e):
    f = np.asarray(e.frame())
    print(f"  {tag:>34s} budget={budget(f):2d} glyphs={glyphs(f)}")


def run(env):
    st("root before", env)
    c1 = env.clone()
    c1.step(6, 20, 44)          # one glyph toggle
    st("c1 after 1 toggle", c1)
    st("root after c1 step", env)
    c2 = env.clone()
    st("c2 (made after c1 stepped)", c2)
    c1.step(6, 24, 44)
    st("c1 after 2nd toggle", c1)
    st("c2 again", c2)
    st("root again", env)
    # deep chain
    d1 = env.clone(); d1.step(6, 20, 44); d1.step(6, 24, 44); d1.step(6, 28, 44)
    st("d1 after 3 toggles", d1)
    d2 = d1.clone()
    st("d2 = d1.clone()", d2)
    d1.step(6, 32, 44)
    st("d1 after 4th", d1)
    st("d2 after d1 4th", d2)


A.run_program("tn36", run)
