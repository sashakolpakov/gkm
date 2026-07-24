import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np

def main(env):
    base = np.asarray(env.frame()).copy()
    tests = [
        ("step(1)", lambda e: e.step(1)),
        ("step(6)", lambda e: e.step(6)),
        ("step(6,13,19)", lambda e: e.step(6,13,19)),
        ("step((6,13,19))", lambda e: e.step((6,13,19))),
        ("step([6,13,19])", lambda e: e.step([6,13,19])),
        ("step(6,(13,19))", lambda e: e.step(6,(13,19))),
    ]
    for name,fn in tests:
        c = env.clone()
        try:
            fn(c)
            diff = int((np.asarray(c.frame())!=base).sum())
            print(name, "OK diff=", diff, "level=", c.levels_completed)
        except Exception as e:
            print(name, "ERR", repr(e)[:80])
A.run_program('sc25', main)
