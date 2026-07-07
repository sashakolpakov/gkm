import json, sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np

ck = json.load(open("checkpoint.json"))
PREFIX = ck["final_path"]

def get_env(extra=()):
    env = A.Arena('wa30')
    for a in PREFIX:
        env.step(a)
    assert env.levels_completed == 4, env.levels_completed
    for a in extra:
        env.step(a)
    return env

CH = '.1234567890abcdef'
def show(f):
    f = np.array(f)
    for r in f:
        print(''.join(CH[int(v)] if int(v)<16 else '?' for v in r))

def diff(f1, f2):
    f1, f2 = np.array(f1), np.array(f2)
    ys, xs = np.where(f1 != f2)
    for y, x in zip(ys, xs):
        print(f"({y},{x}): {f1[y,x]} -> {f2[y,x]}")

if __name__ == '__main__':
    env = get_env()
    print("levels:", env.levels_completed, "terminal:", env.terminal())
    show(env.frame())
