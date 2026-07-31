import json, sys, time, traceback
from collections import deque
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from perception import arr

PATH = json.load(open('checkpoint.json'))['final_path']
UP, DOWN, LEFT, RIGHT, USE = 1, 2, 3, 4, 5
RECT = [UP]*3 + [RIGHT]*4 + [UP]*8 + [RIGHT]*9 + [DOWN]*10


def key(e):
    return arr(e.frame())[:63].tobytes()


def bfs(root, base_level, max_states, max_depth, acts=(UP, DOWN, LEFT, RIGHT)):
    t0 = time.time()
    seen = {key(root)}
    q = deque([(root.clone(), [])])
    steps = 0
    while q and len(seen) <= max_states:
        node, path = q.popleft()
        if len(path) >= max_depth:
            continue
        for a in acts:
            child = node.clone()
            try:
                child.step(a)
                k = key(child)
            except Exception:
                continue
            steps += 1
            if k in seen:
                continue
            seen.add(k)
            if child.levels_completed > base_level:
                print(f'  SOLVED depth={len(path)+1} states={len(seen)} '
                      f'steps={steps} t={time.time()-t0:.1f}s')
                return path + [a]
            q.append((child, path + [a]))
    print(f'  no solution: states={len(seen)} steps={steps} t={time.time()-t0:.1f}s')
    return None


def prog(env):
  try:
    for a in PATH:
        env.step(a)
    c = env.clone()
    for a in RECT:
        c.step(a)
    c.step(USE)   # select the cross
    print('rectangle placed, cross selected; lvl', c.levels_completed)
    sol = bfs(c, 5, max_states=60000, max_depth=64)
    print('cross plan:', sol)
    if sol:
        print('total moves for level 6:', len(RECT) + 1 + len(sol))
  except Exception:
    traceback.print_exc()


A.run_program('re86', prog)
