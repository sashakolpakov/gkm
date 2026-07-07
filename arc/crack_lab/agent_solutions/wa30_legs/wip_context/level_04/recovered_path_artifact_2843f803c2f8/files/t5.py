import json, sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
ck = json.load(open("checkpoint.json"))
env = A.Arena('wa30')
for a in ck["final_path"]:
    env.step(a)
print("after prefix levels:", env.levels_completed)
# replay level3 moves step by step
import legs
moves = []
class Rec:
    def __init__(s, e): s.e=e
    def step(s,a):
        s.e.step(a); moves.append(a)
    def terminal(s): return s.e.terminal()
    @property
    def levels_completed(s): return s.e.levels_completed
r = Rec(env)
import players
# replicate play_level_3 but track when level increments
before = env.levels_completed
players.play_level_3(r)
print("total L3 moves:", len(moves))
# find minimal: replay fresh, step until level increments
env2 = A.Arena('wa30')
for a in ck["final_path"]:
    env2.step(a)
for i, a in enumerate(moves):
    env2.step(a)
    if env2.levels_completed > before:
        print("level 3 done after move index", i, "of", len(moves))
        break
f = np.array(env2.frame())
print("timer 7s at clean L4 start:", (f[63]==7).sum())
json.dump(moves[:i+1], open("l3_moves.json","w"))
