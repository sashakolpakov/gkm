import sys, json
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P

ck = json.load(open("checkpoint.json"))
env = A.Arena('wa30')
for a in ck["final_path"]:
    env.step(a)
print("levels_completed after checkpoint:", env.levels_completed)
print("terminal:", env.terminal())
f = env.frame()
print("shape", f.shape)
print("colors", P.color_counts(f))
