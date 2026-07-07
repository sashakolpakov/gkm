import sys, json, time, copy
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
env=A.Arena('ls20'); env.reset()
for a in json.load(open("checkpoint.json"))["final_path"]: env.step(a)
t0=time.time()
for i in range(300): _=env.clone()
print(f"pure clone: {300/(time.time()-t0):.0f}/s")
t0=time.time()
for i in range(300): _=copy.deepcopy(env._game)
print(f"deepcopy _game: {300/(time.time()-t0):.0f}/s")
# step on fresh clones (clone then 1 step)
t0=time.time()
for i in range(300):
    c=env.clone(); c.step(1)
print(f"clone+1step: {300/(time.time()-t0):.0f}/s")
