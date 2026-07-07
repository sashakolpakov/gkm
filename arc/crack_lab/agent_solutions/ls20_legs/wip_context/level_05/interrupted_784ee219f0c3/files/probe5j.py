import sys, json
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
env=A.Arena('ls20'); env.reset()
for a in json.load(open("checkpoint.json"))["final_path"]: env.step(a)
f0=env.frame()
def bar(f): return int((f[60:63]==11).sum())
print("base bar(col11 rows60-62):",bar(f0), "total col11:",int((f0==11).sum()))
# drain per action
for act in (1,2,3,4):
    c=env.clone(); f=c.step(act)
    print(f"act{act}: bar={bar(f)} totalB={int((f==11).sum())}")
# how many action-1 moves until terminal?
c=env.clone(); n=0
while not c.terminal() and n<400:
    try: c.step(1)
    except Exception: break
    n+=1
print("action1 until terminal/frame-gone:",n, "levels:",c.levels_completed)
