import sys, json, time
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
CK="checkpoint.json"
env=A.Arena('ls20'); env.reset()
for a in json.load(open(CK))["final_path"]: env.step(a)
t0=time.time()
N=2000
c=env.clone()
for i in range(N):
    cc=c.clone(); cc.step(1)
dt=time.time()-t0
print(f"{N} clone+step in {dt:.2f}s = {N/dt:.0f}/s")
