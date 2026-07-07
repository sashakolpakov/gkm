import sys, json
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
env=A.Arena('ls20'); env.reset()
for a in json.load(open("checkpoint.json"))["final_path"]: env.step(a)
def av(f):
    ys,xs=np.where(f==9)
    m=(ys>=33)&(ys<=52)  # exclude fixed 9 at row26
    if m.sum()==0: return None
    return (int(ys[m].min()),int(xs[m].min()))
f0=env.frame(); print("base avatar TL(row,col):",av(f0))
# sequences to test heading/momentum
seqs={"1":[1],"11":[1,1],"111":[1,1,1],"3":[3],"33":[3,3],"333":[3,3,3],
      "13":[1,3],"31":[3,1],"1234":[1,2,3,4],"2":[2],"22":[2,2],"4":[4],"44":[4,4]}
for name,seq in seqs.items():
    c=env.clone(); pos=[av(c.frame())]
    for a in seq:
        try: c.step(a)
        except: pos.append("TERM"); break
        pos.append(av(c.frame()))
    print(f"seq {name:5s}: {pos}")
