import sys
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from collections import deque
env=A.Arena('g50t'); key=lambda e:e.frame().tobytes()
q=deque([env.clone()]); seen={key(env)}
frame_to_env={}
while q:
    n=q.popleft()
    for a in (1,2,3,4,5):
        c=n.clone(); c.step(a); k=key(c)
        if k in seen: continue
        seen.add(k); q.append(c); frame_to_env[k]=c
# gates-open via [4,4,4,4,5]
e1=A.Arena('g50t')
for a in [4,4,4,4,5]: e1.step(a)
k1=key(e1)
print("f1 in seen:",k1 in seen)
# expand stored representative of f1
rep=frame_to_env.get(k1)
if rep is not None:
    d=rep.clone(); d.step(2)  # DOWN
    print("rep DOWN reach frame in seen:",key(d) in seen)
    # what avatar reach does rep have vs e1
    import perception as P
    def av(e):
        bl=[b for b in P.connected_components(e.frame(),colors=[9]) if b.area==24]
        return bl[0].bbox[:2] if bl else None
    print("rep av",av(rep),"e1 av",av(e1))
    print("rep frame == e1 frame:",(rep.frame()==e1.frame()).all())
    # DOWN from e1
    d2=e1.clone(); d2.step(2)
    print("e1 DOWN == rep DOWN:",(d2.frame()==d.frame()).all())
    print("e1 DOWN frame in seen:",key(d2) in seen)
